import os
import numpy as np
import torch
import torch.utils
import torchvision
import argparse
import yaml

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
import torch.utils.data

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import get_resnet_adaptable
from simclr.modules.nt_xent_hard_select import NT_Xent
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.dataset = args.dataset.upper()
    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            train=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR100": 
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "IMAGENET32":
        from utils.imagenet32 import ImageNet32

        train_dataset = ImageNet32(
            root=args.dataset_dir,
            train=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "IMAGENET":
        train_dataset =  ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet_adaptable(args.resnet, args.dataset, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    # Loss policies are implemented inside NT_Xent. In addition to the existing
    # anchor-based models, we support the new per-anchor logit transform policy
    # "tlambda_select" (see nt_xent_hard_select.py).
    criterion = NT_Xent(
        batch_size=args.batch_size,
        temperature=args.temperature,
        world_size=args.world_size,
        policy=args.policy,
        weib_beta=args.weib_beta,
        soft_lambda=args.soft_lambda,
        auto_weib_beta=args.auto_weib_beta,
        use_weib_topm=args.use_weib_topm,
        weib_lambda=args.weib_lambda,
        weib_top_m=args.weib_top_m,
        use_pl_topm=args.use_pl_topm,
        pl_top_m=args.pl_top_m,
        pl_lambda=args.pl_lambda,

        # (4) gate policy
        gate_rho0=args.gate_rho0,
        gate_scale=args.gate_scale,
        gate_r2_min=args.gate_r2_min,

        # (5) per-anchor shrink policy
        anchor_tail_frac=args.anchor_tail_frac,
        anchor_min_tail_points=args.anchor_min_tail_points,
        anchor_r2_min=args.anchor_r2_min,
        shrink_c=args.shrink_c,

        # (6) hard_select policy
        select_q=args.select_q,
        select_aic_margin=args.select_aic_margin,
        select_tail_frac=args.select_tail_frac,
        select_min_tail_points=args.select_min_tail_points,
        select_weib_top_m=args.select_weib_top_m,
        select_kappa_rho=args.select_kappa_rho,
        select_kappa_aic=args.select_kappa_aic,
    )

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if (
            args.nr == 0
            and args.checkpoint_every > 0
            and epoch % args.checkpoint_every == 0
            and epoch != 0
        ):
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # -------------------------------
    # Extra CLI args (not always present in config.yaml)
    # -------------------------------
    def _add_if_missing(name, default, typ):
        if not any(a.dest == name for a in parser._actions):
            parser.add_argument(f"--{name}", default=default, type=typ)

    # mix / EVT policies
    _add_if_missing("soft_lambda", 0.5, float)

    # (4) anchor-wise gating policy="gate"
    _add_if_missing("gate_rho0", 0.05, float)
    _add_if_missing("gate_scale", 5.0, float)
    _add_if_missing("gate_r2_min", 0.0, float)

    # (5) per-anchor beta shrink policy="weibit_shrink"
    _add_if_missing("anchor_tail_frac", 0.10, float)
    _add_if_missing("anchor_min_tail_points", 32, int)
    _add_if_missing("anchor_r2_min", 0.0, float)
    _add_if_missing("shrink_c", 100.0, float)

    # (6) hard_select policy (binary per-anchor switch)
    _add_if_missing("select_q", 0.2, float)
    _add_if_missing("select_aic_margin", 2.0, float)
    _add_if_missing("select_tail_frac", 0.10, float)
    _add_if_missing("select_min_tail_points", 32, int)
    _add_if_missing("select_weib_top_m", 16, int)
    _add_if_missing("select_kappa_rho", 10.0, float)
    _add_if_missing("select_kappa_aic", 1.0, float)

    args = parser.parse_args()

    # save the final configuration
    with open(os.path.join(args.model_path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)