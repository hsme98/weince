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

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet_adaptable
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
    criterion = NT_Xent(
        batch_size=args.batch_size,
        temperature=args.temperature,
        world_size=args.world_size,
        policy=args.policy,
        weib_beta=args.weib_beta,
        auto_weib_beta=args.auto_weib_beta,
        use_weib_topm=args.use_weib_topm,
        weib_lambda=args.weib_lambda,
        weib_top_m=args.weib_top_m,
        use_pl_topm=args.use_pl_topm,
        pl_top_m=args.pl_top_m,
        pl_lambda=args.pl_lambda,
        # EVT top-M regularizer variants
        weib_topm_mode=args.weib_topm_mode,
        weib_gate_kappa=args.weib_gate_kappa,
        weib_gate_q=args.weib_gate_q,
        weib_gate_mix=args.weib_gate_mix,
        weib_gate_hard_threshold=args.weib_gate_hard_threshold,
        weib_shrink_k=args.weib_shrink_k,
        weib_shrink_alpha=args.weib_shrink_alpha,
        weib_shrink_r2_min=args.weib_shrink_r2_min,

        # New lambda estimation options
        weib_gate_estimator=args.weib_gate_estimator,
        weib_gate_knn_k=args.weib_gate_knn_k,
        weib_gate_knn_eta=args.weib_gate_knn_eta,
        weib_gate_knn_temp=args.weib_gate_knn_temp,
        weib_gate_mlp_hidden=args.weib_gate_mlp_hidden,
        weib_gate_mlp_detach=args.weib_gate_mlp_detach,
        weib_gate_mlp_lambda_max=args.weib_gate_mlp_lambda_max,
        gate_embed_dim=args.projection_dim,
        weib_gate_smooth_weight=args.weib_gate_smooth_weight,
        weib_gate_target_mean=args.weib_gate_target_mean,
        weib_gate_mean_weight=args.weib_gate_mean_weight,
        weib_gate_entropy_weight=args.weib_gate_entropy_weight,
        weib_gate_distill_weight=args.weib_gate_distill_weight,
        weib_gate_aic_tail_k=args.weib_gate_aic_tail_k,
    )

    # If criterion has learnable parameters (e.g., a lambda head), move it to device
    criterion = criterion.to(args.device)

    # Reload criterion state if resuming
    if args.reload:
        crit_fp = os.path.join(args.model_path, f"criterion_{args.epoch_num}.pt")
        if not os.path.exists(crit_fp):
            crit_fp = os.path.join(args.model_path, "criterion_final.pt")
        if os.path.exists(crit_fp):
            criterion.load_state_dict(torch.load(crit_fp, map_location=args.device))
            if args.nr == 0:
                print(f"Loaded criterion state from {crit_fp}")

    # Include criterion parameters in the optimizer (needed when using mlp gate)
    crit_params = [p for p in criterion.parameters() if p.requires_grad]
    if len(crit_params) > 0:
        optimizer.add_param_group({"params": crit_params})

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
            # Save criterion state separately (for lambda-head training)
            torch.save(criterion.state_dict(), os.path.join(args.model_path, f"criterion_{epoch}.pt"))

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
    if args.nr == 0:
        torch.save(criterion.state_dict(), os.path.join(args.model_path, "criterion_final.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))


    # --- Extra CLI args for EVT top-M regularizer variants (not in config.yaml by default) ---
    def _maybe_add_arg(name, default, type_, **kwargs):
        opt = f"--{name}"
        existing = {o for a in parser._actions for o in a.option_strings}
        if opt not in existing:
            parser.add_argument(opt, default=default, type=type_, **kwargs)

    _maybe_add_arg("weib_topm_mode", "weib", str, choices=["weib", "gate", "gate_weib", "shrink", "shrink_gate"],
                   help="Top-M regularizer mode: weib (batch), gate (PL<->Weib mix), gate_weib (Weib-only weighting), shrink (anchor-wise beta), shrink_gate (shrink + weighting).")
    _maybe_add_arg("weib_gate_kappa", 10.0, float, help="Gate steepness for weib_topm_mode=gate.")
    _maybe_add_arg("weib_gate_q", 0.2, float, help="Gate pivot quantile on rho for weib_topm_mode=gate.")
    _maybe_add_arg(
        "weib_gate_mix",
        "loss",
        str,
        choices=["loss", "logit", "hard_loss", "hard_logit"],
        help="Mixing rule inside gate regularizer: soft mix (loss/logit) or hard selection (hard_*).",
    )
    _maybe_add_arg(
        "weib_gate_hard_threshold",
        0.5,
        float,
        help="Threshold on lambda for hard_* mixes: if lambda > thr use Weibit else PL (with ST gradients).",
    )
    _maybe_add_arg("weib_shrink_k", 64, int, help="Top-k negatives used for anchor beta in weib_topm_mode=shrink.")
    _maybe_add_arg("weib_shrink_alpha", 0.5, float, help="Shrink weight toward anchor beta_hat in shrink mode.")
    _maybe_add_arg("weib_shrink_r2_min", 0.2, float, help="Min R^2 to trust anchor beta_hat in shrink mode.")

    # --- New: lambda_i estimators for gate modes ---
    _maybe_add_arg(
        "weib_gate_estimator",
        "rho",
        str,
        choices=["rho", "rho_knn", "mlp", "mlp_knn", "aic_knn", "mlp_distill_aic_knn"],
        help="How to estimate anchor-wise gate lambda_i in gate-based top-M modes.",
    )
    _maybe_add_arg("weib_gate_knn_k", 8, int, help="k for kNN smoothing of lambda_i (for *_knn estimators).")
    _maybe_add_arg("weib_gate_knn_eta", 0.5, float, help="Smoothing strength for kNN lambda smoothing.")
    _maybe_add_arg("weib_gate_knn_temp", 0.1, float, help="Softmax temperature for kNN weights.")
    _maybe_add_arg("weib_gate_mlp_hidden", 128, int, help="Hidden width for the learned lambda head.")
    _maybe_add_arg("weib_gate_mlp_detach", True, bool, help="Detach embeddings/stats before lambda head to prevent gaming.")
    _maybe_add_arg("weib_gate_mlp_lambda_max", 1.0, float, help="Max lambda scaling for learned gate (<=1).")

    _maybe_add_arg("weib_gate_smooth_weight", 0.0, float, help="Weight of kNN smoothness penalty for learned gate.")
    _maybe_add_arg("weib_gate_target_mean", -1.0, float, help="Target mean(lambda). Set <0 to disable.")
    _maybe_add_arg("weib_gate_mean_weight", 0.0, float, help="Weight for mean(lambda) prior penalty.")
    _maybe_add_arg("weib_gate_entropy_weight", 0.0, float, help="Entropy bonus weight for learned lambda.")

    _maybe_add_arg("weib_gate_distill_weight", 0.0, float, help="MSE distillation weight (mlp_distill_aic_knn).")
    _maybe_add_arg("weib_gate_aic_tail_k", 0, int, help="Tail length used by AIC teacher (0 uses weib_shrink_k).")
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