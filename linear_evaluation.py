import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
import torch.utils.data
import yaml

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet_adaptable
from simclr.modules.transformations import TransformsSimCLR

from torchvision.datasets import ImageFolder
from utils import yaml_config_hook


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    seed=1,
    val_ratio=0.2,
    shuffle_train=True,
):
    if val_ratio > 0:
        # stratified split of the training set into train and val
        val_idxs = []
        train_idxs = []
        rng = np.random.default_rng(seed=seed)
        for c in np.unique(y_train):
            idx = np.where(y_train == c)[0]
            rng.shuffle(idx)
            n_val = int(val_ratio * len(idx))
            val_idxs.extend(idx[:n_val].tolist())
            train_idxs.extend(idx[n_val:].tolist())
        rng.shuffle(val_idxs)
        rng.shuffle(train_idxs)

        train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train[train_idxs, :]),
            torch.from_numpy(y_train[train_idxs]),
        )
        val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train[val_idxs, :]), torch.from_numpy(y_train[val_idxs])
        )

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle_train
        )
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False
        )
    else:
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=shuffle_train
        )

        val_loader = None

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def get_dataloader(args: argparse.Namespace):
    args.dataset = args.dataset.upper()
    extra_vars = {}
    if args.dataset == "STL10":
        extra_vars["n_classes"] = 10
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        extra_vars["n_classes"] = 10
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR100":
        extra_vars["n_classes"] = 100
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "IMAGENET32":
        extra_vars["n_classes"] = 1000
        from utils.imagenet32 import ImageNet32

        train_dataset = ImageNet32(
            root=args.dataset_dir,
            train=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = ImageNet32(
            root=args.dataset_dir,
            train=False,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "IMAGENET":
        extra_vars["n_classes"] = 1000
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, "train"),
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, "val"),
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    for k, v in extra_vars.items():
        setattr(args, k, v)

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("encoder_path", type=str, help="Path to encoder model")

    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=[1],
        help="List of multiple seeds for evaluation",
    )

    parser.add_argument(
        "--eval-batch-size", type=int, default=256, help="Evaluation batch size"
    )
    parser.add_argument(
        "--eval-epochs", type=int, default=100, help="Number of epochs for evaluation"
    )

    parser.add_argument(
        "--eval-val-ratio",
        type=float,
        default=0.1,
        help="Ratio of the validation set to the size of the complete training set, set 0 to disable",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_folder = os.path.dirname(args.encoder_path)

    try:
        config_path = os.path.join(encoder_folder, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # for compability with earlier trained models
        print("params.yaml not found in model path, loading default config...")
        config = yaml_config_hook("./config/config.yaml")

    # record the config parameters to the args for compability with the rest of the code
    for k, v in config.items():
        setattr(args, k, v)

    # get data loaders
    train_loader, test_loader = get_dataloader(args)
    encoder = get_resnet_adaptable(args.resnet, args.dataset, pretrained=False)

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, encoder.fc.in_features)
    simclr_model.load_state_dict(torch.load(args.encoder_path, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )
    del simclr_model

    # start evaluation for each seed, so the randomness is in the model initialization, batch sampling and optimizer
    for seed in args.eval_seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        arr_train_loader, arr_val_loader, arr_test_loader = (
            create_data_loaders_from_arrays(
                train_X, train_y, test_X, test_y, args.eval_batch_size
            )
        )
        model = LogisticRegression(test_X.shape[1], args.n_classes)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        criterion = torch.nn.CrossEntropyLoss()
        print(f"### Linear evaluation with seed {seed} ###")
        for epoch in range(args.eval_epochs):
            loss_epoch, accuracy_epoch = train(
                args, arr_train_loader, model, criterion, optimizer
            )
            print(
                f"Epoch [{epoch}/{args.eval_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

        # test on the validation set
        if arr_val_loader is not None:
            loss_val, acc_val = test(args, arr_val_loader, model, criterion, optimizer)
            print(
                f"[VALIDATION]\t Loss: {loss_val / len(arr_val_loader)}\t Accuracy: {acc_val / len(arr_val_loader)}"
            )
        # final testing
        loss_test, acc_test = test(args, arr_test_loader, model, criterion, optimizer)

        print(
            f"[TEST]\t Loss: {loss_test / len(arr_test_loader)}\t Accuracy: {acc_test / len(arr_test_loader)}"
        )

        csv_path = os.path.join(encoder_folder, "linear_eval_results_v2.csv")
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            csv_writer = csv.writer(f)

            if not csv_exists:
                # this header does not match the content
                csv_writer.writerow(
                    [
                        "seed",
                        "epoch_num", # this should be ignored
                        "eval_epochs",
                        "val loss",
                        "val accuracy",
                        "test loss",
                        "test accuracy",
                    ]
                )  # header write

            if arr_val_loader is not None:
                csv_writer.writerow(
                    [
                        seed,
                        args.epoch_num,
                        args.eval_epochs,
                        loss_val / len(arr_val_loader),
                        acc_val / len(arr_val_loader),
                        loss_test / len(arr_test_loader),
                        acc_test / len(arr_test_loader),
                    ]
                )
            else:
                csv_writer.writerow(
                    [
                        seed,
                        args.epoch_num,
                        args.eval_epochs,
                        "N/A",
                        loss_test / len(arr_test_loader),
                        acc_test / len(arr_test_loader),
                    ]
                )
