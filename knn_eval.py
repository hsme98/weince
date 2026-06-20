# utils/knn_eval.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.utils.data
from typing import Callable, Tuple, List, Dict

import argparse
import os
import csv
import yaml

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
import json
import pandas as pd


from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet_adaptable
from simclr.modules.transformations import TransformsSimCLR

from torchvision.datasets import ImageFolder
from utils import yaml_config_hook


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


@torch.no_grad()
def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
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


def get_features_with_val_split(
    simclr_model, train_loader, test_loader, device, val_ratio=0.1, split_seed=2
):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    # stratified split
    assert val_ratio > 0
    rng = np.random.default_rng(split_seed)
    classes = np.unique(train_y)
    train_indices = []
    val_indices = []
    for c in classes:
        c_indices = np.where(train_y == c)[0]
        rng.shuffle(c_indices)
        n_val = int(len(c_indices) * val_ratio)
        val_indices.extend(c_indices[:n_val])
        train_indices.extend(c_indices[n_val:])
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    return (
        train_X[train_indices],
        train_y[train_indices],
        train_X[val_indices],
        train_y[val_indices],
        test_X,
        test_y,
    )


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    # further split train into train and val
    return train_X, train_y, test_X, test_y


from enum import Enum


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def average(self):
        return self.sum / self.count if self.count != 0 else 0.0


class AggrType(Enum):
    ACC = "accuracy"
    RECALL_AT_K = "recall_at_k"


class Aggregate:
    name: str
    meter: AverageMeter
    aggr_type: AggrType
    aggr_params: Dict

    def __init__(self, name: str, aggr_type: AggrType, aggr_params: Dict = {}):
        self.name = name
        self.meter = AverageMeter()
        self.aggr_type = aggr_type
        self.aggr_params = aggr_params


@torch.no_grad()
def _eval_aggregate(
    aggr: Aggregate,
    sim_w: torch.Tensor,
    sim_lbls: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
):
    k = aggr.aggr_params.get("k", 1)
    sim_lbls = sim_lbls[:, :k]
    sim_w = sim_w[:, :k]

    if aggr.aggr_type == AggrType.ACC:
        tau = aggr.aggr_params.get("tau", 0.1)

        sim_w = sim_w - sim_w.max(dim=1, keepdim=True).values
        weights = torch.exp(sim_w / tau)  # [B, k]
        B = query_labels.size(0)
        scores = torch.zeros(B, num_classes, device=query_labels.device)
        scores.scatter_add_(1, sim_lbls, weights)
        pred = scores.argmax(dim=1)
        correct = (pred == query_labels).sum().item()
        total = query_labels.size(0)
        acc = correct / total
        aggr.meter.update(acc, total)
    elif aggr.aggr_type == AggrType.RECALL_AT_K:
        correct = (query_labels.unsqueeze(1) == sim_lbls[:, :k]).sum(dim=1)
        total = query_labels.size(0)
        recall = (correct > 0).sum().item() / total
        aggr.meter.update(recall, total)
    else:
        raise NotImplementedError(f"Unknown aggregate type {aggr.aggr_type}")


@torch.no_grad()
def knn_eval_batch(
    query_feats: torch.Tensor,  # [B, D] normalized
    query_labels: torch.Tensor,  # [B]
    num_classes: int,
    bank_loader: torch.utils.data.DataLoader,
    aggregates: List["Aggregate"],
    device: torch.device,
):
    k = max(agr.aggr_params.get("k", 1) for agr in aggregates)

    B = query_feats.size(0)

    rolling_sim = torch.full((B, k), -torch.inf, device=device)
    rolling_lbl = torch.full((B, k), -1, device=device, dtype=torch.long)

    for bank_feats, bank_labels in bank_loader:
        bank_feats = bank_feats.to(device, non_blocking=True)
        bank_labels = bank_labels.to(device, non_blocking=True)

        bank_feats = F.normalize(bank_feats, dim=1)
        sim = query_feats @ bank_feats.T  # [B, n]
        kk = min(k, sim.size(1))

        topk_sim, topk_idx = sim.topk(kk, dim=1)  # [B, kk]
        topk_lbl = bank_labels[topk_idx]  # [B, kk]

        if kk < k:
            pad_sim = torch.full((B, k - kk), -torch.inf, device=device)
            pad_lbl = torch.full((B, k - kk), -1, device=device, dtype=torch.long)
            topk_sim = torch.cat([topk_sim, pad_sim], dim=1)  # [B, k]
            topk_lbl = torch.cat([topk_lbl, pad_lbl], dim=1)  # [B, k]

        merged_sim = torch.cat([rolling_sim, topk_sim], dim=1)  # [B, 2k]
        merged_lbl = torch.cat([rolling_lbl, topk_lbl], dim=1)  # [B, 2k]

        rolling_sim, sel = merged_sim.topk(k, dim=1)  # [B, k]
        rolling_lbl = merged_lbl.gather(1, sel)  # [B, k]

    query_labels = query_labels.to(device, non_blocking=True)

    for aggr in aggregates:
        # Change _eval_aggregate to accept neighbor labels directly
        _eval_aggregate(
            aggr=aggr,
            sim_w=rolling_sim,
            sim_lbls=rolling_lbl,
            query_labels=query_labels,
            num_classes=num_classes,
        )


@torch.no_grad()
def knn_eval(
    args: argparse.Namespace,
    model,
    query_loader: torch.utils.data.DataLoader,
    bank_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    aggregates = [
        Aggregate(
            name=f"knn_accuracy_{max(args.knn_ks_eval)}",
            aggr_type=AggrType.ACC,
            aggr_params={"tau": args.knn_temperature, "k": max(args.knn_ks_eval)},
        )
    ]
    for k in args.knn_ks_eval:
        aggregates.append(
            Aggregate(
                name=f"knn_recall_at_{k}",
                aggr_type=AggrType.RECALL_AT_K,
                aggr_params={"k": k},
            )
        )
    for idx, (query_feats, query_labels) in enumerate(query_loader):
        query_feats = query_feats.to(device)
        query_labels = query_labels.to(device)

        # get normalized features
        query_feats = F.normalize(query_feats, dim=1)

        knn_eval_batch(
            query_feats=query_feats,
            query_labels=query_labels,
            num_classes=args.n_classes,  # set by the get_dataloader method
            bank_loader=bank_loader,
            aggregates=aggregates,
            device=device,
        )

        if idx % 20 == 0:
            print(f"Step [{idx}/{len(query_loader)}]\t kNN evaluation...")
    return aggregates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("encoder_path", type=str, help="Path to pre-trained model")
    parser.add_argument(
        "--knn-temperature",
        type=float,
        default=0.1,
        help="Temperature for knn evaluation",
    )
    parser.add_argument(
        "--knn-ks-eval",
        nargs="+",
        type=int,
        default=[1, 2, 5, 10, 20, 50],
        help="k values to evaluate for accuracy",
    )
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.1,
        help="Ratio of training set to use as validation set",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(
        os.path.join(os.path.dirname(args.encoder_path), "config.yaml"), "r"
    ) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    train_loader, test_loader = get_dataloader(args)

    encoder = get_resnet_adaptable(args.resnet, args.dataset, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    simclr_model.load_state_dict(
        torch.load(args.encoder_path, map_location=args.device.type)
    )
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    print("Extracting features...")
    if args.val_split_ratio > 0:
        (
            train_X,
            train_y,
            val_X,
            val_y,
            test_X,
            test_y,
        ) = get_features_with_val_split(
            simclr_model,
            train_loader,
            test_loader,
            args.device,
            args.val_split_ratio,
            split_seed=args.split_seed,
        )
    else:
        train_X, train_y, test_X, test_y = get_features(
            simclr_model, train_loader, test_loader, args.device
        )
        val_X, val_y = None, None

    print("Feature extraction completed.")
    # create dataloaders for knn eval
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    train_knn_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_X, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.long),
    )
    test_knn_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    results = {}
    if val_X is not None:
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.long),
        )
        val_knn_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )

        print("Evaluating kNN on validation set...")
        val_aggregates = knn_eval(
            args,
            simclr_model,
            val_knn_loader,
            train_knn_loader,
            args.device,
        )

        for aggr in val_aggregates:
            print(f"[VAL] {aggr.name}: {aggr.meter.average():.4f}")
            results[f"val_{aggr.name}"] = aggr.meter.average()

    # evaluate on test set
    print("Evaluating kNN on test set...")
    test_aggregates = knn_eval(
        args, simclr_model, test_knn_loader, train_knn_loader, args.device
    )
    for aggr in test_aggregates:
        print(f"[TEST] {aggr.name}: {aggr.meter.average():.4f}")
        results[f"test_{aggr.name}"] = aggr.meter.average()

    # save results to json
    results_path = os.path.join(
        os.path.dirname(args.encoder_path), "knn_eval_results_v2.csv"
    )
    results["val_split_ratio"] = args.val_split_ratio
    results["split_seed"] = args.split_seed
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    else:
        df = pd.DataFrame([results])
    df.to_csv(results_path, index=False)
