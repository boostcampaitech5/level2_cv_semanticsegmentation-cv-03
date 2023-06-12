import os
import random
from argparse import ArgumentParser

from dataset import XRayDataset, XRayInferenceDataset
from train import train
from inference import test

import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import wandb

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models


CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)


def main(args):
    wandb.init(
        name=args.exp_name + "_Resume" if args.resume else args.exp_name,
        project="Xray-Segmentation",
        entity="ganisokay",
        config=args,
    )

    set_seed(args.seed)

    save_csv = os.path.join(args.save_csv, args.exp_name)
    save_checkpoint = os.path.join(args.save_checkpoint, args.exp_name)

    # CSV file save path
    if not os.path.isdir(save_csv):
        os.makedirs(save_csv, exist_ok=True)

    # Checkpoint file save path
    if not os.path.isdir(save_checkpoint):
        os.makedirs(save_checkpoint, exist_ok=True)

    train_transform = A.Compose([A.Resize(512, 512)])
    valid_transform = A.Compose([A.Resize(512, 512)])
    test_transform = A.Compose([A.Resize(512, 512)])

    train_dataset = XRayDataset(
        args.data_root, transforms=train_transform, split=f"train{args.fold}"
    )
    valid_dataset = XRayDataset(
        args.data_root, transforms=valid_transform, split=f"val{args.fold}"
    )
    test_dataset = XRayInferenceDataset(args.data_root, transforms=test_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    if args.resume:
        previous_state = torch.load(os.path.join(save_checkpoint, "best_model.pt"))
        print("Finished model loading.")
        start_epoch, model = previous_state["epoch"], previous_state["model"]
    else:
        # Model 정의
        model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

        # 시작 epoch 정의
        start_epoch = 0

    # Loss function 정의
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Training
    train(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        args.epochs,
        start_epoch,
        CLASSES,
        args.patience,
        save_checkpoint,
    )

    # Inference
    test(test_loader, CLASSES, save_checkpoint, save_csv, args.make_csv)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Path
    parser.add_argument(
        "--data-root",
        type=str,
        default="../../data",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="./predictions",
    )

    # Default Parameter
    parser.add_argument("--seed", type=int, default=1226)
    parser.add_argument("--exp-name", type=str, default="[test]ExpName")
    parser.add_argument("--resume", type=bool, default=False)

    # DataLoader
    parser.add_argument("--fold", type=str, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    # Inference
    parser.add_argument("--make-csv", type=bool, default=True)

    args = parser.parse_args()

    main(args)
