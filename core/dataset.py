from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE


def get_train_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transform(),
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_eval_transform(),
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
