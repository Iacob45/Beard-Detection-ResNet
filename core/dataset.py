from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transform()
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_eval_transform()
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
