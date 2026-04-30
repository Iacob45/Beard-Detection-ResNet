import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH, MODELS_DIR
from core.dataset import get_dataloaders
from core.model import get_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = running_loss / total_samples
    val_accuracy = correct_predictions / total_samples

    return val_loss, val_accuracy


def train(resume_training: bool = False):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, classes = get_dataloaders()

    model = get_model()
    model = model.to(device)

    if resume_training and MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Resumed training from: {MODEL_PATH}")
    elif resume_training:
        print(f"No saved model found at {MODEL_PATH}. Starting new training.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    best_val_accuracy = 0.0

    print(f"Device: {device}")
    print(f"Classes: {classes}")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"LR: {current_lr:.6f} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved to: {MODEL_PATH}")

    print("Training completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
