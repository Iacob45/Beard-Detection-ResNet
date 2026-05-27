from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, MODEL_PATH
from core.dataset import get_dataloaders
from core.model import get_model


def test(model_path: str | Path = MODEL_PATH) -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    _, _, test_loader, classes = get_dataloaders()

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    test_loss = running_loss / total_samples
    test_accuracy = correct_predictions / total_samples

    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        zero_division=0
    ))

    print("\nTrue labels:")
    print(Counter(all_labels))

    print("\nPredicted labels:")
    print(Counter(all_predictions))

    print("\nMapping:")
    print(test_loader.dataset.class_to_idx)
