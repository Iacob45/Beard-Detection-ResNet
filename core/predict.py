from pathlib import Path

import torch
from PIL import Image

from config import DEVICE, MODEL_PATH, PREDICT_IMAGE_PATH
from core.dataset import get_eval_transform, get_dataloaders
from core.model import get_model


def predict(image_path: str | Path = PREDICT_IMAGE_PATH, model_path: str | Path = MODEL_PATH) -> None:
    if image_path is None:
        raise ValueError("No image path provided for prediction.")

    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    _, _, _, classes = get_dataloaders()

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform()

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()
    predicted_class = classes[predicted_index]

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print("\nAll probabilities:")
    for class_name, probability in zip(classes, probabilities.squeeze().cpu().tolist()):
        print(f"{class_name}: {probability:.4f}")
