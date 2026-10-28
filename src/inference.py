"""Inferencia reutilizable para el clasificador de senales de transito."""
from pathlib import Path

import torch
from PIL import Image

from .data import CLASS_NAMES, get_transforms
from .models import TrafficCNN


class TrafficSignPredictor:
    """Carga la CNN entrenada y devuelve predicciones con probabilidades."""

    def __init__(self, checkpoint, device=None):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"No se encontro el checkpoint: {self.checkpoint}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TrafficCNN().to(self.device)
        saved = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        state = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = get_transforms(size=48, augment=False, imagenet_norm=False)

    def prepare(self, image):
        """Convierte una imagen PIL o arreglo RGB a un lote normalizado."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_tensor(self, batch, top_k=5):
        """Predice un lote ya normalizado y devuelve los top-k de cada imagen."""
        probabilities = torch.softmax(self.model(batch.to(self.device)), dim=1)
        values, indices = probabilities.topk(min(top_k, probabilities.shape[1]), dim=1)
        return [
            [
                {
                    "class_id": int(class_id),
                    "label": CLASS_NAMES[int(class_id)],
                    "probability": float(probability),
                }
                for probability, class_id in zip(row_values.cpu(), row_indices.cpu())
            ]
            for row_values, row_indices in zip(values, indices)
        ]

    def predict(self, image, top_k=5):
        """Predice una imagen y devuelve una lista ordenada de resultados."""
        return self.predict_tensor(self.prepare(image), top_k=top_k)[0]

