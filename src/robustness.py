"""Perturbaciones reproducibles para evaluar robustez del clasificador."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


PERTURBATIONS = ("brillo", "ruido_gaussiano", "desenfoque")


def apply_perturbation(image, kind, severity, seed=42):
    """Aplica una perturbacion con severidad entera entre 0 y 3."""
    if kind not in PERTURBATIONS:
        raise ValueError(f"Perturbacion desconocida: {kind}")
    if severity not in (0, 1, 2, 3):
        raise ValueError("La severidad debe estar entre 0 y 3")

    image = image.convert("RGB")
    if severity == 0:
        return image.copy()
    if kind == "brillo":
        return ImageEnhance.Brightness(image).enhance(1.0 - 0.22 * severity)
    if kind == "desenfoque":
        return image.filter(ImageFilter.GaussianBlur(radius=0.8 * severity))

    rng = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float32) / 255.0
    sigma = 0.045 * severity
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray(np.uint8(noisy * 255))


class PerturbedGTSRB(Dataset):
    """Vista perturbada de un GTSRBDataset sin duplicar imagenes."""

    def __init__(self, base_dataset, kind, severity, transform=None, seed=42):
        self.base = base_dataset
        self.kind = kind
        self.severity = severity
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = self.base.load_image(index)
        image = apply_perturbation(image, self.kind, self.severity, seed=self.seed + index)
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.base.df.iloc[index]["ClassId"])
        return image, label

