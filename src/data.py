"""Carga del dataset GTSRB: descarga, Dataset de PyTorch, particiones y transformaciones."""
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

KAGGLE_ID = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
NUM_CLASSES = 43
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "Límite 20 km/h", "Límite 30 km/h", "Límite 50 km/h", "Límite 60 km/h", "Límite 70 km/h",
    "Límite 80 km/h", "Fin límite 80 km/h", "Límite 100 km/h", "Límite 120 km/h", "Prohibido adelantar",
    "Prohibido adelantar (camiones)", "Prioridad en próxima intersección", "Vía con prioridad", "Ceda el paso", "Alto",
    "Circulación prohibida", "Prohibido camiones", "Prohibido el paso", "Peligro general", "Curva peligrosa a la izquierda",
    "Curva peligrosa a la derecha", "Curva doble", "Camino irregular", "Pavimento deslizante", "Estrechamiento a la derecha",
    "Obras", "Semáforo", "Peatones", "Niños", "Ciclistas",
    "Hielo o nieve", "Animales salvajes", "Fin de restricciones", "Girar a la derecha", "Girar a la izquierda",
    "Solo de frente", "De frente o a la derecha", "De frente o a la izquierda", "Mantenerse a la derecha", "Mantenerse a la izquierda",
    "Glorieta obligatoria", "Fin de prohibido adelantar", "Fin de prohibido adelantar (camiones)",
]


def download_gtsrb():
    """Descarga el dataset con kagglehub (queda en su caché) y devuelve la carpeta raíz."""
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_ID))


def load_tables(root):
    """Lee Train.csv y Test.csv. Las rutas de la columna Path son relativas a root."""
    root = Path(root)
    return pd.read_csv(root / "Train.csv"), pd.read_csv(root / "Test.csv")


def add_track_id(df):
    """Agrega la columna track a partir del nombre del archivo (clase_track_frame.png).

    Cada track son ~30 fotos consecutivas de la misma señal física, casi idénticas entre sí.
    """
    df = df.copy()
    df["track"] = df["Path"].str.extract(r"(\d+)_(\d+)_\d+\.png$").agg("_".join, axis=1)
    return df


def split_train_val(train_df, val_size=0.15, seed=42):
    """Partición estratificada por clase y agrupada por track; el test oficial no se toca.

    Si se partiera imagen por imagen, fotos casi idénticas de un mismo track quedarían
    en train y en val, y la exactitud de validación sería optimista.
    """
    df = add_track_id(train_df)
    tracks = df.groupby("track")["ClassId"].first().reset_index()
    tr_tracks, va_tracks = train_test_split(tracks["track"], test_size=val_size, stratify=tracks["ClassId"],
                                            random_state=seed)
    tr = df[df["track"].isin(set(tr_tracks))]
    va = df[df["track"].isin(set(va_tracks))]
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class GTSRBDataset(Dataset):
    """Lee las imágenes listadas en un DataFrame con columnas Path, ClassId y Roi.*."""

    def __init__(self, df, root, transform=None, use_roi=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self):
        return len(self.df)

    def load_image(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["Path"]).convert("RGB")
        if self.use_roi:
            img = img.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))
        return img

    def __getitem__(self, i):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.df.iloc[i]["ClassId"])


def get_transforms(size=48, augment=False, imagenet_norm=False):
    """Transformaciones para entrenamiento o evaluación.

    No se usa flip horizontal: invierte el sentido de señales como "girar a la izquierda".
    """
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet_norm else ((0.5,) * 3, (0.5,) * 3)
    ops = [transforms.Resize((size, size))]
    if augment:
        ops += [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def denormalize(x, imagenet_norm=False):
    """Revierte Normalize para poder graficar un tensor (C,H,W)."""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet_norm else ((0.5,) * 3, (0.5,) * 3)
    mean = x.new_tensor(mean).view(3, 1, 1)
    std = x.new_tensor(std).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
