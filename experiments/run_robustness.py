"""Evalua la CNN ante brillo reducido, ruido y desenfoque."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import GTSRBDataset, download_gtsrb, get_transforms, load_tables
from src.inference import TrafficSignPredictor
from src.metrics import predict, summary_metrics
from src.robustness import PERTURBATIONS, PerturbedGTSRB


DISPLAY_NAMES = {
    "brillo": "Brillo reducido",
    "ruido_gaussiano": "Ruido gaussiano",
    "desenfoque": "Desenfoque gaussiano",
}


def evaluate(checkpoint, batch_size=256, num_workers=2):
    data_root = download_gtsrb()
    _, test_df = load_tables(data_root)
    raw_test = GTSRBDataset(test_df, data_root)
    transform = get_transforms(48)
    predictor = TrafficSignPredictor(checkpoint)
    rows = []

    conditions = [("sin_perturbacion", 0)]
    conditions += [(kind, severity) for kind in PERTURBATIONS for severity in (1, 2, 3)]
    for kind, severity in conditions:
        effective_kind = "brillo" if kind == "sin_perturbacion" else kind
        dataset = PerturbedGTSRB(raw_test, effective_kind, severity, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        start = time.time()
        y_true, y_pred, _ = predict(predictor.model, loader, predictor.device)
        row = summary_metrics(y_true, y_pred, "CNN desde cero")
        row.update(
            perturbacion=kind,
            severidad=severity,
            segundos=time.time() - start,
            muestras=len(y_true),
        )
        rows.append(row)
        print(
            f"{kind:18s} nivel {severity}: accuracy={row['accuracy']:.4f}, "
            f"F1 macro={row['f1_macro']:.4f}, {row['segundos']:.0f}s"
        )

    return pd.DataFrame(rows)


def save_outputs(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robustez_test.csv"
    results.to_csv(csv_path, index=False)

    baseline = results.loc[results["perturbacion"] == "sin_perturbacion", "accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for kind in PERTURBATIONS:
        subset = results[results["perturbacion"] == kind].sort_values("severidad")
        x = [0, *subset["severidad"].tolist()]
        y = [baseline, *subset["accuracy"].tolist()]
        ax.plot(x, y, marker="o", linewidth=2, label=DISPLAY_NAMES[kind])
    ax.set(
        xlabel="Nivel de severidad",
        ylabel="Accuracy en test",
        title="Robustez de la CNN ante perturbaciones",
        xticks=(0, 1, 2, 3),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "robustez_curvas.png", dpi=160)
    plt.close(fig)
    return csv_path


def main():
    results = evaluate(ROOT / "outputs" / "cnn_scratch.pt")
    save_outputs(results, ROOT / "outputs")


if __name__ == "__main__":
    main()

