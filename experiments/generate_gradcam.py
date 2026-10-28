"""Genera ejemplos Grad-CAM de aciertos y errores de la CNN."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.data import CLASS_NAMES, GTSRBDataset, download_gtsrb, load_tables
from src.gradcam import GradCAM, overlay_heatmap
from src.inference import TrafficSignPredictor


def collect_examples(predictor, dataset, correct_count=3, error_count=3):
    correct, errors = [], []
    for index in range(len(dataset)):
        image = dataset.load_image(index)
        true_id = int(dataset.df.iloc[index]["ClassId"])
        prediction = predictor.predict(image, top_k=1)[0]
        item = (image, true_id, prediction["class_id"], prediction["probability"])
        target = correct if true_id == prediction["class_id"] else errors
        if len(target) < (correct_count if target is correct else error_count):
            target.append(item)
        if len(correct) >= correct_count and len(errors) >= error_count:
            break
    return correct + errors


def main():
    predictor = TrafficSignPredictor(ROOT / "outputs" / "cnn_scratch.pt")
    data_root = download_gtsrb()
    _, test_df = load_tables(data_root)
    dataset = GTSRBDataset(test_df, data_root)
    examples = collect_examples(predictor, dataset)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    with GradCAM(predictor.model) as gradcam:
        for ax, (image, true_id, predicted_id, probability) in zip(axes.ravel(), examples):
            batch = predictor.prepare(image)
            heatmap, _, _ = gradcam(batch, [predicted_id])
            overlay = overlay_heatmap(image, heatmap[0].cpu().numpy())
            ax.imshow(overlay)
            result = "Acierto" if true_id == predicted_id else "Error"
            ax.set_title(
                f"{result} ({probability:.1%})\nReal: {CLASS_NAMES[true_id]}\nPred.: {CLASS_NAMES[predicted_id]}",
                fontsize=9,
            )
            ax.axis("off")
    fig.suptitle("Regiones relevantes para las predicciones de la CNN")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "gradcam_ejemplos.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

