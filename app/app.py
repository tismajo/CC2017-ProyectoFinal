"""Interfaz Gradio para clasificar senales de transito y mostrar Grad-CAM."""
import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gradcam import GradCAM, overlay_heatmap
from src.inference import TrafficSignPredictor


def build_demo(checkpoint):
    predictor = TrafficSignPredictor(checkpoint)
    gradcam = GradCAM(predictor.model)

    def classify(image):
        if image is None:
            raise gr.Error("Seleccione una imagen antes de clasificar")
        batch = predictor.prepare(image)
        results = predictor.predict_tensor(batch, top_k=5)[0]
        heatmap, _, _ = gradcam(batch)
        probabilities = {row["label"]: row["probability"] for row in results}
        explanation = overlay_heatmap(image, heatmap[0].cpu().numpy())
        return probabilities, explanation

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Senal de transito"),
        outputs=[
            gr.Label(num_top_classes=5, label="Predicciones"),
            gr.Image(type="numpy", label="Regiones relevantes (Grad-CAM)"),
        ],
        title="Clasificacion de senales de transito",
        description="CNN entrenada con GTSRB. Cargue una imagen para obtener las cinco clases mas probables.",
        flagging_mode="never",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Inicia la interfaz Gradio")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "cnn_scratch.pt",
        help="Ruta al checkpoint de la CNN",
    )
    parser.add_argument("--share", action="store_true", help="Crea un enlace publico temporal")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo(args.checkpoint).launch(share=args.share)

