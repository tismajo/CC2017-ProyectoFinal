# CC2017 – Proyecto Final · Clasificación de señales de tránsito (GTSRB)

Comparación de una CNN entrenada desde cero con un modelo preentrenado (MobileNetV2, y ResNet18 como comparación adicional) usando transfer learning y fine-tuning sobre el dataset [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), que tiene 43 clases.

Incluye:
- Evaluación con accuracy, precision, recall, F1-score y matrices de confusión.
- Experimentos de regularización, optimización e hiperparámetros.
- Robustez ante cambios de iluminación, ruido y desenfoque.
- Grad-CAM e interfaz en Gradio.

## Estructura
```
src/        código reutilizable (datos, modelos, entrenamiento, métricas, robustez, Grad-CAM)
notebooks/  experimentos numerados 01–07
app/        interfaz Gradio
docs/       checklist, propuesta, informe, historial de prompts, bitácora de iteraciones
outputs/    checkpoints, figuras y tablas
```

## Cómo ejecutar
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
El dataset se descarga con `kagglehub` la primera vez que se ejecuta `notebooks/01_exploracion_datos.ipynb` y queda en la caché de kagglehub. Los notebooks se ejecutan en orden.

Avance del proyecto: [docs/checklist.md](docs/checklist.md).
