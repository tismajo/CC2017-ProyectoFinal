# CC2017 – Proyecto Final · Clasificación de señales de tránsito (GTSRB)

Comparación de una CNN entrenada desde cero con MobileNetV2 mediante transfer learning y fine-tuning sobre las 43 clases del dataset [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Incluye:
- Evaluación con accuracy, precision, recall, F1-score y matrices de confusión.
- Robustez de la CNN ante brillo reducido, ruido gaussiano y desenfoque.
- Explicaciones Grad-CAM en aciertos y errores.
- Inferencia top-5 e interfaz interactiva en Gradio.

## Resultados principales

| Modelo | Accuracy test | F1 macro | ms por imagen (CPU) |
|---|---:|---:|---:|
| CNN desde cero | **0.9918** | **0.9870** | 7.41 |
| MobileNetV2 congelada | 0.5991 | 0.5164 | **3.09** |
| MobileNetV2 fine-tuning | 0.7676 | 0.7102 | 3.62 |

## Estructura
```
src/          datos, modelos, entrenamiento, métricas, inferencia, robustez y Grad-CAM
notebooks/    EDA, entrenamientos, evaluación comparativa y robustez
experiments/  scripts reproducibles para robustez y Grad-CAM
app/          interfaz Gradio
docs/         informe, checklist, historial de prompts y bitácora
outputs/      checkpoint seleccionado, figuras y tablas
```

## Cómo ejecutar
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
El dataset se descarga con `kagglehub` la primera vez que se ejecuta `notebooks/01_exploracion_datos.ipynb` y queda en la caché de kagglehub. Los notebooks se ejecutan en orden.

## Aplicación

Con el checkpoint `outputs/cnn_scratch.pt` disponible:

```bash
python app/app.py
```

La interfaz acepta una imagen, presenta las cinco clases más probables y genera un mapa Grad-CAM. Para evaluar robustez o volver a generar las explicaciones:

```bash
python experiments/run_robustness.py
python experiments/generate_gradcam.py
```

## Verificación

```bash
python -m pytest -q
```

## Informe

La fuente editable se encuentra en `docs/informe_final.tex` y puede compilarse con una distribución estándar de LaTeX o en Overleaf.

Avance del proyecto: [docs/checklist.md](docs/checklist.md).
