# Checklist — Proyecto Final GTSRB

Leyenda: 🟢 entra en la presentación de avances · 🔵 entrega final

## 0. Setup 🟢
- [x] requirements.txt, .gitignore, README inicial
- [x] Verificar torchvision, kagglehub, gradio instalados (.venv)
- [ ] Repo público en GitHub
- [ ] Tema aprobado: propuesta enviada + captura del correo

## 1. Datos 🟢
- [x] src/data.py: descarga kagglehub, Dataset, split estratificado 85/15 por track, transforms (sin flip)
- [x] nb01: distribución de clases, tamaños, muestras por clase, tracks (probado)
- [ ] Ejecutar nb01 en Jupyter y escribir la interpretación

## 2. CNN desde cero 🟢
- [x] TrafficCNN en src/models.py
- [x] fit/evaluate en src/train.py (early stopping, checkpoint, history)
- [x] nb02 probado con FAST_DEV_RUN
- [ ] Entrenamiento completo de nb02 + interpretación + fila en la bitácora

## 3. Transfer learning
- [x] 🟢 Pesos de MobileNetV2 descargados (en caché de torch)
- [x] 🟢 nb03 (etapas A y B) probado con FAST_DEV_RUN
- [ ] 🟢 MobileNetV2 feature extraction (backbone congelado): entrenamiento completo
- [ ] 🔵 Fine-tuning de últimos bloques (lr 10× menor)
- [ ] 🔵 ResNet18 como comparación adicional
- [ ] 🔵 Tabla: accuracy, parámetros entrenables, tiempo/época

## 4. Hiperparámetros y regularización 🔵
- [ ] Ablación: dropout, weight decay, BN, augmentation, label smoothing
- [ ] Optimizadores: SGD+momentum, Adam, AdamW; schedulers
- [ ] Grid lr × batch size → outputs/hparams.csv

## 5. Evaluación
- [x] 🟢 src/metrics.py y nb05 probados con FAST_DEV_RUN
- [ ] 🟢 Accuracy, precision, recall, F1 (macro/weighted) en test
- [ ] 🟢 Matrices de confusión de ambos modelos
- [ ] 🔵 Classification report por clase, top pares confundidos, errores visuales
- [ ] 🔵 Tabla final con tamaño y latencia

## 6. Robustez 🔵
- [ ] Perturbaciones: brillo/contraste/gamma, ruido gaussiano/sal-pimienta, blur gaussiano/movimiento
- [ ] Curvas accuracy vs severidad para ambos modelos
- [ ] (Opcional) reentrenar con perturbaciones como augmentation

## 7. Grad-CAM 🔵
- [ ] src/gradcam.py con hooks
- [ ] Mapas en aciertos, errores e imágenes perturbadas

## 8. Interfaz 🔵
- [ ] app/app.py en Gradio: imagen, modelo, perturbación, top-5, Grad-CAM

## 9. Entregables
- [x] 🟢 docs/prompts_ia.md (actualizar en cada sesión)
- [x] 🟢 docs/bitacora_iteraciones.md
- [ ] 🟢 docs/propuesta.md
- [ ] 🔵 Informe: problema, análisis, propuesta, solución, herramientas + prompts, resultados, conclusión, bibliografía → PDF
- [ ] 🔵 Slides ≤10 en PDF
- [ ] 🔵 Video demo
- [ ] 🔵 Entrega en Canvas: PDF con link al repo + slides PDF
