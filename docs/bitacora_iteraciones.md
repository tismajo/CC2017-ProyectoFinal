# Bitácora de iteraciones

Cada fila es un experimento: qué se cambió, qué se obtuvo y qué se decidió a partir de eso. Sirve para mostrar el proceso iterativo que pide el enunciado.

| # | Fecha | Modelo | Cambio respecto a la versión anterior | Val acc | Test acc | F1 macro | Decisión |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-21 | CNN desde cero | Tres bloques convolucionales, BatchNorm, dropout, aumento de datos y early stopping | 0.9968 | 0.9918 | 0.9870 | Usar como modelo principal por su mejor desempeño. |
| 2 | 2026-10-22 | MobileNetV2 congelada | Clasificador nuevo; backbone de ImageNet congelado | 0.5635 | 0.5991 | 0.5164 | Las características congeladas no son suficientes para este dominio. |
| 3 | 2026-10-28 | MobileNetV2 fine-tuning | Últimos cuatro bloques descongelados con lr 10 veces menor | 0.7501 | 0.7676 | 0.7102 | El ajuste mejora 16.85 puntos de accuracy, pero no supera a la CNN. |
| 4 | 2026-10-29 | CNN + perturbaciones | Brillo reducido, ruido gaussiano y desenfoque en tres severidades | - | 0.9815 / 0.7188 / 0.5574 | 0.9690 / 0.7349 / 0.5945 | El brillo tiene poco efecto; el desenfoque severo es la mayor debilidad. |
