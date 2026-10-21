"""Predicción y métricas de clasificación."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def predict(model, loader, device, max_batches=None):
    """Devuelve (y_true, y_pred, probabilidades) como arreglos de NumPy."""
    model.eval()
    ys, probs = [], []
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    probs = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), probs.argmax(1), probs


def summary_metrics(y_true, y_pred, name=None):
    """Accuracy y precision/recall/F1 en promedio macro y weighted."""
    row = {"modelo": name, "accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        row.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    return row


def plot_confusion(y_true, y_pred, num_classes, title="", normalize=True, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    if normalize:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax = ax or plt.subplots(figsize=(11, 9))[1]
    sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None, square=True, cbar=True, ax=ax,
                xticklabels=5, yticklabels=5)
    ax.set(title=title, xlabel="Predicción", ylabel="Real")
    return cm


def top_confusions(y_true, y_pred, class_names, k=10):
    """Pares (real, predicho) más frecuentes entre los errores."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:k], cm.shape))[0]
    return pd.DataFrame([{"real": class_names[i], "predicho": class_names[j], "errores": cm[i, j]}
                         for i, j in idx if cm[i, j] > 0])
