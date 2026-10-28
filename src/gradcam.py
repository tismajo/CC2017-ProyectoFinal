"""Grad-CAM para explicar las predicciones de modelos convolucionales."""
import cv2
import numpy as np
import torch
from torch import nn


def last_conv_layer(model):
    """Devuelve la ultima capa convolucional de un modelo."""
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("El modelo no contiene capas convolucionales")
    return layers[-1]


class GradCAM:
    """Calcula mapas Grad-CAM mediante hooks en una capa convolucional."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or last_conv_layer(model)
        self.activations = None
        self.gradients = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activations),
        ]

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient):
        self.gradients = gradient.detach()

    def __call__(self, batch, class_ids=None):
        """Devuelve (mapas, logits, clases objetivo) para un lote."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(batch)
        if class_ids is None:
            class_ids = logits.argmax(dim=1)
        elif not torch.is_tensor(class_ids):
            class_ids = torch.as_tensor(class_ids, device=logits.device)
        class_ids = class_ids.to(logits.device).long()

        scores = logits.gather(1, class_ids[:, None]).sum()
        scores.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No se capturaron activaciones o gradientes")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1))
        flat = maps.flatten(1)
        minima = flat.min(dim=1).values[:, None, None]
        maxima = flat.max(dim=1).values[:, None, None]
        maps = (maps - minima) / (maxima - minima).clamp_min(1e-8)
        return maps.detach(), logits.detach(), class_ids.detach()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


def overlay_heatmap(image, heatmap, alpha=0.42):
    """Superpone un mapa Grad-CAM sobre una imagen PIL o un arreglo RGB."""
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)

