import numpy as np
import pytest
import torch
from PIL import Image

from src.gradcam import GradCAM, overlay_heatmap
from src.models import TrafficCNN
from src.robustness import PERTURBATIONS, apply_perturbation


@pytest.fixture
def sample_image():
    array = np.linspace(0, 255, 48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    return Image.fromarray(array)


@pytest.mark.parametrize("kind", PERTURBATIONS)
@pytest.mark.parametrize("severity", (0, 1, 2, 3))
def test_perturbations_preserve_image(sample_image, kind, severity):
    result = apply_perturbation(sample_image, kind, severity, seed=7)
    assert result.mode == "RGB"
    assert result.size == sample_image.size


def test_gaussian_noise_is_reproducible(sample_image):
    first = np.asarray(apply_perturbation(sample_image, "ruido_gaussiano", 2, seed=10))
    second = np.asarray(apply_perturbation(sample_image, "ruido_gaussiano", 2, seed=10))
    assert np.array_equal(first, second)


def test_gradcam_shape_and_overlay(sample_image):
    model = TrafficCNN()
    batch = torch.randn(2, 3, 48, 48)
    with GradCAM(model) as gradcam:
        maps, logits, classes = gradcam(batch)
    assert maps.shape == (2, 12, 12)
    assert logits.shape == (2, 43)
    assert classes.shape == (2,)
    assert torch.isfinite(maps).all()
    overlay = overlay_heatmap(sample_image, maps[0].numpy())
    assert overlay.shape == (48, 48, 3)

