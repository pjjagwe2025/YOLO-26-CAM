from __future__ import annotations

import numpy as np

from .base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, task: str = "od", reshape_transform=None):
        super().__init__(
            model=model,
            target_layers=target_layers,
            task=task,
            reshape_transform=reshape_transform,
            uses_gradients=True,
        )

    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads) -> np.ndarray:
        if grads is None:
            raise RuntimeError("GradCAM needs gradients, but no gradients were captured.")
        return np.mean(grads, axis=(2, 3))
