from __future__ import annotations

import numpy as np

from .base_cam import BaseCAM


class GradCAMPlusPlus(BaseCAM):
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
            raise RuntimeError("GradCAM++ needs gradients, but no gradients were captured.")

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=(2, 3), keepdims=True)

        eps = 1e-8
        aij = grads_power_2 / (2.0 * grads_power_2 + sum_activations * grads_power_3 + eps)
        aij = np.where(np.isnan(aij), 0.0, aij)
        positive_gradients = np.maximum(grads, 0.0)
        weights = np.sum(aij * positive_gradients, axis=(2, 3))
        return weights
