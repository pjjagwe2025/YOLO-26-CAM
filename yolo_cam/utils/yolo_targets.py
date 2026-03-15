from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class YOLOBoxScoreTarget:
    """Target for detection CAMs.

    This sums the top-k class scores, optionally for one chosen class.
    It works with:
    - Ultralytics raw dict outputs: {'scores': [B, C, N], ...}
    - Decoded detection tensors: [B, 4+C, N] or [B, N, 4+C]
    - RT-DETR style tensors: [B, N, 4+C]
    """

    class_idx: int | None = None
    topk: int = 10
    conf_threshold: float = 0.0

    def _scores_from_output(self, model_output: torch.Tensor | dict) -> torch.Tensor:
        if isinstance(model_output, dict) and "scores" in model_output:
            scores = model_output["scores"]
            if scores.ndim != 3:
                raise ValueError(f"Expected raw YOLO scores with 3 dims, got {scores.shape}")
            # raw Ultralytics scores are [B, C, N]
            return scores.sigmoid().permute(0, 2, 1)

        if not isinstance(model_output, torch.Tensor):
            raise TypeError(f"Unsupported output type for target: {type(model_output)}")

        if model_output.ndim != 3:
            raise ValueError(f"Expected model output with 3 dims, got {model_output.shape}")

        # [B, 4+C, N]
        if model_output.shape[1] > 4 and model_output.shape[1] < model_output.shape[2]:
            return model_output[:, 4:, :].permute(0, 2, 1)

        # [B, N, 4+C]
        if model_output.shape[-1] > 4:
            return model_output[..., 4:]

        raise ValueError(f"Could not infer score layout from output shape {tuple(model_output.shape)}")

    def __call__(self, model_output: torch.Tensor | dict) -> torch.Tensor:
        scores = self._scores_from_output(model_output)

        if self.class_idx is None:
            selected = scores.max(dim=-1).values
        else:
            selected = scores[..., self.class_idx]

        if self.conf_threshold > 0:
            selected = selected[selected >= self.conf_threshold]
        else:
            selected = selected.reshape(-1)

        if selected.numel() == 0:
            return scores.sum() * 0.0

        k = min(self.topk, selected.numel()) if self.topk and self.topk > 0 else selected.numel()
        return selected.topk(k).values.sum()
