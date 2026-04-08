"""Token 重要性计算（Grad-CAM 融合 + 纯注意力对角线 fallback）。

此模块将 pipeline 中的 token 重要性计算逻辑独立为纯函数，
供 VisualizationManager 和其他模块直接调用。
"""

import logging
from typing import Dict, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_token_importance_with_fallback(
    attention_maps: Dict[int, Tensor],
    hidden_gradients: Dict[int, Tensor],
) -> Optional[Tensor]:
    """统一的 token 重要性计算逻辑：先尝试 Grad-CAM 融合，失败则 fallback 到纯注意力对角线。

    Args:
        attention_maps: 各层注意力图 {layer_idx: (B, H, L, L) 或 (B, L, L)}
        hidden_gradients: 各层隐藏状态梯度 {layer_idx: (B, L) 或标量}

    Returns:
        (B, L) 形状的 token 重要性张量，归一化到 [0, 1]；失败时返回 None。
    """
    from .fusion_utils import compute_token_importance, normalize_for_fusion

    if not attention_maps:
        return None

    # 检查是否有完整的 (B, L) 梯度
    has_complete_gradients = any(
        grad.dim() == 2 for grad in hidden_gradients.values()
    )

    token_importance = None

    # 尝试使用 Grad-CAM 融合
    if has_complete_gradients and hidden_gradients:
        try:
            token_importance = compute_token_importance(
                attention_maps=attention_maps,
                hidden_gradients=hidden_gradients,
                method='gradcam',
            )  # (B, L)
        except (ValueError, Exception) as e:
            logger.warning("Grad-CAM 融合失败（%s），使用纯注意力对角线", e)
            token_importance = None

    # Fallback: 使用纯注意力对角线
    if token_importance is None:
        logger.info("使用纯注意力对角线作为重要性得分")
        diagonals_per_layer = []
        for layer_idx, attn in attention_maps.items():
            if attn.dim() == 4:  # (B, H, L, L) 或 (num_windows*B, H, ws², ws²)
                if attn.shape[0] > 100:  # 窗口注意力
                    diag = attn.mean(dim=[1, 2, 3])  # (num_windows,)
                    diagonals_per_layer.append(diag.unsqueeze(0))  # (1, num_windows)
                else:
                    B, H, L, _ = attn.shape
                    diagonal = attn[:, :, range(L), range(L)].mean(dim=1)  # (B, L)
                    diagonals_per_layer.append(diagonal)
            elif attn.dim() == 3:  # (B, L, L)
                B, L, _ = attn.shape
                diagonal = attn[:, range(L), range(L)]  # (B, L)
                diagonals_per_layer.append(diagonal)

        if diagonals_per_layer:
            stacked = torch.stack(diagonals_per_layer, dim=0)  # (num_layers, B, L)
            token_importance = stacked.mean(dim=0)  # (B, L)
            token_importance = normalize_for_fusion(token_importance)

    return token_importance
