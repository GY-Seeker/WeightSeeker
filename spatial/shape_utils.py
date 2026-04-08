"""张量形状转换工具函数（纯函数，供可视化模块使用）。

将注意力/梯度张量从模型输出格式转换为可视化所需的 2D (H, W) 格式。
这些函数不持有任何状态，可独立调用。
"""

import logging
from typing import Dict, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def extract_attn_2d(attn: Tensor) -> Optional[Tensor]:
    """将注意力张量转换为 2D 格式 (H, W) 用于可视化叠加。

    支持输入形状：
    - 4D (B, H, L, L)：对 batch/heads 取平均，尝试 reshape 到方形
    - 3D (B, L, L)：对 batch 取平均，尝试 reshape
    - 其他：直接返回
    """
    if attn.dim() == 4:  # (B, H, L, L)
        attn_2d = attn[0].mean(dim=0)  # (L, L)
        L = attn_2d.shape[0]
        sqrt_L = int(L ** 0.5)
        if sqrt_L * sqrt_L == L:
            attn_2d = attn_2d.view(sqrt_L, sqrt_L)
        else:
            logger.warning("注意力图不是方形 (%d)，使用第一行作为 1D 注意力", L)
            attn_2d = attn_2d[0]  # 取第一行 (L,)
    elif attn.dim() == 3:  # (B, L, L)
        attn_2d = attn[0].mean(dim=0)
        L = attn_2d.shape[0]
        sqrt_L = int(L ** 0.5)
        if sqrt_L * sqrt_L == L:
            attn_2d = attn_2d.view(sqrt_L, sqrt_L)
        else:
            attn_2d = attn_2d[0]
    else:
        attn_2d = attn
    return attn_2d


def extract_grad_2d(grad: Tensor) -> Optional[Tensor]:
    """将梯度张量转换为 2D 格式 (H, W) 用于可视化叠加。

    支持输入形状：
    - 4D (B, C, H, W)：对通道取平均 -> (H, W)
    - 2D (B, L)：尝试 reshape 到方形，窗口注意力格式则返回 None
    """
    if grad.dim() == 4:  # (B, C, H, W)
        return grad[0].mean(dim=0)  # 对通道取平均 (H, W)
    elif grad.dim() == 2:  # (B, L)
        if grad.shape[0] > 100:  # 窗口注意力，跳过
            logger.info("梯度是窗口注意力格式，跳过梯度叠加可视化")
            return None
        grad_1d = grad[0]
        L = grad_1d.shape[0]
        sqrt_L = int(L ** 0.5)
        if sqrt_L * sqrt_L == L:
            return grad_1d.view(sqrt_L, sqrt_L)
        return None
    return None


def process_multi_layer_attention(
    attention_maps: Dict[int, Tensor],
) -> Dict[int, Tensor]:
    """处理多层注意力图，过滤窗口注意力并 reshape 到 2D。

    Returns:
        处理后的注意力图 {layer_idx: 2D tensor}，仅包含标准全局注意力的层。
    """
    processed_attention: Dict[int, Tensor] = {}
    for layer_idx, attn in attention_maps.items():
        if attn.dim() == 4:  # (B, H, L, L) 或 (num_windows, H, ws, ws)
            if attn.shape[0] > 100:  # 窗口注意力，跳过
                continue
            else:
                attn_proc = attn.mean(dim=[0, 1])
                L = attn_proc.shape[0]
                sqrt_L = int(L ** 0.5)
                if sqrt_L * sqrt_L == L:
                    attn_proc = attn_proc.view(sqrt_L, sqrt_L)
                processed_attention[layer_idx] = attn_proc
        elif attn.dim() == 3:  # (B, L, L)
            attn_proc = attn.mean(dim=0)
            L = attn_proc.shape[0]
            sqrt_L = int(L ** 0.5)
            if sqrt_L * sqrt_L == L:
                attn_proc = attn_proc.view(sqrt_L, sqrt_L)
            processed_attention[layer_idx] = attn_proc
        else:
            processed_attention[layer_idx] = attn
    return processed_attention
