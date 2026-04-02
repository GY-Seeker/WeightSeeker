"""融合工具函数 - 注意力与梯度的简化融合（无策略模式，纯工具函数）。"""

import torch
from torch import Tensor


def normalize_for_fusion(
    tensor: Tensor,
    low_percentile: float = 0.01,
    high_percentile: float = 0.99,
) -> Tensor:
    """百分位裁剪 + Min-Max 归一化至 [0, 1]。"""
    if low_percentile >= high_percentile:
        raise ValueError(f"low_percentile({low_percentile}) >= high_percentile({high_percentile})")
    flat = tensor.flatten().float()
    low_val = torch.quantile(flat, low_percentile).item()
    high_val = torch.quantile(flat, high_percentile).item()
    clipped = tensor.float().clamp(min=low_val, max=high_val)
    v_min = clipped.min().item()
    v_max = clipped.max().item()
    if abs(v_max - v_min) < 1e-8:
        return torch.zeros_like(clipped)
    return (clipped - v_min) / (v_max - v_min)


def weighted_sum_fusion(
    attention: Tensor,
    gradient: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """加权求和融合：alpha * attention + (1-alpha) * gradient。"""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 必须在 [0,1]，当前: {alpha}")
    if attention.shape != gradient.shape:
        raise ValueError(f"形状不一致: {attention.shape} vs {gradient.shape}")
    return alpha * attention + (1.0 - alpha) * gradient


def gradcam_fusion(
    attention: Tensor,
    gradient: Tensor,
) -> Tensor:
    """GradCAM 式融合：normalize(attention * gradient)。"""
    if attention.shape != gradient.shape:
        raise ValueError(f"形状不一致: {attention.shape} vs {gradient.shape}")
    product = attention * gradient
    v_min = product.min().item()
    v_max = product.max().item()
    if abs(v_max - v_min) < 1e-8:
        return torch.zeros_like(product)
    return (product - v_min) / (v_max - v_min)
