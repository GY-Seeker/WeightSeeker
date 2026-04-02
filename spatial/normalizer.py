"""尺度对齐器 - 数值归一化

可选模块 — 仅图像输入场景需要。
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.exceptions import InvalidInputError


class Normalizer:
    """尺度对齐器：百分位裁剪与多种归一化方法。

    将注意力图、梯度图等统一到 [0, 1] 数值尺度，用于可视化和跨层比较。
    """

    def __init__(
        self,
        low_percentile: float = 0.01,
        high_percentile: float = 0.99,
    ) -> None:
        """初始化尺度对齐器。

        Args:
            low_percentile: 下百分位裁剪点，范围 [0, 1)。
            high_percentile: 上百分位裁剪点，范围 (0, 1]。

        Raises:
            InvalidInputError: low_percentile >= high_percentile 时。
        """
        if low_percentile >= high_percentile:
            raise InvalidInputError(
                expected="low_percentile < high_percentile",
                actual=f"low={low_percentile}, high={high_percentile}",
            )
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def percentile_clip(self, tensor: Tensor) -> Tensor:
        """按初始化的百分位阈值裁剪张量，去除极端值。

        Args:
            tensor: 任意形状的输入张量。

        Returns:
            裁剪后的张量，形状与输入相同。
        """
        flat = tensor.float().flatten()
        low_val = torch.quantile(flat, self.low_percentile)
        high_val = torch.quantile(flat, self.high_percentile)
        return tensor.float().clamp(min=low_val.item(), max=high_val.item())

    def min_max_normalize(
        self,
        tensor: Tensor,
        target_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tensor:
        """Min-Max 线性归一化到目标值域。

        Args:
            tensor: 任意形状的输入张量。
            target_range: 目标值域 (min_val, max_val)。

        Returns:
            归一化后的张量，值域在 target_range 内。

        Raises:
            InvalidInputError: target_range[0] >= target_range[1] 时。
        """
        if target_range[0] >= target_range[1]:
            raise InvalidInputError(
                expected="target_range[0] < target_range[1]",
                actual=f"target_range={target_range}",
            )
        x = tensor.float()
        t_min = x.min()
        t_max = x.max()
        if t_max == t_min:
            return torch.zeros_like(x)
        normalized = (x - t_min) / (t_max - t_min)
        low, high = target_range
        return normalized * (high - low) + low

    def normalize_for_visualization(self, tensor: Tensor) -> Tensor:
        """完整可视化归一化：百分位裁剪 + Min-Max 映射到 [0, 1]。

        Args:
            tensor: 原始注意力或梯度张量。

        Returns:
            值域 [0, 1] 的归一化张量。
        """
        clipped = self.percentile_clip(tensor)
        return self.min_max_normalize(clipped, target_range=(0.0, 1.0))

    def z_score_normalize(self, tensor: Tensor) -> Tensor:
        """Z-Score 标准化：(x - mean) / (std + eps)。

        Args:
            tensor: 任意形状的输入张量。

        Returns:
            零均值、单位方差的标准化张量。
        """
        x = tensor.float()
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # 高阶接口（用户需求扩展）
    # ------------------------------------------------------------------

    def normalize(self, tensor: Tensor, method: str = "minmax") -> Tensor:
        """通用归一化接口，按 method 选择方式。

        Args:
            tensor: 输入张量。
            method: "minmax"、"percentile"、"sigmoid" 或 "softmax"。

        Returns:
            归一化后的张量，值域 [0, 1]。
        """
        if method == "minmax":
            return self.min_max_normalize(tensor)
        elif method == "percentile":
            return self.normalize_for_visualization(tensor)
        elif method == "sigmoid":
            return torch.sigmoid(tensor.float())
        elif method == "softmax":
            # 在最后两个维度上做 softmax
            x = tensor.float()
            shape = x.shape
            x_flat = x.view(*shape[:-2], -1)
            out = F.softmax(x_flat, dim=-1)
            return out.view(shape)
        else:
            raise InvalidInputError(
                expected="method in ['minmax', 'percentile', 'sigmoid', 'softmax']",
                actual=f"method='{method}'",
            )

    def normalize_attention(self, attention: Tensor) -> Tensor:
        """对每个注意力头独立做 Min-Max 归一化。

        Args:
            attention: (B, num_heads, h, w)。

        Returns:
            (B, num_heads, h, w)，每个头独立归一化到 [0, 1]。
        """
        B, H, h, w = attention.shape
        out = torch.zeros_like(attention.float())
        for b in range(B):
            for head in range(H):
                out[b, head] = self.min_max_normalize(attention[b, head])
        return out

    def normalize_gradient(self, gradient: Tensor) -> Tensor:
        """对每个样本独立做 Min-Max 归一化。

        Args:
            gradient: (B, h, w)。

        Returns:
            (B, h, w)，每个样本独立归一化到 [0, 1]。
        """
        B = gradient.shape[0]
        out = torch.zeros_like(gradient.float())
        for b in range(B):
            out[b] = self.min_max_normalize(gradient[b])
        return out

    def clip_outliers(self, tensor: Tensor, percentile: float = 99.0) -> Tensor:
        """将超过指定百分位的值截断。

        Args:
            tensor: 任意形状的输入张量。
            percentile: 截断百分位，如 99 表示在 99% 处截断。

        Returns:
            截断后的张量。
        """
        q = percentile / 100.0
        threshold = torch.quantile(tensor.float().flatten(), q)
        return tensor.float().clamp(max=threshold.item())
