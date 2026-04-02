"""插值处理器 - 空间插值与平滑

可选模块 — 仅图像输入场景需要。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.exceptions import InvalidInputError


class Interpolator:
    """插值处理器：双线性插值与高斯模糊平滑。"""

    def __init__(self) -> None:
        """初始化插值器。"""
        pass  # 无状态，无需额外初始化

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def bilinear_interpolate(
        self,
        input_tensor: Tensor,
        target_size: Tuple[int, int],
    ) -> Tensor:
        """双线性插值，将输入张量上采样至 target_size。

        Args:
            input_tensor: (..., H, W)，支持 2D/3D/4D 输入。
            target_size: 目标空间尺寸 (target_h, target_w)。

        Returns:
            插值后的张量，形状与输入相同（H,W 替换为 target_size）。

        Raises:
            InvalidInputError: target_size 含非正整数时。
        """
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise InvalidInputError(
                expected="target_size > 0",
                actual=f"target_size={target_size}",
            )

        ndim = input_tensor.dim()
        x = input_tensor.float()

        if ndim == 2:
            # (H, W) → (1, 1, H, W) → 插值 → (H, W)
            x = x.unsqueeze(0).unsqueeze(0)
            out = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
            return out.squeeze(0).squeeze(0)
        elif ndim == 3:
            # (C, H, W) → (1, C, H, W) → 插值 → (C, H, W)
            x = x.unsqueeze(0)
            out = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
            return out.squeeze(0)
        elif ndim == 4:
            # (B, C, H, W) → 直接插值
            return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        else:
            raise InvalidInputError(
                expected="input_tensor.dim() in [2, 3, 4]",
                actual=f"dim={ndim}",
            )

    def gaussian_smooth(
        self,
        tensor: Tensor,
        sigma: float = 1.0,
        kernel_size: int = 5,
    ) -> Tensor:
        """高斯模糊平滑。

        Args:
            tensor: (H, W)、(B, H, W) 或 (B, C, H, W)。
            sigma: 高斯核标准差，值越大越模糊。
            kernel_size: 高斯核大小（奇数），偶数自动加 1。

        Returns:
            平滑后的张量，形状与输入相同。

        Raises:
            InvalidInputError: sigma <= 0 或 kernel_size < 1 时。
        """
        if sigma <= 0:
            raise InvalidInputError(expected="sigma > 0", actual=f"sigma={sigma}")
        if kernel_size < 1:
            raise InvalidInputError(expected="kernel_size >= 1", actual=f"kernel_size={kernel_size}")

        # 确保奇数核
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 构建 1D 高斯核
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - half
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # 2D 高斯核 = 外积
        kernel_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # (ks, ks)

        ndim = tensor.dim()
        x = tensor.float()

        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            num_channels = 1
        elif ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
            num_channels = 1
        elif ndim == 4:
            num_channels = x.shape[1]
        else:
            raise InvalidInputError(
                expected="tensor.dim() in [2, 3, 4]",
                actual=f"dim={ndim}",
            )

        # 扩展为 (out_channels, in_channels/groups, ks, ks)
        weight = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)
        weight = weight.expand(num_channels, 1, kernel_size, kernel_size)

        padding = kernel_size // 2
        if ndim == 4:
            out = F.conv2d(x, weight, padding=padding, groups=num_channels)
        else:
            out = F.conv2d(x, weight, padding=padding, groups=1)

        # 还原维度
        if ndim == 2:
            out = out.squeeze(0).squeeze(0)
        elif ndim == 3:
            out = out.squeeze(1)

        return out

    # ------------------------------------------------------------------
    # 高阶接口（用户需求扩展）
    # ------------------------------------------------------------------

    def interpolate(
        self,
        feature_map: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        mode: str = "bilinear",
    ) -> Tensor:
        """通用插值接口，支持多种模式。

        Args:
            feature_map: (B, H, W) 或 (B, C, H, W)。
            target_size: 目标尺寸，None 则保持不变。
            mode: "bilinear"、"bicubic" 或 "nearest"。

        Returns:
            插值后的张量。
        """
        if target_size is None:
            return feature_map

        x = feature_map.float()
        needs_squeeze = False

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
            needs_squeeze = True

        align = True if mode in ("bilinear", "bicubic") else False
        out = F.interpolate(x, size=target_size, mode=mode, align_corners=align if mode != "nearest" else None)

        if needs_squeeze:
            out = out.squeeze(1)

        return out

    def interpolate_attention(
        self,
        attention_2d: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """对多头注意力图插值。

        Args:
            attention_2d: (B, num_heads, H, W)。
            target_size: 目标尺寸。

        Returns:
            (B, num_heads, target_H, target_W)。
        """
        if target_size is None:
            return attention_2d
        x = attention_2d.float()
        B, num_heads, H, W = x.shape
        # 合并 batch 和 heads 维度一起插值
        x = x.view(B * num_heads, 1, H, W)
        out = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        return out.view(B, num_heads, target_size[0], target_size[1])

    def interpolate_gradient(
        self,
        gradient_2d: Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """对梯度图插值。

        Args:
            gradient_2d: (B, H, W) 或 (B, C, H, W)。
            target_size: 目标尺寸。

        Returns:
            插值后的梯度图。
        """
        return self.interpolate(gradient_2d, target_size=target_size)
