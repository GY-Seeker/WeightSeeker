"""插值处理器 - 空间插值与平滑

可选模块 — 仅图像输入场景需要。

提供双线性插值和高斯模糊平滑两种方法，用于将低分辨率的
Patch 级热力图上采样至任意目标尺寸。

通常由 :class:`SpatialReshaper` 内部调用，也可单独使用。
"""

from typing import Tuple

import torch
from torch import Tensor


class Interpolator:
    """插值处理器：提供空间插值与平滑方法。

    支持两种主要操作：
    1. 双线性插值（bilinear_interpolate）：将任意分辨率张量上采样至目标尺寸，
       适合需要保留精确数值对应关系的场景。
    2. 高斯模糊平滑（gaussian_smooth）：对热力图施加高斯核平滑，
       消除上采样后的方块感，产生更自然的视觉效果。

    注意：
        本类仅处理图像空间张量（含 H, W 维度），
        对于 1D 序列数据不适用。
    """

    def __init__(self) -> None:
        """初始化插值器。

        当前无需额外参数，所有配置在调用时按需传入。
        """
        raise NotImplementedError("待实现")

    def bilinear_interpolate(
        self,
        input_tensor: Tensor,
        target_size: Tuple[int, int],
    ) -> Tensor:
        """对输入张量执行双线性插值，上采样至目标尺寸。

        使用 PyTorch ``F.interpolate`` 的双线性模式实现，
        支持任意批次维度的输入张量（至少含 H、W 两个空间维度）。

        Args:
            input_tensor: 输入张量，形状为 (..., H, W)。
                          支持 (H, W)、(C, H, W)、(B, C, H, W) 等形式，
                          内部会自动添加/移除批次维度以满足 F.interpolate 要求。
            target_size: 目标空间尺寸 (target_h, target_w)。

        Returns:
            Tensor: 插值后的张量，形状与输入相同（除 H, W 被替换为目标尺寸）。

        Raises:
            ValueError: 当 target_size 中存在非正整数时。
        """
        raise NotImplementedError("待实现")

    def gaussian_smooth(
        self,
        tensor: Tensor,
        sigma: float = 1.0,
        kernel_size: int = 5,
    ) -> Tensor:
        """对输入张量施加高斯模糊平滑。

        通过构建高斯卷积核并应用于输入张量，产生平滑的热力图效果。
        适合在上采样后消除方块感，提升可视化质量。

        Args:
            tensor: 输入张量，形状为 (H, W) 或 (B, H, W) 或 (B, C, H, W)。
            sigma: 高斯核标准差，控制平滑程度。值越大越模糊，默认 1.0。
            kernel_size: 高斯核大小（应为奇数），默认 5。
                         若传入偶数，内部会自动加 1 使其为奇数。

        Returns:
            Tensor: 平滑后的张量，形状与输入相同。

        Raises:
            ValueError: 当 sigma <= 0 或 kernel_size < 1 时。
        """
        raise NotImplementedError("待实现")
