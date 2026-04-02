"""空间重构器 - Patch 粒度转换与上采样

可选模块 — 仅图像输入场景需要。

负责将 Patch 级注意力/梯度向量重塑为与原始图像对应的 2D 空间网格，
并通过插值方法将 Patch 网格上采样至像素级热力图。

同时支持 Swin Transformer 的窗口注意力重组（窗口 → 全局）。
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..core.types import ModelArchitecture


class SpatialReshaper:
    """空间重构器：负责 Patch 粒度转换与上采样。

    核心职责：
    1. 将 Patch 级一维向量重塑为二维网格（patch_to_grid）
    2. 将二维 Patch 网格上采样至原始图像像素尺寸（upsample_to_image）
    3. 将 Swin 的窗口注意力重组为全局格式（swin_window_reorganize）

    适用架构：
        - ViT：标准 (num_patches_h, num_patches_w) 网格
        - Swin Transformer：需要额外的窗口重组处理

    注意：
        此类假设输入为 2D 图像模型。对于 1D 序列模型（ECG/NLP），
        应在 pipeline 中设置 skip_spatial=True，跳过此模块。
    """

    def __init__(
        self,
        patch_size: int,
        image_size: Tuple[int, int],
        architecture: ModelArchitecture = ModelArchitecture.VIT,
        num_stages: Optional[int] = None,
    ) -> None:
        """初始化空间重构器。

        Args:
            patch_size: Patch 大小（像素），例如 16 表示 16×16 的 Patch。
            image_size: 原始图像尺寸 (H, W)，例如 (224, 224)。
            architecture: 模型架构类型，用于处理 Swin 等特殊架构的额外逻辑。
                          默认为 ModelArchitecture.VIT。
            num_stages: stage 数量（Swin 架构必填，用于各 stage 分辨率适配）。
                        ViT 等无 stage 的架构可传 None。

        Raises:
            ValueError: 当 architecture=Swin 且 num_stages 未提供时。
        """
        raise NotImplementedError("待实现")

    def patch_to_grid(
        self,
        patch_vector: Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """将 Patch 级一维向量重塑为二维网格。

        将形状为 (B, num_patches) 或 (num_patches,) 的 Patch 重要性向量，
        重塑为对应 Patch 空间排列的二维网格形式，以便后续上采样。

        Args:
            patch_vector: Patch 级向量，形状为 (B, num_patches) 或 (num_patches,)。
            num_patches_h: Patch 网格高度（即垂直方向的 Patch 数量）。
            num_patches_w: Patch 网格宽度（即水平方向的 Patch 数量）。

        Returns:
            Tensor: 重塑后的二维网格，形状为
                    (B, num_patches_h, num_patches_w) 或
                    (num_patches_h, num_patches_w)。

        Raises:
            ValueError: 当 patch_vector 的元素数量与
                        num_patches_h * num_patches_w 不匹配时。
        """
        raise NotImplementedError("待实现")

    def upsample_to_image(self, grid: Tensor, method: str = "bilinear") -> Tensor:
        """将 Patch 级二维网格上采样至原始图像像素级尺寸。

        使用指定的插值方法将 Patch 网格 (num_patches_h, num_patches_w)
        上采样为与原始图像相同分辨率 (image_h, image_w) 的热力图。

        Args:
            grid: Patch 级二维网格，形状为 (H, W) 或 (B, H, W)。
            method: 插值方法，支持：
                    - "bilinear"：双线性插值（默认，平滑效果好）
                    - "gaussian"：高斯模糊平滑（边缘更柔和）

        Returns:
            Tensor: 上采样后的热力图，形状为 (image_h, image_w) 或 (B, image_h, image_w)，
                    与初始化时传入的 image_size 对应。

        Raises:
            ValueError: 当 method 不在支持列表内时。
        """
        raise NotImplementedError("待实现")

    def swin_window_reorganize(
        self,
        window_attention: Tensor,
        stage_idx: int,
        feature_h: int,
        feature_w: int,
        window_size: int,
        shift_size: int = 0,
    ) -> Tensor:
        """将 Swin Transformer 窗口注意力重组为全局格式。

        Swin Transformer 的注意力在局部窗口内计算，输出格式为
        (num_windows, window_size^2, window_size^2)。本方法将其重组为
        以特征图像素为单位的全局注意力图 (feature_h, feature_w)，
        以便后续上采样到原图尺寸。

        对于带位移的窗口（shift_size > 0），需额外处理循环位移后的
        Patch 位置映射。

        Args:
            window_attention: 窗口内注意力张量，形状为
                              (num_windows, window_size^2, window_size^2)。
            stage_idx: 当前所属 stage 的索引（从 0 开始），
                       用于查找当前 stage 的分辨率信息。
            feature_h: 当前 stage 的特征图高度（Patch 数量）。
            feature_w: 当前 stage 的特征图宽度（Patch 数量）。
            window_size: 窗口大小（以 Patch 为单位）。
            shift_size: 窗口循环位移大小（0 表示无位移）。

        Returns:
            Tensor: 全局注意力图，形状为 (feature_h, feature_w)，
                    每个位置的值为该 Patch 在所有窗口注意力中的聚合得分。

        Raises:
            ValueError: 当 feature_h 或 feature_w 不能被 window_size 整除时。
        """
        raise NotImplementedError("待实现")
