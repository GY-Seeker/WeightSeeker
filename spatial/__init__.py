"""空间重构与数值对齐模块

可选模块 — 仅图像输入场景需要。

本模块假设注意力矩阵对应 2D 图像 Patch 网格，负责将注意力或梯度从
Patch 粒度重构回像素空间，并完成跨层的数值对齐（归一化）。

适用场景：
    - 图像输入模型（ViT、Swin Transformer 等使用 Patch Embedding 的架构）

跳过条件：
    - 输入为 1D 序列（ECG、NLP 等）时，注意力矩阵本身即为最终结果，
      应在 pipeline.py 中设置 ``skip_spatial=True`` 跳过此模块。

主要组件：
    - :class:`SpatialReshaper`：将 Patch 级向量重塑为 2D 网格并上采样至原图尺寸
    - :class:`Normalizer`：百分位裁剪与 Min-Max 归一化，统一跨层数值尺度
    - :class:`Interpolator`：双线性插值与高斯平滑
"""

from .reshaper import SpatialReshaper
from .normalizer import Normalizer
from .interpolator import Interpolator

# 别名，与需求文档保持一致
SpatialNormalizer = Normalizer
SpatialInterpolator = Interpolator

__all__ = [
    "SpatialReshaper",
    "Normalizer",
    "Interpolator",
    "SpatialNormalizer",
    "SpatialInterpolator",
]
