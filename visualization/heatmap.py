"""
热力图渲染器模块（核心可视化组件）

根据 design.md §3.6.1，HeatmapRenderer 是 visualization/ 模块的核心类，
负责将注意力矩阵、梯度矩阵渲染为可读的热力图，支持 1D 时序信号与 2D 图像叠加。
"""

from typing import Dict, Optional
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray as NDArray
from matplotlib.figure import Figure


class HeatmapRenderer:
    """
    热力图渲染器（核心可视化组件）。

    支持：
    - 单层注意力 / 梯度热力图渲染
    - 多层注意力面板渲染
    - 将热力图叠加到原始信号（1D 时序）或图像（2D）上

    适用于 ECG、NLP 等 1D 序列模型以及 ViT、Swin 等图像模型的注意力可视化。
    """

    def __init__(self, colormap: str = "jet") -> None:
        """
        初始化渲染器。

        Args:
            colormap: matplotlib 颜色映射方案，默认 "jet"。
                      常用选项：'jet'、'viridis'、'hot'、'RdBu_r'。
        """
        raise NotImplementedError("待实现")

    def render_attention(
        self,
        attention_map: Tensor,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        渲染单层注意力热力图。

        支持 1D 序列注意力（形状 (L,) 或 (H, L)）和
        2D 图像注意力（形状 (H, W)）两种格式，自动检测并适配渲染方式。

        Args:
            attention_map: 注意力张量。支持以下形状：
                - (L,)：1D 序列，平均头后的注意力权重向量
                - (num_heads, L)：多头 1D 序列注意力
                - (H, W)：2D 图像注意力热力图
            title: 图表标题，默认为空字符串。
            save_path: 图像保存路径（含扩展名，如 "output/attn.png"）。
                       None 表示不保存到磁盘。

        Returns:
            NDArray: 热力图 RGB 数组，形状 (H, W, 3)，值域 [0, 255]，dtype uint8。
        """
        raise NotImplementedError("待实现")

    def render_gradient(
        self,
        gradient_map: Tensor,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        渲染梯度热力图。

        渲染逻辑与 render_attention 一致，使用相同的 colormap。
        梯度值在渲染前会经过百分位裁剪归一化到 [0, 1]。

        Args:
            gradient_map: 梯度张量，支持 1D 或 2D 格式（同 render_attention）。
            title: 图表标题。
            save_path: 图像保存路径，None 则不保存。

        Returns:
            NDArray: 热力图 RGB 数组，形状 (H, W, 3)，值域 [0, 255]，dtype uint8。
        """
        raise NotImplementedError("待实现")

    def render_multi_layer(
        self,
        attention_dict: Dict[int, Tensor],
        num_cols: int = 4,
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        渲染多层注意力热力图面板（子图网格）。

        按层索引升序排列各层注意力图，自动计算行数。

        Args:
            attention_dict: 字典 {layer_idx: attention_tensor}，
                            每个 tensor 支持 1D 或 2D 注意力格式。
            num_cols: 面板列数，默认 4。层数不足时自动补空。
            save_path: 面板图像保存路径，None 则不保存。

        Returns:
            NDArray: 拼接后的面板图像，形状 (panel_H, panel_W, 3)，dtype uint8。
        """
        raise NotImplementedError("待实现")

    def overlay_on_signal(
        self,
        heatmap: NDArray,
        signal: NDArray,
        alpha: float = 0.4,
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        将热力图叠加到原始信号或图像上。

        支持两种叠加模式：
        - 1D 时序信号：在折线图上方叠加颜色背景，颜色深浅表示重要性。
        - 2D 图像：采用 alpha 混合将热力图叠加到 RGB 图像上。

        Args:
            heatmap: 热力图数组。
                     - 1D 模式：形状 (L, 3) 的 RGB 数组，表示每个时间步的颜色。
                     - 2D 模式：形状 (H, W, 3) 的 RGB 数组。
            signal: 原始信号数组。
                    - 1D 模式：形状 (L,) 或 (C, L) 的时序信号。
                    - 2D 模式：形状 (H, W, 3) 的 RGB 图像，值域 [0, 255]。
            alpha: 热力图不透明度，范围 [0, 1]，默认 0.4。
                   值越大热力图越突出，原始信号/图像越淡。
            save_path: 保存路径，None 则不保存。

        Returns:
            NDArray: 叠加后的图像，dtype uint8。
        """
        raise NotImplementedError("待实现")
