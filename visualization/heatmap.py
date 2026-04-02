"""
热力图渲染器模块（核心可视化组件）

根据 design.md §3.6.1，HeatmapRenderer 是 visualization/ 模块的核心类，
负责将注意力矩阵、梯度矩阵渲染为可读的热力图，支持 1D 时序信号与 2D 图像叠加。
"""

import matplotlib
matplotlib.use('Agg')

import math
import os
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from numpy import ndarray as NDArray
from torch import Tensor


def _to_numpy_2d(tensor: Tensor) -> NDArray:
    """将任意形状的张量转为 2D numpy 数组（用于热力图渲染）。

    - (L,)           → (1, L)
    - (num_heads, L) → 对 heads 取均值 → (1, L)
    - (H, W)         → (H, W)
    """
    t = tensor.detach().cpu().float()
    if t.ndim == 1:
        return t.numpy().reshape(1, -1)
    if t.ndim == 2:
        return t.numpy()
    # 多头：(num_heads, L) 或多余维度 → 取均值压缩到 2D
    while t.ndim > 2:
        t = t.mean(dim=0)
    return t.numpy()


def _normalize_01(arr: NDArray) -> NDArray:
    """Min-Max 归一化到 [0, 1]，全零时直接返回。"""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _percentile_clip_normalize(arr: NDArray,
                                low: float = 1.0,
                                high: float = 99.0) -> NDArray:
    """百分位裁剪后归一化到 [0, 1]。"""
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    clipped = np.clip(arr, lo, hi)
    return _normalize_01(clipped)


def _arr_to_rgb(arr_2d: NDArray, cmap_name: str) -> NDArray:
    """将 [0,1] 的 2D 数组映射为 (H, W, 3) uint8 RGB 数组。"""
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(arr_2d)          # (H, W, 4) float [0,1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


class HeatmapRenderer:
    """
    热力图渲染器（核心可视化组件）。

    支持：
    - 单层注意力 / 梯度热力图渲染
    - 多层注意力面板渲染
    - 将热力图叠加到原始信号（1D 时序）或图像（2D）上
    - 多头子图面板（注意力张量含 num_heads 维度）
    - 四象限离散色彩图渲染

    适用于 ECG、NLP 等 1D 序列模型以及 ViT、Swin 等图像模型的注意力可视化。
    """

    # 四象限离散颜色（与 Quadrant 枚举顺序对应）
    _QUADRANT_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#7f7f7f"]

    def __init__(self,
                 cmap: str = "viridis",
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 150,
                 colormap: Optional[str] = None) -> None:
        """
        初始化渲染器。

        Args:
            cmap: matplotlib 颜色映射方案，默认 "viridis"。
            figsize: 默认图像尺寸 (宽, 高)，单位英寸。
            dpi: 输出分辨率，默认 150。
            colormap: 兼容旧接口，与 cmap 等价，优先使用 colormap（若传入）。
        """
        self.cmap = colormap if colormap is not None else cmap
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def render_attention(
        self,
        attention_map: Tensor,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        渲染单层注意力热力图。

        支持形状：
        - (L,)            → 1D 序列，单行热力图
        - (num_heads, L)  → 多头，子图面板（每头一行）
        - (H, W)          → 2D 注意力图
        - (num_heads, H, W) → 多头 2D，子图面板

        Args:
            attention_map: 注意力张量。
            title: 图表标题。
            save_path: 保存路径，None 则不保存。

        Returns:
            NDArray: 热力图 RGB 数组，dtype uint8。
        """
        t = attention_map.detach().cpu().float()

        # 多头 2D：(num_heads, H, W)
        if t.ndim == 3:
            return self._render_multihead_2d(t, title, save_path, self.cmap)

        # 多头 1D：(num_heads, L)
        if t.ndim == 2 and t.shape[0] > 1 and t.shape[0] <= 64:
            # 启发式判断：行数 <= 64 视为多头序列
            return self._render_multihead_1d(t, title, save_path, self.cmap)

        # 单图
        arr = _to_numpy_2d(t)
        arr_norm = _percentile_clip_normalize(arr)
        fig = self._make_single_heatmap(arr_norm, title, self.cmap)
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def render_gradient(
        self,
        gradient_map: Tensor,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        渲染梯度热力图，使用 "hot" colormap。

        Args:
            gradient_map: 梯度张量，支持 1D 或 2D（同 render_attention）。
            title: 图表标题。
            save_path: 保存路径，None 则不保存。

        Returns:
            NDArray: 热力图 RGB 数组，dtype uint8。
        """
        t = gradient_map.detach().cpu().float()
        arr = _to_numpy_2d(t)
        arr_norm = _percentile_clip_normalize(arr)
        fig = self._make_single_heatmap(arr_norm, title, cmap="hot")
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def render_multi_layer(
        self,
        layer_maps: Dict[int, Tensor],
        title: str = "",
        save_path: Optional[str] = None,
        num_cols: int = 4,
        attention_dict: Optional[Dict[int, Tensor]] = None,
    ) -> NDArray:
        """
        多层注意力/梯度对比面板，每层一个子图，自动计算行列数。

        Args:
            layer_maps: {layer_idx: tensor} 字典（或通过 attention_dict 传入）。
            title: 面板总标题。
            save_path: 保存路径，None 则不保存。
            num_cols: 列数，默认 4。
            attention_dict: 兼容旧接口，与 layer_maps 等价。

        Returns:
            NDArray: 面板 RGB 数组，dtype uint8。
        """
        maps = attention_dict if attention_dict is not None else layer_maps
        if not maps:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.axis("off")
            rgb = self._fig_to_rgb(fig)
            self._save_fig(fig, save_path)
            return rgb

        sorted_keys = sorted(maps.keys())
        n = len(sorted_keys)
        ncols = min(num_cols, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3, nrows * 2.5),
                                 dpi=self.dpi,
                                 squeeze=False)
        if title:
            fig.suptitle(title, fontsize=12)

        for i, key in enumerate(sorted_keys):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            t = maps[key].detach().cpu().float()
            arr = _to_numpy_2d(t)
            arr_norm = _percentile_clip_normalize(arr)
            ax.imshow(arr_norm, aspect="auto", cmap=self.cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Layer {key}", fontsize=9)
            ax.axis("off")

        # 关掉多余子图
        for i in range(n, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].axis("off")

        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def render_quadrant_map(
        self,
        quadrant_map: Tensor,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """
        渲染四象限图，使用离散 colormap（4种颜色）。

        Args:
            quadrant_map: (H, W) 整数张量，值域 [1, 4]（对应 4 个象限）。
            title: 图表标题。
            save_path: 保存路径，None 则返回 Figure，传入则保存并 close。

        Returns:
            Figure 对象（save_path=None 时）或 None（保存后 close）。
        """
        arr = quadrant_map.detach().cpu().float().numpy()
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap_q = ListedColormap(self._QUADRANT_COLORS)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_q.N)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(arr, cmap=cmap_q, norm=norm, interpolation="nearest")
        if title:
            ax.set_title(title, fontsize=11)
        ax.axis("off")
        labels = ["Core Discriminative", "Redundant Attention",
                  "Potential Influence", "Irrelevant"]
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=self._QUADRANT_COLORS[i], label=labels[i])
                   for i in range(4)]
        ax.legend(handles=patches, loc="lower right", fontsize=8)
        plt.tight_layout()
        return self._save_and_close(fig, save_path)

    def overlay_on_image(
        self,
        heatmap: Tensor,
        image: Tensor,
        alpha: float = 0.5,
        save_path: Optional[str] = None,
    ) -> Optional[Figure]:
        """
        将热力图叠加到原始图像上（alpha 混合）。

        Args:
            heatmap: (H, W) 热力图张量，值域任意（内部归一化）。
            image: (C, H, W) 或 (H, W, C) 图像张量，值域 [0,1] 或 [0,255]。
            alpha: 热力图透明度，默认 0.5。
            save_path: 保存路径，None 则返回 Figure。

        Returns:
            Figure 或 None。
        """
        # 归一化 heatmap
        hm = heatmap.detach().cpu().float()
        hm_arr = _normalize_01(hm.numpy())  # (H, W)

        # 处理 image
        img = image.detach().cpu().float()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            # (C, H, W) → (H, W, C)
            img = img.permute(1, 2, 0)
        img_arr = img.numpy()
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0
        img_arr = np.clip(img_arr, 0, 1)

        # 将 heatmap 插值到图像尺寸
        img_h, img_w = img_arr.shape[:2]
        hm_tensor = torch.from_numpy(hm_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        hm_resized = F.interpolate(hm_tensor, size=(img_h, img_w),
                                   mode="bilinear", align_corners=False)
        hm_resized = hm_resized.squeeze().numpy()  # (img_h, img_w)

        # 将 heatmap 转为 RGB
        cmap_fn = cm.get_cmap(self.cmap)
        hm_rgb = cmap_fn(hm_resized)[:, :, :3]  # (H, W, 3)

        # 如果 image 是灰度，扩展为 3 通道
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[2] == 1:
            img_arr = np.concatenate([img_arr] * 3, axis=-1)

        blended = (1 - alpha) * img_arr + alpha * hm_rgb
        blended = np.clip(blended, 0, 1)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(blended)
        ax.axis("off")
        plt.tight_layout()
        return self._save_and_close(fig, save_path)

    def overlay_on_signal(
        self,
        heatmap: NDArray,
        signal: NDArray,
        alpha: float = 0.4,
        save_path: Optional[str] = None,
    ) -> NDArray:
        """
        将热力图叠加到原始信号或图像上。

        支持两种模式：
        - 1D 时序信号：折线图上方叠加颜色背景。
        - 2D 图像：alpha 混合。

        Args:
            heatmap: RGB 数组，1D 模式 (L, 3) 或 2D 模式 (H, W, 3)。
            signal:  1D 模式 (L,)/(C, L)，2D 模式 (H, W, 3)。
            alpha:   热力图透明度，默认 0.4。
            save_path: 保存路径，None 则不保存。

        Returns:
            NDArray: 叠加后的图像，dtype uint8。
        """
        # 判断是 1D 还是 2D 模式
        if signal.ndim == 1 or (signal.ndim == 2 and signal.shape[0] <= 64):
            return self._overlay_1d(heatmap, signal, alpha, save_path)
        else:
            return self._overlay_2d(heatmap, signal, alpha, save_path)

    def _save_and_close(self, fig: Figure, save_path: Optional[str]) -> Optional[Figure]:
        """保存/返回通用逻辑：有 save_path 则保存并 close，无则返回 Figure。"""
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _make_single_heatmap(self, arr_norm: NDArray,
                             title: str, cmap: str) -> Figure:
        """创建单张热力图 Figure。"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(arr_norm, aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")
        if title:
            ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    def _render_multihead_2d(self, t: Tensor, title: str,
                             save_path: Optional[str], cmap: str) -> NDArray:
        """多头 2D 注意力：(num_heads, H, W) → 子图面板。"""
        num_heads = t.shape[0]
        ncols = min(4, num_heads)
        nrows = math.ceil(num_heads / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3, nrows * 2.5),
                                 dpi=self.dpi, squeeze=False)
        if title:
            fig.suptitle(title, fontsize=12)
        for i in range(num_heads):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            arr = t[i].numpy()
            arr_norm = _percentile_clip_normalize(arr)
            ax.imshow(arr_norm, aspect="auto", cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Head {i}", fontsize=8)
            ax.axis("off")
        for i in range(num_heads, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].axis("off")
        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def _render_multihead_1d(self, t: Tensor, title: str,
                             save_path: Optional[str], cmap: str) -> NDArray:
        """多头 1D 注意力：(num_heads, L) → 子图面板（每头一行）。"""
        num_heads = t.shape[0]
        fig, axes = plt.subplots(num_heads, 1,
                                 figsize=(self.figsize[0], num_heads * 1.5),
                                 dpi=self.dpi, squeeze=False)
        if title:
            fig.suptitle(title, fontsize=12)
        for i in range(num_heads):
            ax = axes[i][0]
            arr = t[i].numpy().reshape(1, -1)
            arr_norm = _percentile_clip_normalize(arr)
            ax.imshow(arr_norm, aspect="auto", cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_ylabel(f"H{i}", fontsize=8, rotation=0, labelpad=15)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def _overlay_1d(self, heatmap: NDArray, signal: NDArray,
                    alpha: float, save_path: Optional[str]) -> NDArray:
        """1D 折线图 + 颜色背景叠加。"""
        if signal.ndim == 2:
            sig = signal[0]  # 取第一个通道
        else:
            sig = signal
        L = len(sig)

        # heatmap 可能是 (L, 3) 或 (1, L, 3) 等
        hm = heatmap
        if hm.ndim == 3 and hm.shape[0] == 1:
            hm = hm[0]  # (L, 3)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        x = np.arange(L)

        # 逐段绘制颜色背景
        if hm.ndim == 2 and hm.shape[-1] == 3 and len(hm) == L:
            for i in range(L - 1):
                color = hm[i] / 255.0 if hm.max() > 1 else hm[i]
                ax.axvspan(i, i + 1, facecolor=color, alpha=alpha, linewidth=0)
        ax.plot(x, sig, color="black", linewidth=0.8)
        ax.set_xlim(0, L - 1)
        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    def _overlay_2d(self, heatmap: NDArray, signal: NDArray,
                    alpha: float, save_path: Optional[str]) -> NDArray:
        """2D 图像 alpha 混合叠加。"""
        img = signal.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        hm = heatmap.astype(np.float32)
        if hm.max() > 1.0:
            hm = hm / 255.0
        blended = np.clip((1 - alpha) * img + alpha * hm, 0, 1)
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(blended)
        ax.axis("off")
        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

    @staticmethod
    def _fig_to_rgb(fig: Figure) -> NDArray:
        """将 Figure 渲染为 (H, W, 3) uint8 RGB 数组。"""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # RGBA bytes
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, :3]  # 丢弃 alpha 通道

    @staticmethod
    def _save_fig(fig: Figure, save_path: Optional[str]) -> None:
        """有 save_path 则保存并 close，无则不操作。"""
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
