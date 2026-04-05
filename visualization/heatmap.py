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

    def render_multihead_all_heads(
        self,
        attention_maps: Dict[int, Tensor],
        title: str = "",
        save_path: Optional[str] = None,
        num_cols: int = 6,
    ) -> NDArray:
        """
        渲染所有层的所有注意力头的热力图面板。
        
        每行显示 num_cols 个头，行数根据总头数自动计算。
        支持多个层的注意力图，每层独立一个子图区域。

        Args:
            attention_maps: {layer_idx: tensor} 字典，每个 tensor 形状为 (B, H, L, L)。
            title: 图表总标题。
            save_path: 保存路径，None 则不保存。
            num_cols: 每行显示的列数，默认 6 个。

        Returns:
            NDArray: 面板 RGB 数组，dtype uint8。
        """
        if not attention_maps:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.axis("off")
            rgb = self._fig_to_rgb(fig)
            self._save_fig(fig, save_path)
            return rgb

        # 获取第一层的注意力图来确定头数和尺寸
        first_layer_idx = sorted(attention_maps.keys())[0]
        first_attn = attention_maps[first_layer_idx]  # (B, H, L, L)
        
        # 取第一个 batch 样本
        if first_attn.dim() == 4:
            first_attn = first_attn[0]  # (H, L, L)
        elif first_attn.dim() == 3:
            # 可能已经是 (H, L, L)
            pass
        
        num_heads = first_attn.shape[0]
        seq_len = first_attn.shape[1]
        
        # 计算总层数和总图数
        num_layers = len(attention_maps)
        total_plots = num_layers * num_heads
        
        # 计算行列数
        ncols = min(num_cols, total_plots)
        nrows = math.ceil(total_plots / ncols)
        
        # 创建子图
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.5, nrows * 2.5),
            dpi=self.dpi,
            squeeze=False
        )
        
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')
        
        plot_idx = 0
        for layer_idx in sorted(attention_maps.keys()):
            attn = attention_maps[layer_idx]
            
            # 取第一个 batch
            if attn.dim() == 4:
                attn_sample = attn[0].detach().cpu().numpy()  # (H, L, L)
            elif attn.dim() == 3:
                attn_sample = attn.detach().cpu().numpy()
            else:
                continue
            
            # 绘制该层的所有头
            for head_idx in range(min(num_heads, attn_sample.shape[0])):
                if plot_idx >= nrows * ncols:
                    break
                    
                row = plot_idx // ncols
                col = plot_idx % ncols
                ax = axes[row][col]
                
                head_attn = attn_sample[head_idx]  # (L, L)
                arr_norm = _percentile_clip_normalize(head_attn)
                
                im = ax.imshow(arr_norm, aspect='auto', cmap=self.cmap,
                              vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=7)
                ax.set_xlabel('Key', fontsize=5)
                ax.set_ylabel('Query', fontsize=5)
                ax.tick_params(labelsize=4)
                
                # 只在每行的最后一个图添加 colorbar
                if col == ncols - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                plot_idx += 1
        
        # 关闭多余的子图
        for i in range(plot_idx, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb

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
        background_image: Optional[Tensor] = None,
        alpha: float = 0.6,
        original_signal: Optional[Tensor] = None,  # 新增：原始波形数据
    ) -> Optional[Figure]:
        """
        渲染四象限图，使用离散 colormap（4 种颜色）。
    
        Args:
            quadrant_map: (H, W) 整数张量，值域 [1, 4]（对应 4 个象限）。
                         或 (1, L, 1) 用于 1D 序列数据。
            title: 图表标题。
            save_path: 保存路径，None 则返回 Figure，传入则保存并 close。
            background_image: 背景原图 (C, H, W) 或 (H, W, C)，如果有则叠加显示。
            alpha: 四象限图透明度（0-1），默认 0.6。
            original_signal: 原始 1D 波形数据 (1, C, L) 或 (C, L)，用于叠加显示。
    
        Returns:
            Figure 对象（save_path=None 时）或 None（保存后 close）。
        """
        arr = quadrant_map.detach().cpu().float().numpy()
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap_q = ListedColormap(self._QUADRANT_COLORS)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_q.N)
    
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
        # 处理 1D 序列数据：(1, L, 1) -> 展平为水平条形图
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 1:
            arr_1d = arr[0, :, 0]  # (L,)
            L = len(arr_1d)
            
            # 如果有原始波形数据，创建单图并叠加四象限背景
            if original_signal is not None:
                fig, ax1 = plt.subplots(figsize=(self.figsize[0], self.figsize[1]*0.8), 
                                        dpi=self.dpi)
                
                # 先绘制四象限颜色背景
                # 创建与信号长度对应的四象限背景
                bg_data = arr_1d.reshape(1, -1)  # (1, L)
                # 使用 extent 让颜色块精确对齐 x 轴
                ax1.imshow(bg_data, cmap=cmap_q, norm=norm,
                          interpolation="nearest", aspect="auto",
                          extent=[0, L, -1.2, 1.2],  # x: [0, L], y: [-1.2, 1.2]
                          alpha=0.25)  # 透明度较低，不干扰波形
                
                # 处理原始波形
                sig = original_signal.detach().cpu().float()
                
                # 处理 batch 维度：取第一个样本
                if sig.ndim == 3:
                    sig = sig[0]  # (B, C, L) -> (C, L)
    
                # 处理多导联：取平均或第一个
                if sig.ndim == 2:
                    if sig.shape[0] > 1:
                        sig = sig.mean(dim=0)  # (C, L) -> (L,) 多导联平均
                    else:
                        sig = sig.squeeze(0)  # (1, L) -> (L,)
                
                # 归一化波形到 [-1, 1]
                sig_max = sig.abs().max()
                if sig_max > 0:
                    sig = sig / sig_max
                
                # 绘制波形（在四象限背景之上）
                ax1.plot(sig.numpy(), color='#2c3e50', linewidth=1.0, alpha=0.9, zorder=10)
                ax1.set_ylabel("Normalized Amplitude", fontsize=9)
                ax1.set_title(title if title else "Original Signal + Quadrant Analysis", fontsize=11)
                ax1.set_xlim(0, L)
                ax1.set_ylim(-1.15, 1.15)
                ax1.grid(True, alpha=0.2, zorder=5)
                ax1.set_xlabel("Sequence Position (Token)", fontsize=9)
                
                # 添加图例
                labels = ["Core Discriminative", "Redundant Attention",
                          "Potential Influence", "Irrelevant"]
                import matplotlib.patches as mpatches
                patches = [mpatches.Patch(color=self._QUADRANT_COLORS[i], label=labels[i], alpha=0.25)
                           for i in range(4)]
                ax1.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9)
                
                plt.tight_layout()
                return self._save_and_close(fig, save_path)
            
            # 无波形数据：仅显示四象限（继续执行后续代码）
            im = ax.imshow(arr_1d.reshape(1, -1), cmap=cmap_q, norm=norm,
                          interpolation="nearest", aspect="auto")
            ax.set_xlim(0, L)
            ax.set_yticks([])
            ax.set_xlabel("Sequence Position (Token)")
        else:
            # 2D 图像数据：正常 imshow
            # 如果有背景图，先显示原图
            if background_image is not None:
                bg = background_image.detach().cpu().float()
                if bg.ndim == 3 and bg.shape[0] in (1, 3, 4):
                    bg = bg.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                bg_arr = bg.numpy()
                if bg_arr.max() > 1.0:
                    bg_arr = bg_arr / 255.0
                bg_arr = np.clip(bg_arr, 0, 1)
                ax.imshow(bg_arr)
            
            # 叠加四象限图
            im = ax.imshow(arr, cmap=cmap_q, norm=norm, interpolation="nearest", alpha=alpha)
            ax.axis("off")
            
        if title:
            ax.set_title(title, fontsize=11)
            
        # 添加图例
        labels = ["Core Discriminative", "Redundant Attention",
                  "Potential Influence", "Irrelevant"]
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=self._QUADRANT_COLORS[i], label=labels[i])
                   for i in range(4)]
        ax.legend(handles=patches, loc="lower right", fontsize=7)
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
