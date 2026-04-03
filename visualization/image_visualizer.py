"""
图像模型专用可视化模块

专门为 2D 图像 Transformer 模型（如 SwinIR、ViT、Swin Transformer 等）设计的可视化工具。
提供 patch 重组、窗口注意力展开、图像质量对比、PSNR/SSIM 指标等图像特有的可视化功能。
"""

import math
import os
import logging
logger = logging.getLogger(__name__)

from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from numpy import ndarray as NDArray
from torch import Tensor


def _to_numpy(tensor: Tensor) -> NDArray:
    """将张量转为 numpy 数组。"""
    return tensor.detach().cpu().float().numpy()


def _normalize_01(arr: NDArray) -> NDArray:
    """Min-Max 归一化到 [0, 1]。"""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _percentile_clip_normalize(arr: NDArray, low: float = 1.0, high: float = 99.0) -> NDArray:
    """百分位裁剪后归一化到 [0, 1]。"""
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    clipped = np.clip(arr, lo, hi)
    return _normalize_01(clipped)


def compute_psnr(img1: Tensor, img2: Tensor, max_val: float = 1.0) -> float:
    """
    计算 Peak Signal-to-Noise Ratio (PSNR)。
    
    Args:
        img1: 原始图像张量 (C, H, W) 或 (B, C, H, W)
        img2: 重建图像张量 (C, H, W) 或 (B, C, H, W)
        max_val: 像素最大值，默认 1.0
    
    Returns:
        float: PSNR 值（dB）
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr


def compute_ssim(img1: Tensor, img2: Tensor, window_size: int = 11) -> float:
    """
    计算 Structural Similarity Index (SSIM)。
    
    Args:
        img1: 原始图像张量 (C, H, W) 或 (B, C, H, W)
        img2: 重建图像张量 (C, H, W) 或 (B, C, H, W)
        window_size: 滑动窗口大小
    
    Returns:
        float: SSIM 值 [0, 1]
    """
    # 简化的 SSIM 实现
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


class ImageVisualizer:
    """
    图像模型专用可视化工具。
    
    专门针对 2D 图像 Transformer 模型（SwinIR、ViT、Swin Transformer 等）设计，提供：
    - Patch 重组和可视化
    - 窗口注意力展开为全局图像空间
    - 图像质量对比（原图 vs 重建图）
    - PSNR/SSIM 指标显示
    - 注意力热力图叠加到原始图像
    - 多层注意力对比面板
    
    与 HeatmapRenderer 的区别：
    - HeatmapRenderer 通用性强但偏向 1D 序列
    - ImageVisualizer 专为 2D 图像优化，支持图像特有操作
    """
    
    def __init__(self,
                 cmap: str = "viridis",
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 150) -> None:
        """
        初始化图像可视化器。
        
        Args:
            cmap: matplotlib 颜色映射方案，默认 "viridis"
            figsize: 默认图像尺寸 (宽, 高)，单位英寸
            dpi: 输出分辨率，默认 150
        """
        self.cmap = cmap
        self.figsize = figsize
        self.dpi = dpi
    
    # ==================================================================
    # Patch 重组和窗口注意力展开
    # ==================================================================
    
    def visualize_patches(self,
                         image: Tensor,
                         patch_size: int = 16,
                         token_importance: Optional[Tensor] = None,
                         save_path: Optional[str] = None) -> NDArray:
        """
        可视化图像的 patch 划分。
        
        将图像分割为 patch 网格，并在原图上绘制 patch 边界。
        可选地叠加注意力×梯度的重要性得分 mask。
        
        Args:
            image: 图像张量 (C, H, W) 或 (H, W, C)
            patch_size: patch 大小
            token_importance: token 重要性得分 (L,) 或 (B, L)，已计算的注意力×梯度融合结果
            save_path: 保存路径
        
        Returns:
            NDArray: 可视化图像 RGB 数组
        """
        img = image.detach().cpu().float()
        
        # 统一为 (H, W, C)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)
        
        img_arr = img.numpy()
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0
        img_arr = np.clip(img_arr, 0, 1)
        
        H, W = img_arr.shape[:2]
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        
        # 处理 token_importance：将 1D 重要性分数 reshape 到 2D patch 网格
        importance_mask_2d = None
        if token_importance is not None:
            importance = token_importance.detach().cpu().float()
            
            # 如果是 2D (B, L)，取第一个样本
            if importance.dim() == 2:
                # 确保形状是 (B, L) 而不是 (L, B)
                if importance.shape[0] > importance.shape[1]:
                    # 可能是 (L, B)，转置为 (B, L)
                    importance = importance.T
                importance = importance[0]  # (L,)
            
            # 检查 token 数量是否匹配 patch 数量
            L = importance.shape[0]
            if L == total_patches:
                # 直接 reshape 到 2D 网格
                importance_mask_2d = importance.view(num_patches_h, num_patches_w).numpy()
            elif L > total_patches:
                # 如果 token 数多于 patch 数，取前 total_patches 个
                importance_mask_2d = importance[:total_patches].view(num_patches_h, num_patches_w).numpy()
            else:
                # 如果 token 数少于 patch 数，使用插值
                logger.warning(
                    f"Token 数量 ({L}) 少于 patch 数量 ({total_patches})，使用插值对齐"
                )
                importance_tensor = importance.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
                importance_resized = F.interpolate(
                    importance_tensor,
                    size=(num_patches_h, num_patches_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                importance_mask_2d = importance_resized
        
        # 创建图表：如果有重要性 mask，使用双子图布局
        if importance_mask_2d is not None:
            fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0]*2, self.figsize[1]), dpi=self.dpi)
            
            # 左图：原始 Patch 网格
            ax = axes[0]
            ax.imshow(img_arr)
            for i in range(1, num_patches_h):
                ax.axhline(y=i * patch_size, color='yellow', linewidth=0.5, alpha=0.7)
            for j in range(1, num_patches_w):
                ax.axvline(x=j * patch_size, color='yellow', linewidth=0.5, alpha=0.7)
            ax.set_title(f'Patch Grid ({num_patches_h}×{num_patches_w})',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # 右图：Patch 网格 + 重要性 mask 叠加
            ax2 = axes[1]
            ax2.imshow(img_arr)
            
            # 将重要性 mask 上采样到图像尺寸
            importance_tensor = torch.from_numpy(_normalize_01(importance_mask_2d)).unsqueeze(0).unsqueeze(0)
            importance_resized = F.interpolate(
                importance_tensor,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # 映射为彩色热力图
            cmap_fn = cm.get_cmap('hot')
            importance_rgb = cmap_fn(importance_resized)[:, :, :3]
            
            # Alpha 混合
            alpha = 0.5
            blended = (1 - alpha) * img_arr + alpha * importance_rgb
            blended = np.clip(blended, 0, 1)
            
            ax2.imshow(blended)
            for i in range(1, num_patches_h):
                ax2.axhline(y=i * patch_size, color='cyan', linewidth=0.5, alpha=0.7)
            for j in range(1, num_patches_w):
                ax2.axvline(x=j * patch_size, color='cyan', linewidth=0.5, alpha=0.7)
            ax2.set_title(f'Importance Mask (Attention × Gradient)\n'                         f'Overlay with Patch Grid',
                         fontsize=11, fontweight='bold')
            ax2.axis('off')
            
            # 添加颜色条
            im = ax2.imshow(importance_resized, cmap='hot', alpha=0.0)  # 仅用于获取颜色条
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Importance Score')
            im.remove()  # 移除多余的图像
            
            plt.suptitle('Patch Visualization with Importance Mask', 
                        fontsize=13, fontweight='bold', y=1.02)
        else:
            # 仅有 Patch 网格
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.imshow(img_arr)
            
            # 绘制 patch 网格
            for i in range(1, num_patches_h):
                ax.axhline(y=i * patch_size, color='yellow', linewidth=0.5, alpha=0.7)
            for j in range(1, num_patches_w):
                ax.axvline(x=j * patch_size, color='yellow', linewidth=0.5, alpha=0.7)
            
            ax.set_title(f'Patch Visualization (Patch Size: {patch_size}×{patch_size})\n'
                        f'Grid: {num_patches_h}×{num_patches_w} = {total_patches} patches',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    def unfold_window_attention_to_image(self,
                                        window_attention: Tensor,
                                        num_windows_h: int,
                                        num_windows_w: int,
                                        window_size: int,
                                        save_path: Optional[str] = None) -> NDArray:
        """
        将窗口注意力展开为全局图像空间。
        
        将 Swin Transformer 的窗口注意力 (B, num_heads, num_windows, window_size^2, window_size^2)
        重组为全局注意力图 (B, num_heads, H, W)。
        
        Args:
            window_attention: 窗口注意力张量
                - 形状: (B, num_heads, num_windows, window_size^2, window_size^2) 或
                - 形状: (num_windows, window_size^2, window_size^2)
            num_windows_h: 垂直方向窗口数量
            num_windows_w: 水平方向窗口数量
            window_size: 窗口大小
            save_path: 保存路径
        
        Returns:
            NDArray: 全局注意力图 RGB 数组
        """
        attn = window_attention.detach().cpu().float()
        
        # 处理不同的输入形状
        if attn.dim() == 3:
            # (num_windows, window_size^2, window_size^2) → 添加 batch 和 head 维度
            attn = attn.unsqueeze(0).unsqueeze(0)
        elif attn.dim() == 5:
            # 已经是 (B, num_heads, num_windows, ws^2, ws^2)
            pass
        else:
            raise ValueError(f"不支持的注意力形状: {attn.shape}")
        
        B, num_heads, num_windows, ws2_q, ws2_k = attn.shape
        window_size_actual = int(ws2_q ** 0.5)
        
        if window_size_actual * window_size_actual != ws2_q:
            raise ValueError(f"window_size^2 不是完全平方数: {ws2_q}")
        
        # 计算全局图像尺寸
        H_global = num_windows_h * window_size_actual
        W_global = num_windows_w * window_size_actual
        
        # 对每个 batch 和 head 进行展开
        global_attention_maps = []
        
        for b in range(B):
            for h in range(num_heads):
                # 取单个窗口注意力 (num_windows, ws^2, ws^2)
                window_attn = attn[b, h]
                
                # 重组为全局网格
                # 先将 num_windows 维度展开为 (num_windows_h, num_windows_w)
                # 然后将 window_size^2 展开为 (window_size, window_size)
                # 形状变换: (num_windows, ws^2, ws^2) → (num_windows_h, num_windows_w, ws, ws, ws, ws)
                window_attn_reshaped = window_attn.view(
                    num_windows_h, num_windows_w,
                    window_size_actual, window_size_actual,
                    window_size_actual, window_size_actual
                )
                
                # 重新排列维度，将窗口维度和窗口内维度交织
                # (num_windows_h, ws_q, num_windows_w, ws_q, num_windows_h, ws_k, num_windows_w, ws_k)
                # 注意：这里我们需要将 query 和 key 的窗口维度和窗口内维度分别组合
                # 为了简化，我们只对 query 维度进行空间重组，对 key 维度取平均
                
                # 方法：先对 key 的所有维度取平均，得到 query 的空间分布
                # window_attn_reshaped: (nh, nw, ws, ws, ws, ws)
                # 对最后 4 个维度（key）取平均 → (nh, nw, ws, ws)
                query_spatial = window_attn_reshaped.mean(dim=[1, 3, 4, 5])  # (num_windows_h, ws,)
                
                # 更好的方法：对 query 和 key 的窗口内维度进行合理重组
                # 让我们使用另一种策略：将窗口注意力视为局部注意力，直接对每个窗口的注意力取平均
                # 然后将这些平均值排列成全局网格
                
                # 对每个窗口的注意力矩阵取平均（得到该窗口的注意力强度）
                window_importance = window_attn.mean(dim=[1, 2])  # (num_windows,)
                
                # 将窗口重要性重组为 2D 网格
                window_grid = window_importance.view(num_windows_h, num_windows_w)
                
                # 使用双线性插值将窗口级别的注意力上采样到像素级别
                window_grid_tensor = window_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, nh, nw)
                global_attn_2d = F.interpolate(
                    window_grid_tensor,
                    size=(H_global, W_global),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()  # (H_global, W_global)
                
                global_attention_maps.append(global_attn_2d)
        
        # 将所有注意力图堆叠
        if len(global_attention_maps) == 1:
            final_attn = global_attention_maps[0]
        else:
            final_attn = torch.stack(global_attention_maps).mean(dim=0)
        
        # 可视化
        arr_norm = _percentile_clip_normalize(final_attn.numpy())
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(arr_norm, cmap=self.cmap, aspect='auto', interpolation='nearest')
        ax.set_title(f'Global Window Attention Map\n'
                    f'Shape: {H_global}×{W_global} | Windows: {num_windows_h}×{num_windows_w}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Width', fontsize=10)
        ax.set_ylabel('Height', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention Weight')
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    # ==================================================================
    # 图像质量对比
    # ==================================================================
    
    def visualize_image_comparison(self,
                                   original: Tensor,
                                   reconstructed: Tensor,
                                   metrics: Optional[Dict[str, float]] = None,
                                   save_path: Optional[str] = None) -> NDArray:
        """
        可视化原图与重建图的对比。
        
        并排显示原图、重建图和差异图，可选显示质量指标。
        
        Args:
            original: 原始图像 (C, H, W)
            reconstructed: 重建图像 (C, H, W)
            metrics: 质量指标字典，如 {'PSNR': 28.5, 'SSIM': 0.92}
            save_path: 保存路径
        
        Returns:
            NDArray: 对比图 RGB 数组
        """
        orig = original.detach().cpu().float()
        recon = reconstructed.detach().cpu().float()
        
        # 统一为 (H, W, C)
        if orig.ndim == 3 and orig.shape[0] in (1, 3, 4):
            orig = orig.permute(1, 2, 0)
        if recon.ndim == 3 and recon.shape[0] in (1, 3, 4):
            recon = recon.permute(1, 2, 0)
        
        orig_arr = orig.numpy()
        recon_arr = recon.numpy()
        
        # 归一化到 [0, 1]
        if orig_arr.max() > 1.0:
            orig_arr = orig_arr / 255.0
        if recon_arr.max() > 1.0:
            recon_arr = recon_arr / 255.0
        
        orig_arr = np.clip(orig_arr, 0, 1)
        recon_arr = np.clip(recon_arr, 0, 1)
        
        # 计算差异图
        diff_arr = np.abs(orig_arr - recon_arr)
        diff_arr = _normalize_01(diff_arr)
        
        # 如果没有提供指标，计算 PSNR 和 SSIM
        if metrics is None:
            metrics = {}
            if 'PSNR' not in metrics:
                orig_tensor = original
                recon_tensor = reconstructed
                if orig_tensor.dim() == 3:
                    orig_tensor = orig_tensor.unsqueeze(0)
                if recon_tensor.dim() == 3:
                    recon_tensor = recon_tensor.unsqueeze(0)
                metrics['PSNR'] = compute_psnr(orig_tensor, recon_tensor)
            if 'SSIM' not in metrics:
                metrics['SSIM'] = compute_ssim(original, reconstructed)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        
        # 原图
        axes[0].imshow(orig_arr)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 重建图
        axes[1].imshow(recon_arr)
        axes[1].set_title('Reconstructed Image', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 差异图
        im = axes[2].imshow(diff_arr, cmap='hot', interpolation='nearest')
        axes[2].set_title('Difference Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 添加指标文本
        metrics_text = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='monospace')
        
        plt.suptitle('Image Quality Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    # ==================================================================
    # PSNR/SSIM 指标可视化
    # ==================================================================
    
    def visualize_quality_metrics(self,
                                 psnr_values: Optional[list] = None,
                                 ssim_values: Optional[list] = None,
                                 sample_labels: Optional[list] = None,
                                 save_path: Optional[str] = None) -> NDArray:
        """
        可视化图像质量指标（PSNR/SSIM）的分布。
        
        Args:
            psnr_values: PSNR 值列表
            ssim_values: SSIM 值列表
            sample_labels: 样本标签列表
            save_path: 保存路径
        
        Returns:
            NDArray: 指标图 RGB 数组
        """
        if psnr_values is None and ssim_values is None:
            raise ValueError("至少需要提供 psnr_values 或 ssim_values 之一")
        
        n_metrics = sum(1 for v in [psnr_values, ssim_values] if v is not None)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        plot_idx = 0
        
        if psnr_values is not None:
            ax = axes[plot_idx]
            if sample_labels is not None:
                x = range(len(psnr_values))
                ax.bar(x, psnr_values, color='skyblue', edgecolor='navy', alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=8)
            else:
                ax.hist(psnr_values, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
            
            ax.set_title(f'PSNR Distribution\nMean: {np.mean(psnr_values):.2f} dB',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Sample' if sample_labels else 'PSNR (dB)', fontsize=10)
            ax.set_ylabel('PSNR (dB)' if sample_labels else 'Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        if ssim_values is not None:
            ax = axes[plot_idx]
            if sample_labels is not None:
                x = range(len(ssim_values))
                ax.bar(x, ssim_values, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=8)
            else:
                ax.hist(ssim_values, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            
            ax.set_title(f'SSIM Distribution\nMean: {np.mean(ssim_values):.4f}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Sample' if sample_labels else 'SSIM', fontsize=10)
            ax.set_ylabel('SSIM' if sample_labels else 'Count', fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        plt.suptitle('Image Quality Metrics', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    # ==================================================================
    # 注意力叠加到原始图像
    # ==================================================================
    
    def overlay_attention_on_image(self,
                                   attention: Tensor,
                                   image: Tensor,
                                   alpha: float = 0.5,
                                   attention_type: str = 'query_mean',
                                   save_path: Optional[str] = None) -> NDArray:
        """
        将注意力热力图叠加到原始图像上。
        
        支持多种注意力聚合方式：
        - query_mean: 对 query 维度取平均
        - key_mean: 对 key 维度取平均
        - mean: 对所有维度取平均
        
        Args:
            attention: 注意力张量
                - 2D: (H, W) 直接使用
                - 3D: (H, W, H, W) 需要聚合
                - 4D: (B, num_heads, H, W) 对 batch 和 heads 取平均
            image: 原始图像 (C, H, W) 或 (H, W, C)
            alpha: 热力图透明度
            attention_type: 注意力聚合方式
            save_path: 保存路径
        
        Returns:
            NDArray: 叠加后的 RGB 图像
        """
        # 处理注意力张量
        attn = attention.detach().cpu().float()
        
        # 如果是 4D (B, num_heads, H, W)，对 batch 和 heads 取平均
        if attn.dim() == 4:
            attn = attn.mean(dim=[0, 1])  # (H, W)
        # 如果是 3D，根据类型聚合
        elif attn.dim() == 3:
            if attention_type == 'query_mean':
                attn = attn.mean(dim=0)  # 对第一维取平均
            elif attention_type == 'key_mean':
                attn = attn.mean(dim=1)  # 对第二维取平均
            else:
                attn = attn.mean(dim=[0, 1])
        elif attn.dim() != 2:
            raise ValueError(f"不支持的注意力维度: {attn.shape}")
        
        attn_arr = _percentile_clip_normalize(attn.numpy())
        
        # 处理图像
        img = image.detach().cpu().float()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)
        img_arr = img.numpy()
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0
        img_arr = np.clip(img_arr, 0, 1)
        
        # 确保图像是 3 通道
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[2] == 1:
            img_arr = np.concatenate([img_arr] * 3, axis=-1)
        
        # 插值注意力图到图像尺寸
        img_h, img_w = img_arr.shape[:2]
        attn_tensor = torch.from_numpy(attn_arr).unsqueeze(0).unsqueeze(0)
        attn_resized = F.interpolate(attn_tensor, size=(img_h, img_w),
                                     mode='bilinear', align_corners=False)
        attn_resized = attn_resized.squeeze().numpy()
        
        # 将注意力图映射为 RGB
        cmap_fn = cm.get_cmap(self.cmap)
        attn_rgb = cmap_fn(attn_resized)[:, :, :3]
        
        # Alpha 混合
        blended = (1 - alpha) * img_arr + alpha * attn_rgb
        blended = np.clip(blended, 0, 1)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        axes[0].imshow(img_arr)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(blended)
        axes[1].set_title(f'Attention Overlay (α={alpha})', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle('Attention Visualization on Image', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    # ==================================================================
    # 多层注意力对比面板
    # ==================================================================
    
    def visualize_multi_layer_attention(self,
                                       attention_maps: Dict[int, Tensor],
                                       titles: Optional[Dict[int, str]] = None,
                                       num_cols: int = 4,
                                       save_path: Optional[str] = None) -> NDArray:
        """
        可视化多层注意力对比面板。
        
        Args:
            attention_maps: {layer_idx: attention_tensor} 字典
                每个 tensor 形状为 (H, W) 或 (B, num_heads, H, W)
            titles: {layer_idx: title} 字典，自定义标题
            num_cols: 每行列数
            save_path: 保存路径
        
        Returns:
            NDArray: 面板 RGB 数组
        """
        if not attention_maps:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.axis('off')
            rgb = self._fig_to_rgb(fig)
            self._save_fig(fig, save_path)
            return rgb
        
        sorted_layers = sorted(attention_maps.keys())
        n = len(sorted_layers)
        num_cols = min(num_cols, n)
        num_rows = math.ceil(n / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols,
                                figsize=(num_cols * 3.5, num_rows * 3),
                                dpi=self.dpi, squeeze=False)
        
        for i, layer_idx in enumerate(sorted_layers):
            row, col = divmod(i, num_cols)
            ax = axes[row][col]
            
            attn = attention_maps[layer_idx].detach().cpu().float()
            
            # 处理多维注意力
            if attn.dim() == 4:
                attn = attn.mean(dim=[0, 1])  # (H, W)
            elif attn.dim() == 3:
                attn = attn.mean(dim=0)  # (H, W)
            
            arr_norm = _percentile_clip_normalize(attn.numpy())
            
            im = ax.imshow(arr_norm, cmap=self.cmap, aspect='auto', interpolation='nearest')
            
            # 标题
            if titles and layer_idx in titles:
                title = titles[layer_idx]
            else:
                title = f'Layer {layer_idx}'
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 关闭多余的子图
        for i in range(n, num_rows * num_cols):
            row, col = divmod(i, num_cols)
            axes[row][col].axis('off')
        
        plt.suptitle('Multi-Layer Attention Comparison', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        rgb = self._fig_to_rgb(fig)
        self._save_fig(fig, save_path)
        return rgb
    
    # ==================================================================
    # 辅助方法
    # ==================================================================
    
    def _fig_to_rgb(self, fig: Figure) -> NDArray:
        """将 Figure 渲染为 (H, W, 3) uint8 RGB 数组。"""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, :3]
    
    def _save_fig(self, fig: Figure, save_path: Optional[str]) -> None:
        """保存图像并关闭 Figure。"""
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
