"""
时序数据通用可视化工具

支持任何 1D 序列数据的注意力叠加和关键点标注。

设计原则：
- 通用方法：只使用统计学检测，不依赖领域知识
- 职责单一：只做可视化和基础信号处理
- 易于使用：开箱即用，无需配置

使用示例：
    ts_vis = TimeSeriesVisualizer(sampling_rate=500)
    
    # 渲染注意力叠加图
    fig = ts_vis.render_sequence_with_attention(
        sequence=data,              # (L,) 或 (C, L)
        attention_weights=importance,  # (L,)
        save_path="output.png",
    )
    
    # 自动检测并标注峰值
    key_positions = ts_vis.detect_key_positions(data, method='peaks')
    # key_positions = {
    #     "peaks": [(start, end), ...],
    #     "valleys": [(start, end), ...],
    # }
"""

from typing import Dict, List, Optional, Tuple
import os
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.signal import find_peaks


class TimeSeriesVisualizer:
    """时序数据通用可视化器"""
    
    def __init__(
        self,
        sampling_rate: Optional[float] = None,
        time_unit: str = "steps",
    ):
        """
        Args:
            sampling_rate: 采样率 (Hz)，None 表示用时间步索引
            time_unit: X 轴单位 ('seconds', 'milliseconds', 'steps')
        """
        self.sampling_rate = sampling_rate
        self.time_unit = time_unit
    
    @staticmethod
    def detect_peaks(
        signal: np.ndarray,
        distance: int = 50,
        prominence: float = 0.1,
    ) -> List[Tuple[int, int]]:
        """
        检测序列中的峰值区域（通用方法）
        
        Args:
            signal: 输入信号 (L,)
            distance: 峰值间最小距离（点数）
            prominence: 峰值突出度阈值
        
        Returns:
            [(start_idx, end_idx), ...] - 每个峰值区域的起止索引
        """
        peaks, properties = find_peaks(signal, distance=distance, prominence=prominence)
        
        peak_regions = []
        for peak in peaks:
            # 峰值前后各扩展 10 个点作为区域
            start = max(0, peak - 10)
            end = min(len(signal), peak + 10)
            peak_regions.append((start, end))
        
        return peak_regions
    
    @staticmethod
    def detect_valleys(
        signal: np.ndarray,
        distance: int = 50,
        prominence: float = 0.1,
    ) -> List[Tuple[int, int]]:
        """检测谷值区域（对负信号使用 find_peaks）"""
        return TimeSeriesVisualizer.detect_peaks(-signal, distance, prominence)
    
    @staticmethod
    def detect_change_points(
        signal: np.ndarray,
        threshold_factor: float = 2.0,
    ) -> List[Tuple[int, int]]:
        """
        检测序列中的突变点（通用方法）
        
        Args:
            signal: 输入信号 (L,)
            threshold_factor: 差分阈值因子（均值 + factor*标准差）
        
        Returns:
            [(start_idx, end_idx), ...] - 每个突变点的邻域
        """
        diff = np.abs(np.diff(signal))
        threshold = np.mean(diff) + threshold_factor * np.std(diff)
        change_points = np.where(diff > threshold)[0]
        
        change_regions = []
        for cp in change_points:
            start = max(0, cp - 5)
            end = min(len(signal), cp + 5)
            change_regions.append((start, end))
        
        return change_regions
    
    def detect_key_positions(
        self,
        signal: np.ndarray,
        method: str = "auto",
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        自动检测关键位置（通用方法）
        
        Args:
            signal: 输入信号 (L,)
            method: 检测方法
                - 'peaks': 只检测峰值
                - 'valleys': 只检测谷值
                - 'changes': 只检测突变点
                - 'all': 检测所有类型
                - 'auto': 自动选择（默认同'all'）
        
        Returns:
            {
                "peaks": [(start, end), ...],
                "valleys": [(start, end), ...],
                "change_points": [(start, end), ...],
            }
        """
        result = {}
        
        if method in ["peaks", "all", "auto"]:
            result["peaks"] = self.detect_peaks(signal)
        
        if method in ["valleys", "all", "auto"]:
            result["valleys"] = self.detect_valleys(signal)
        
        if method in ["changes", "all", "auto"]:
            result["change_points"] = self.detect_change_points(signal)
        
        return result
    
    def render_sequence_with_attention(
        self,
        sequence: Tensor,           # (C, L) 多通道或 (L,) 单通道
        attention_weights: Tensor,  # (L,) 或 (H, L) - 每时间点的重要性
        gradient_weights: Optional[Tensor] = None,  # (L,) - 梯度重要性 (可选)
        fusion_method: str = "gradcam",  # 'gradcam' | 'sum' | 'multiply'
        key_positions: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        channel_names: List[str] = None,  # ['Lead I', 'Lead II'] 或 ['Embed_0', ...]
        title: str = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> Figure:
        """
        渲染时序数据 + 注意力热力图叠加
        
        支持多通道分别显示，每行一个通道
        
        Args:
            sequence: 原始输入序列
            attention_weights: 注意力权重（已融合为每时间点的标量）
            gradient_weights: 梯度权重（可选，用于 Grad-CAM 融合）
            fusion_method: 注意力和梯度的融合方式
            key_positions: 关键位置字典（由外部检测器提供）
                {
                    "peaks": [(start, end), ...],
                    "valleys": [(start, end), ...],
                    "change_points": [(start, end), ...],
                }
            channel_names: 通道名称列表
            title: 图表标题
            save_path: 保存路径
            figsize: 图像尺寸
        
        Returns:
            Figure 对象
        """
        # 1. 处理序列数据
        if sequence.dim() == 2:
            sequences = [sequence[i] for i in range(sequence.shape[0])]
            if channel_names is None:
                channel_names = [f"Channel_{i}" for i in range(len(sequences))]
        else:
            sequences = [sequence]
            if channel_names is None:
                channel_names = ["Input Sequence"]
        
        # 2. 创建多子图
        fig, axes = plt.subplots(
            nrows=len(sequences),
            ncols=1,
            figsize=figsize,
            sharex=True,
            dpi=150,
        )
        if len(sequences) == 1:
            axes = [axes]
        
        # 3. 对每个通道绘图
        for ax, seq, name in zip(axes, sequences, channel_names):
            # 绘制原始波形
            x = np.arange(len(seq))
            ax.plot(x, seq.cpu().numpy(), color='black', linewidth=0.8, label=name)
            
            # 叠加注意力热力图背景
            self._overlay_attention(ax, attention_weights, seq.shape[0])
            
            # 标注关键区段（如果有提供）
            if key_positions is not None:
                self.annotate_key_segments(
                    ax,
                    key_positions,
                    attention_weights,
                    highlight_color="yellow",
                    alpha=0.3,
                )
            
            # 设置坐标轴
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # X 轴标签和标题
        if self.sampling_rate is not None:
            x = np.arange(len(sequences[0]))
            time_axis = x / self.sampling_rate
            axes[-1].set_xticks(time_axis[::int(self.sampling_rate)])
            axes[-1].set_xticklabels([f"{t:.1f}" for t in time_axis[::int(self.sampling_rate)]])
            axes[-1].set_xlabel("Time (seconds)")
        else:
            axes[-1].set_xlabel("Time Steps")
        
        fig.suptitle(title or "Sequence Attention Overlay", fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存或返回
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def _overlay_attention(
        self,
        ax: Axes,
        attention_weights: Tensor,
        num_points: int,
        cmap_name: str = "Reds",
        alpha: float = 0.5,
    ):
        """
        在轴上叠加注意力热力图背景
        
        Args:
            ax: matplotlib 轴对象
            attention_weights: 注意力权重 (L,)
            num_points: 序列长度
            cmap_name: 颜色映射
            alpha: 透明度
        """
        # 归一化到 [0, 1]
        attn = attention_weights.cpu().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # 如果序列长度不匹配，进行插值
        if len(attn) != num_points:
            from scipy.interpolate import interp1d
            f = interp1d(np.linspace(0, 1, len(attn)), attn, kind='linear')
            attn = f(np.linspace(0, 1, num_points))
        
        # 逐段绘制颜色背景
        cmap = plt.get_cmap(cmap_name)
        for i in range(num_points - 1):
            ax.axvspan(i, i+1, 
                      facecolor=cmap(attn[i]), 
                      alpha=alpha,
                      linewidth=0)
    
    def annotate_key_segments(
        self,
        ax: Axes,
        key_positions: Dict[str, List[Tuple[int, int]]],
        attention_weights: Tensor,
        highlight_color: str = "yellow",
        alpha: float = 0.3,
        min_importance: float = 0.3,
    ):
        """
        在已有轴上标注关键区段
        
        Args:
            ax: matplotlib 轴对象
            key_positions: 关键位置字典（由外部提供）
            attention_weights: 注意力权重（用于筛选重要区段）
            highlight_color: 高亮颜色
            alpha: 透明度
            min_importance: 最小重要性阈值，低于此值不标注
        """
        for segment_type, positions in key_positions.items():
            for start_idx, end_idx in positions:
                # 确保索引在有效范围内
                start_idx = max(0, min(start_idx, len(attention_weights)-1))
                end_idx = max(start_idx+1, min(end_idx, len(attention_weights)))
                
                # 计算该区段的平均重要性
                seg_importance = attention_weights[start_idx:end_idx].mean().item()
                
                # 只标注重要性较高的区段
                if seg_importance > min_importance:
                    # 添加高亮背景
                    ax.axvspan(start_idx, end_idx, 
                              facecolor=highlight_color, 
                              alpha=alpha,
                              linewidth=0)
                    
                    # 添加标注文字
                    ax.annotate(
                        f"{segment_type}\n(imp={seg_importance:.2f})",
                        xy=((start_idx + end_idx) / 2, 0),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='white', 
                                 edgecolor=highlight_color, 
                                 alpha=0.8),
                    )
    
    def render_multi_layer_comparison(
        self,
        layer_sequences: Dict[int, Tensor],  # {layer_idx: (L,)}
        sequence: Tensor,
        save_path: str = None,
    ) -> Figure:
        """
        多层注意力对比图
        
        每行显示一个 Transformer 层的注意力分布
        
        Args:
            layer_sequences: 各层的注意力序列 {layer_idx: tensor}
            sequence: 原始序列（用于参考）
            save_path: 保存路径
        
        Returns:
            Figure 对象
        """
        sorted_layers = sorted(layer_sequences.keys())
        num_layers = len(sorted_layers)
        
        fig, axes = plt.subplots(
            nrows=num_layers,
            ncols=1,
            figsize=(12, 2 * num_layers),
            sharex=True,
            dpi=150,
        )
        if num_layers == 1:
            axes = [axes]
        
        for idx, layer_idx in enumerate(sorted_layers):
            ax = axes[idx]
            attn = layer_sequences[layer_idx].cpu().numpy()
            
            # 归一化
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            
            # 绘制热力图
            x = np.arange(len(attn))
            ax.bar(x, attn, color='steelblue', alpha=0.7)
            ax.set_ylabel(f'Layer {layer_idx}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time Steps")
        fig.suptitle("Multi-Layer Attention Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
