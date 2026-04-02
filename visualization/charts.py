"""
通用绘图工具模块（plot_utils 风格）

根据 design.md §3.6.2，本文件提供一组独立的绘图工具函数，
替代原 ChartGenerator 类，以函数式 API 提供更轻量的调用方式。

包含：
- plot_layer_importance      : 层重要性柱状图
- plot_head_scatter          : 注意力头"频率-重要性"散点图
- plot_accumulator_stats     : 累积器统计摘要图
- save_figure                : 图表保存工具
"""

import matplotlib
matplotlib.use('Agg')

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..core.types import AccumulatorState


def plot_layer_importance(
    layer_importance: Dict[int, float],
    title: str = "Layer Importance",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    绘制各层重要性柱状图。

    X 轴为层索引，Y 轴为重要性得分（通常为梯度 L2 范数），
    按层索引升序排列。

    Args:
        layer_importance: 字典 {layer_idx: importance_score}。
        title: 图表标题，默认 "Layer Importance"。
        save_path: 图像保存路径，None 表示不保存，返回 Figure。

    Returns:
        Figure 对象（save_path=None 时），否则 None。
    """
    sorted_items = sorted(layer_importance.items())
    layers = [str(k) for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.6), 4), dpi=150)
    bars = ax.bar(layers, scores, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_ylabel("Importance Score", fontsize=10)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45 if len(layers) > 10 else 0, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return _save_or_return(fig, save_path)


def plot_head_scatter(
    frequency_data: List[Dict[str, Any]],
    importance_data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Head Analysis",
    head_freq: Optional[Any] = None,
    head_concentration: Optional[Any] = None,
) -> Optional[Figure]:
    """
    绘制注意力头"激活频率 - 梯度重要性"二维散点图。

    每个点代表一个注意力头，颜色区分不同层。
    也支持直接传入 head_freq / head_concentration 张量（兼容新接口）。

    Args:
        frequency_data: 频率数据列表，每项含 "layer_idx", "head_idx", "freq"。
        importance_data: 重要性数据列表，每项含 "layer_idx", "head_idx", "importance"。
        save_path: 保存路径，None 则返回 Figure。
        title: 图表标题。
        head_freq: (num_layers, num_heads) 张量，激活频率（新接口）。
        head_concentration: (num_layers, num_heads) 张量，集中度（新接口）。

    Returns:
        Figure 或 None。
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # 新接口：直接传张量
    if head_freq is not None and head_concentration is not None:
        import torch
        freq_arr = head_freq.detach().cpu().float().numpy()   # (L, H)
        conc_arr = head_concentration.detach().cpu().float().numpy()  # (L, H)
        num_layers, num_heads = freq_arr.shape
        cmap = plt.cm.get_cmap("tab10", num_layers)
        for li in range(num_layers):
            ax.scatter(
                freq_arr[li], conc_arr[li],
                c=[cmap(li)] * num_heads,
                label=f"Layer {li}", s=50, alpha=0.8, edgecolors="white", linewidths=0.3
            )
        ax.set_xlabel("Activation Frequency", fontsize=10)
        ax.set_ylabel("Attention Concentration", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        return _save_or_return(fig, save_path)

    # 旧接口：列表字典
    # 构建 importance 查找表
    imp_lookup: Dict[tuple, float] = {
        (d["layer_idx"], d["head_idx"]): d.get("importance", 0.0)
        for d in importance_data
    }

    # 按层分组绘制
    layer_groups: Dict[int, List] = {}
    for d in frequency_data:
        li = d["layer_idx"]
        hi = d["head_idx"]
        freq = d.get("freq", 0.0)
        imp = imp_lookup.get((li, hi), 0.0)
        layer_groups.setdefault(li, []).append((freq, imp))

    num_layers = len(layer_groups)
    cmap = plt.cm.get_cmap("tab10", max(num_layers, 1))
    for i, (li, points) in enumerate(sorted(layer_groups.items())):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, c=[cmap(i)] * len(xs),
                   label=f"Layer {li}", s=50, alpha=0.8,
                   edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Activation Frequency", fontsize=10)
    ax.set_ylabel("Gradient Importance", fontsize=10)
    ax.set_title(title, fontsize=12)
    if num_layers <= 12:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _save_or_return(fig, save_path)


def plot_accumulator_stats(
    accumulator_state: AccumulatorState,
    title: str = "Accumulator Statistics",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    绘制累积器统计摘要图（双子图布局）。

    左图：头激活频率热力矩阵（行=层，列=头）。
    右图：层梯度范数柱状图（X=层索引，Y=梯度范数）。

    Args:
        accumulator_state: AccumulatorState 对象。
        title: 总标题，默认 "Accumulator Statistics"。
        save_path: 保存路径，None 则返回 Figure。

    Returns:
        Figure 或 None。
    """
    freq = accumulator_state.head_activation_freq.detach().cpu().float().numpy()
    grad_norm = accumulator_state.layer_gradient_norm.detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        f"{title} (samples={accumulator_state.sample_count})",
        fontsize=12
    )

    # 左图：头激活频率热力矩阵
    ax_left = axes[0]
    if freq.size > 0:
        im = ax_left.imshow(freq, aspect="auto", cmap="YlOrRd",
                            vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax_left, fraction=0.046, pad=0.04)
        ax_left.set_title("Head Activation Frequency", fontsize=10)
        ax_left.set_xlabel("Head Index", fontsize=9)
        ax_left.set_ylabel("Layer Index", fontsize=9)
        num_layers, num_heads = freq.shape
        ax_left.set_xticks(range(num_heads))
        ax_left.set_yticks(range(num_layers))
    else:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_left.set_title("Head Activation Frequency", fontsize=10)
        ax_left.axis("off")

    # 右图：层梯度范数折线图
    ax_right = axes[1]
    if grad_norm.size > 0:
        layers = list(range(len(grad_norm)))
        ax_right.plot(layers, grad_norm, marker='o', linestyle='-', 
                     color='#DD8452', linewidth=2, markersize=6,
                     markerfacecolor='white', markeredgewidth=1.5, 
                     markeredgecolor='#DD8452')
        ax_right.set_title("Layer Gradient Norm", fontsize=10)
        ax_right.set_xlabel("Layer Index", fontsize=9)
        ax_right.set_ylabel("Gradient L2 Norm", fontsize=9)
        ax_right.set_xticks(layers)
        ax_right.set_xticklabels([str(i) for i in layers],
                                  rotation=45 if len(layers) > 10 else 0,
                                  fontsize=8)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)
        # 添加数值标签
        for i, v in enumerate(grad_norm):
            ax_right.annotate(f'{v:.3f}', (i, v), textcoords="offset points",
                            xytext=(0, 5), ha='center', fontsize=7)
    else:
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_right.set_title("Layer Gradient Norm", fontsize=10)
        ax_right.axis("off")

    plt.tight_layout()
    return _save_or_return(fig, save_path)


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """
    保存 matplotlib 图表到磁盘，自动创建父目录。

    Args:
        fig: 待保存的 matplotlib Figure 对象。
        path: 保存路径（含扩展名，如 "output/fig.png"）。
        dpi: 输出分辨率，默认 150 DPI。
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------

def _save_or_return(fig: Figure, save_path: Optional[str]) -> Optional[Figure]:
    """有 save_path 则保存并 close，返回 None；否则返回 Figure。"""
    if save_path is not None:
        save_figure(fig, save_path)
        return None
    return fig
