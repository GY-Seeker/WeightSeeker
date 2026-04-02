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

from typing import Dict, List, Optional, Any
from matplotlib.figure import Figure

from ..core.types import AccumulatorState


def plot_layer_importance(
    layer_importance: Dict[int, float],
    title: str = "Layer Importance",
    save_path: Optional[str] = None,
) -> Figure:
    """
    绘制各层重要性柱状图。

    X 轴为层索引，Y 轴为重要性得分（通常为梯度 L2 范数），
    按层索引升序排列。

    Args:
        layer_importance: 字典 {layer_idx: importance_score}，
                          importance_score 通常由梯度范数计算得到。
        title: 图表标题，默认 "Layer Importance"。
        save_path: 图像保存路径（含扩展名，如 "output/layer.png"）。
                   None 表示不保存到磁盘。

    Returns:
        Figure: matplotlib Figure 对象，包含单个柱状图子图。
    """
    raise NotImplementedError("待实现")


def plot_head_scatter(
    frequency_data: List[Dict[str, Any]],
    importance_data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """
    绘制注意力头"激活频率 - 梯度重要性"二维散点图。

    每个点代表一个注意力头，位置由 (activation_freq, importance_score) 决定，
    颜色区分不同层，可直观识别"高频高效"与"高频低效"等异常头。

    Args:
        frequency_data: 频率数据列表，每项为字典，至少包含键：
                        - "layer_idx": int，层索引
                        - "head_idx": int，头索引
                        - "freq": float，激活频率 [0, 1]
        importance_data: 重要性数据列表，每项为字典，至少包含键：
                         - "layer_idx": int
                         - "head_idx": int
                         - "importance": float，重要性得分
        save_path: 图像保存路径，None 则不保存。

    Returns:
        Figure: matplotlib Figure 对象，包含单个散点图子图。
                X 轴：激活频率 [0, 1]；Y 轴：梯度重要性得分。
    """
    raise NotImplementedError("待实现")


def plot_accumulator_stats(
    accumulator_state: AccumulatorState,
    save_path: Optional[str] = None,
) -> Figure:
    """
    绘制累积器统计摘要图（双子图布局）。

    布局：
    - 左图：头激活频率热力矩阵（行=层，列=头）
    - 右图：层梯度范数柱状图（X=层索引，Y=梯度范数均值）

    Args:
        accumulator_state: 累积器状态对象（AccumulatorState），包含：
                           - head_activation_freq: (num_layers, num_heads) 激活频率矩阵
                           - layer_gradient_norm:  (num_layers,) 梯度范数向量
                           - sample_count: int，已累积样本数
        save_path: 图像保存路径，None 则不保存。

    Returns:
        Figure: matplotlib Figure 对象，包含 1 行 2 列子图。
    """
    raise NotImplementedError("待实现")


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """
    保存 matplotlib 图表到磁盘。

    自动创建父目录（若不存在）。

    Args:
        fig: 待保存的 matplotlib Figure 对象。
        path: 保存路径（含扩展名，如 "output/fig.png"）。
              支持 PNG、PDF、SVG 等 matplotlib 支持的格式。
        dpi: 输出分辨率，默认 150 DPI。
    """
    raise NotImplementedError("待实现")
