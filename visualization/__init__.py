"""
可视化模块（简化版）

根据 design.md §3.6，当前仅提供以下核心可视化功能：
- HeatmapRenderer : 热力图渲染器（注意力 / 梯度 / 多层面板 / 信号叠加）
- 通用绘图工具函数（plot_layer_importance / plot_head_scatter /
                    plot_accumulator_stats / save_figure）

[未来扩展] DashboardGenerator 和 ReportExporter 留存于各自文件中，
当前不导出，待功能实现后再添加至 __all__。
"""

from .heatmap import HeatmapRenderer
from .timeseries_visualizer import TimeSeriesVisualizer
from .charts import (
    plot_layer_importance,
    plot_head_scatter,
    plot_accumulator_stats,
    save_figure,
)

__all__ = [
    # 核心渲染器
    "HeatmapRenderer",
    "TimeSeriesVisualizer",
    # 通用绘图工具函数
    "plot_layer_importance",
    "plot_head_scatter",
    "plot_accumulator_stats",
    "save_figure",
]
