"""
仪表板整合模块
"""

from typing import Dict, Any, Optional
import numpy as np
from numpy import ndarray as NDArray
from matplotlib.figure import Figure

from .heatmap import HeatmapGenerator
from .charts import ChartGenerator
from .exporters import ResultExporter


class Dashboard:
    """可视化仪表板"""
    
    def __init__(self, 
                 heatmap_generator: HeatmapGenerator,
                 chart_generator: ChartGenerator,
                 exporter: ResultExporter) -> None:
        """初始化仪表板"""
        pass
    
    def create_single_sample_panel(self,
                                   attention_heatmap: NDArray,
                                   gradient_heatmap: NDArray,
                                   fusion_heatmap: NDArray,
                                   quadrant_heatmap: NDArray) -> Figure:
        """
        创建单样本空间热力图面板（面板A）
        
        Args:
            attention_heatmap: 注意力热力图 (H, W, 3) RGB格式，值域[0, 255]
            gradient_heatmap: 梯度热力图 (H, W, 3) RGB格式，值域[0, 255]
            fusion_heatmap: 融合重要性热力图 (H, W, 3) RGB格式，值域[0, 255]
            quadrant_heatmap: 四象限分类图 (H, W, 3) RGB格式，用不同颜色表示四个象限
        
        Returns:
            Figure: 包含四张子图的Figure（2x2布局）
                - 左上: 注意力热力图
                - 右上: 梯度热力图
                - 左下: 融合重要性热力图
                - 右下: 四象限解释图
        """
        pass
    
    def create_global_statistics_panel(self,
                                       layer_ranking_fig: Figure,
                                       scatter_fig: Figure,
                                       activity_fig: Figure,
                                       moe_pie_fig: Optional[Figure] = None) -> Figure:
        """
        创建全局权重统计图面板（面板B）
        
        Args:
            layer_ranking_fig: 层重要性排名柱状图（matplotlib Figure对象）
                - X轴: 层索引
                - Y轴: 重要性得分（梯度范数）
            scatter_fig: 头"频率-重要性"散点图（matplotlib Figure对象）
                - X轴: 激活频率 [0, 1]
                - Y轴: 重要性得分（梯度范数）
                - 每个点代表一个注意力头，用颜色区分类别
            activity_fig: 头活跃度热力矩阵（matplotlib Figure对象）
                - X轴: 样本索引
                - Y轴: 头编号（展平为 layer_idx * num_heads + head_idx）
                - 颜色深浅表示激活强度
            moe_pie_fig: MoE专家路由频率饼图（matplotlib Figure对象，可选）
                - 各扇区大小对应专家被选中频率
                - 仅MoE架构时提供
        
        Returns:
            Figure: 包含多个统计图表的综合Figure（网格布局）
        """
        pass
    
    def generate_full_report(self,
                            single_sample_results: Dict[str, Any],
                            global_diagnosis_results: Dict[str, Any],
                            output_dir: str) -> str:
        """
        生成完整分析报告
        
        Args:
            single_sample_results: 单样本分析结果
            global_diagnosis_results: 全局诊断结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        pass
