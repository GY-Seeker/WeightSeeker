"""Type definitions and enums for transformer analyzer.

本模块定义了全项目共用的类型别名、枚举类和数据类，
为模型架构识别、注意力分析、梯度追踪等功能提供统一的类型基础。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# =============================================================================
# 类型别名
# =============================================================================

Tensor = torch.Tensor
"""PyTorch张量类型别名"""

NDArray = np.ndarray
"""NumPy数组类型别名"""

AttentionMap = Tensor
"""注意力矩阵类型，形状通常为 (B, H, N, N) 或窗口格式"""

GradientMap = Tensor
"""梯度矩阵类型"""

Heatmap = NDArray
"""热力图类型，形状为 (H, W) 的numpy数组"""


# =============================================================================
# 枚举类定义
# =============================================================================

class ModelArchitecture(Enum):
    """支持的模型架构枚举。
    
    本软件严格限定支持四种主流目标架构：
    - TRANSFORMER: 标准Transformer架构
    - MOE_TRANSFORMER: 混合专家(MoE)Transformer架构
    - SWIN: Swin Transformer架构（基于窗口的自注意力）
    - VIT: Vision Transformer架构
    """
    
    TRANSFORMER = auto()
    MOE_TRANSFORMER = auto()
    SWIN = auto()
    VIT = auto()


class Quadrant(Enum):
    """四象限枚举，用于单样本解释的区域分类。
    
    根据注意力值和梯度值的阈值划分四个象限：
    - CORE_DISCRIMINATIVE: 核心判别区（注意力高、梯度高）
    - REDUNDANT_ATTENTION: 冗余关注区（注意力高、梯度低）
    - POTENTIAL_INFLUENCE: 潜在影响区（注意力低、梯度高）
    - IRRELEVANT: 无关区域（注意力低、梯度低）
    """
    
    CORE_DISCRIMINATIVE = auto()  # 核心判别区
    REDUNDANT_ATTENTION = auto()  # 冗余关注区
    POTENTIAL_INFLUENCE = auto()  # 潜在影响区
    IRRELEVANT = auto()  # 无关区域


class FusionStrategyType(Enum):
    """融合策略类型枚举。
    
    注意：命名为 FusionStrategyType 以避免与 fusion/strategies.py 
    中的 FusionStrategy 抽象基类重名。
    
    支持的融合策略：
    - PRODUCT: 乘积融合，重要性 = 注意力 × 梯度
    - WEIGHTED_SUM: 加权求和，重要性 = α×注意力 + (1-α)×梯度
    - ATTENTION_MASK: 注意力掩码筛选，先对注意力做阈值掩码，再与梯度相乘
    """
    
    PRODUCT = auto()  # 乘积融合
    WEIGHTED_SUM = auto()  # 加权求和
    ATTENTION_MASK = auto()  # 注意力掩码筛选


class HookType(Enum):
    """Hook类型枚举，用于标识注册的钩子类型。
    
    - ATTENTION: 注意力钩子，捕获注意力矩阵
    - MOE_ROUTER: MoE路由钩子，捕获专家分配
    - HIDDEN_STATE: 隐藏状态钩子，捕获中间层输出
    """
    
    ATTENTION = auto()
    MOE_ROUTER = auto()
    HIDDEN_STATE = auto()


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class ModelInfo:
    """模型信息数据类，存储架构探测结果。
    
    Attributes:
        architecture: 模型架构类型
        num_layers: 模型层数
        num_heads: 每层注意力头数
        patch_size: Patch大小（用于ViT/Swin等）
        hidden_dim: 隐藏层维度
        window_size: 窗口大小（Swin特有，可选）
        num_experts: 专家数量（MoE特有，可选）
    """
    
    architecture: ModelArchitecture = ModelArchitecture.VIT
    num_layers: int = 12
    num_heads: int = 12
    patch_size: int = 16
    hidden_dim: int = 768
    window_size: Optional[int] = None  # Swin特有
    num_experts: Optional[int] = None  # MoE特有


@dataclass
class AccumulatorState:
    """累积器状态数据类，存储跨样本统计信息。
    
    Attributes:
        head_activation_freq: 头的激活频率，形状 (num_layers, num_heads)
        head_attention_concentration: 头的注意力集中度，形状 (num_layers, num_heads)
        layer_gradient_norm: 层的梯度L2范数，形状 (num_layers,)
        attention_gradient_norm: 注意力梯度范数，形状 (num_layers, num_heads)，可选
        expert_selection_count: 专家选中计数，形状 (num_experts,)，可选
        sample_count: 已处理的样本数
    """
    
    head_activation_freq: Tensor = field(default_factory=lambda: torch.empty(0))
    head_attention_concentration: Tensor = field(default_factory=lambda: torch.empty(0))
    layer_gradient_norm: Tensor = field(default_factory=lambda: torch.empty(0))
    attention_gradient_norm: Optional[Tensor] = None  # (num_layers, num_heads) 注意力梯度范数
    expert_selection_count: Optional[Tensor] = None  # (num_experts,)
    sample_count: int = 0


@dataclass
class HeadClassification:
    """头分类结果数据类，描述单个注意力头的分类信息。
    
    Attributes:
        layer_idx: 层索引
        head_idx: 头索引
        activation_freq: 激活频率值
        concentration: 注意力集中度值
        importance_score: 重要性得分（基于梯度）
        category: 分类类别，可选值：
            - "high_freq_high_focus": 高频高聚焦头
            - "high_freq_low_focus": 高频低聚焦头（均匀头）
            - "low_freq": 低频头
    """
    
    layer_idx: int = 0
    head_idx: int = 0
    activation_freq: float = 0.0
    concentration: float = 0.0
    importance_score: float = 0.0
    category: str = "low_freq"  # "high_freq_high_focus" | "high_freq_low_focus" | "low_freq"


@dataclass
class DiagnosisReport:
    """全局诊断报告数据类，汇总全局权重诊断结果。
    
    Attributes:
        activation_frequency_ranking: 基于激活频率和集中度的头排名列表
        gradient_importance_ranking: 基于梯度范数的重要性排名
            包含 "layer_ranking" 和 "head_ranking" 两个字典
        anomaly_analysis: 异常模式识别结果，包括：
            - "redundant_heads": 高频低效头列表
            - "sparse_critical_heads": 低频高效头列表
            - "moe_load_imbalance": MoE负载偏斜分析（MoE架构）
        head_classification: 头分类结果，按类别组织
    """
    
    activation_frequency_ranking: List[Dict[str, Any]] = field(default_factory=list)
    gradient_importance_ranking: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    anomaly_analysis: Dict[str, Any] = field(default_factory=dict)
    head_classification: Dict[str, List[HeadClassification]] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """分析配置数据类，存储分析流程的配置参数。
    
    Attributes:
        model_path: 模型文件路径或模型名称
        data_path: 数据文件路径或数据目录
        output_dir: 输出目录，默认为"./results"
        device: 计算设备，可选 "auto", "cpu", "cuda" 等
        precision: 计算精度，可选 "fp32" 或 "fp16"
        batch_size: 批处理大小
        max_samples: 最大处理样本数，None表示无限制
    """
    
    model_path: str = ""
    data_path: str = ""
    output_dir: str = "./results"
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 16
    max_samples: Optional[int] = None
