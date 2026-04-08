"""
Transformer权重监控与分析系统

一款专为深度解析Transformer类模型内部决策机制而设计的可解释性分析与可视化工具，
支持标准Transformer、MoE-Transformer、Swin Transformer以及ViT四种架构。
"""

# 核心模块
from .core.config import Config
from .core.types import (
    ModelArchitecture,
    Quadrant,
    FusionStrategyType,
    HookType,
    AdaptStrategy,
    DetectionResult,
    ModelInfo,
    AccumulatorState,
    HeadClassification,
    DiagnosisReport,
    AnalysisConfig,
)
from .core.exceptions import (
    AnalyzerException,
    ArchitectureNotSupportedError,
    HookRegistrationError,
    AccumulatorOverflowError,
    InvalidInputError,
    FusionError,
)

# 模型适配模块
from .model_adapter.detector import ArchitectureDetector
from .model_adapter.hooks import HookManager, AttentionHook

# 追踪模块
from .tracker.forward_tracker import ForwardTracker
from .tracker.backward_tracker import BackwardTracker
from .tracker.accumulator import CrossSampleAccumulator
from .tracker.metrics import MetricsCalculator

# 空间处理模块
from .spatial.reshaper import SpatialReshaper
from .spatial.normalizer import Normalizer
from .spatial.interpolator import Interpolator

# 分析模块
from .analyzer.single_sample import SingleSampleAnalyzer
from .analyzer.quadrant import QuadrantAnalyzer
from .analyzer.global_diagnosis import GlobalDiagnosisEngine
from .analyzer.anomaly_detector import AnomalyDetector

# 可视化模块
from .visualization.heatmap import HeatmapRenderer
from .visualization.charts import (
    plot_layer_importance,
    plot_head_scatter,
    plot_accumulator_stats,
    save_figure,
)

# 主入口
from .pipeline import AnalysisPipeline

__version__ = "1.2.0"
__author__ = "Transformer Analyzer Team"

__all__ = [
    # 核心
    "Config",
    "ModelArchitecture",
    "Quadrant",
    "FusionStrategyType",
    "HookType",
    "AdaptStrategy",
    "DetectionResult",
    "ModelInfo",
    "AccumulatorState",
    "HeadClassification",
    "DiagnosisReport",
    "AnalysisConfig",
    # 异常
    "AnalyzerException",
    "ArchitectureNotSupportedError",
    "HookRegistrationError",
    "AccumulatorOverflowError",
    "InvalidInputError",
    "FusionError",
    # 模型适配
    "ArchitectureDetector",
    "HookManager",
    "AttentionHook",
    # 追踪
    "ForwardTracker",
    "BackwardTracker",
    "CrossSampleAccumulator",
    "MetricsCalculator",
    # 空间处理
    "SpatialReshaper",
    "Normalizer",
    "Interpolator",
    # 分析
    "SingleSampleAnalyzer",
    "QuadrantAnalyzer",
    "GlobalDiagnosisEngine",
    "AnomalyDetector",
    # 可视化
    "HeatmapRenderer",
    "plot_layer_importance",
    "plot_head_scatter",
    "plot_accumulator_stats",
    "save_figure",
    # 主入口
    "AnalysisPipeline",
]
