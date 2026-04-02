"""Core module for transformer analyzer.

该包提供整个项目的基础类型、配置和异常定义，其余模块均依赖于此。
"""

from .config import Config
from .types import (
    AnalysisConfig,
    AttentionMap,
    DiagnosisReport,
    FusionStrategyType,
    GradientMap,
    Heatmap,
    HookType,
    ModelArchitecture,
    ModelInfo,
    NDArray,
    Quadrant,
    Tensor,
    AccumulatorState,
    HeadClassification,
)
from .exceptions import (
    AccumulatorOverflowError,
    AnalyzerException,
    ArchitectureNotSupportedError,
    FusionError,
    HookRegistrationError,
    InvalidInputError,
)

__all__ = [
    "Config",
    # 类型与枚举
    "ModelArchitecture",
    "Quadrant",
    "FusionStrategyType",
    "HookType",
    "ModelInfo",
    "AccumulatorState",
    "HeadClassification",
    "DiagnosisReport",
    "AnalysisConfig",
    "Tensor",
    "NDArray",
    "AttentionMap",
    "GradientMap",
    "Heatmap",
    # 异常类
    "AnalyzerException",
    "ArchitectureNotSupportedError",
    "HookRegistrationError",
    "AccumulatorOverflowError",
    "InvalidInputError",
    "FusionError",
]
