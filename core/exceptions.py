"""Custom exception classes for transformer analyzer.

本模块定义了分析流程中使用的自定义异常类型，用于在各模块之间
传递清晰且结构化的错误信息，便于上层统一处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class AnalyzerException(Exception):
    """分析系统基础异常类。

    所有自定义异常均应继承自该类，便于调用方进行统一捕获。
    """

    def __init__(self, message: str = "Analyzer error", *args: Any) -> None:
        """初始化基础异常。

        Args:
            message: 错误消息描述。
            *args: 额外的位置参数，会传递给基类 :class:`Exception`。
        """
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - 简单委托
        """返回可读的错误描述字符串。"""
        return str(self.message)


class ArchitectureNotSupportedError(AnalyzerException):
    """不支持的模型架构异常。

    当 :class:`model_adapter.detector.ArchitectureDetector` 无法识别
    给定模型或检测到的架构不在支持列表中时抛出。
    """

    def __init__(self, architecture: Optional[str] = None, message: Optional[str] = None) -> None:
        """初始化异常。

        Args:
            architecture: 检测到但不被支持的架构名称。
            message: 自定义错误消息；若未提供，将根据 ``architecture`` 自动生成。
        """
        if message is None:
            if architecture is None:
                message = "Model architecture is not supported."
            else:
                message = f"Model architecture '{architecture}' is not supported."
        super().__init__(message)
        self.architecture = architecture


class HookRegistrationError(AnalyzerException):
    """Hook 注册失败异常。

    在 :mod:`model_adapter.hooks` 中注册前向/反向 Hook 失败时抛出，
    通常用于指示模块名称不匹配或模型结构与预期不一致等问题。
    """

    def __init__(self, hook_type: Optional[str] = None, message: Optional[str] = None) -> None:
        """初始化异常。

        Args:
            hook_type: 发生错误的 Hook 类型名称，例如 "attention"、"hidden_state"。
            message: 自定义错误消息；若未提供，将根据 ``hook_type`` 自动生成。
        """
        if message is None:
            if hook_type is None:
                message = "Failed to register hook."
            else:
                message = f"Failed to register hook of type '{hook_type}'."
        super().__init__(message)
        self.hook_type = hook_type


class AccumulatorOverflowError(AnalyzerException):
    """累积器溢出异常。

    当跨样本累积器中的样本数量超过预设上限
    (:attr:`core.config.Config.ACCUMULATOR_LIMIT`) 时抛出。
    """

    def __init__(self, current_count: int, limit: int, message: Optional[str] = None) -> None:
        """初始化异常。

        Args:
            current_count: 当前已累积的样本数。
            limit: 样本上限值。
            message: 自定义错误消息；若未提供，将根据 ``current_count`` 和 ``limit`` 自动生成。
        """
        if message is None:
            message = (
                f"Accumulator overflow: current sample count {current_count} exceeds limit {limit}."
            )
        super().__init__(message)
        self.current_count = current_count
        self.limit = limit


class InvalidInputError(AnalyzerException):
    """非法输入异常。

    在数据加载、配置解析或前向/反向计算中发现输入格式与预期不符时抛出。
    例如张量形状不合法、配置字段缺失/类型错误等。
    """

    def __init__(
        self,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """初始化异常。

        Args:
            expected: 期望的输入格式描述，例如 "Tensor of shape (B, C, H, W)"。
            actual: 实际收到的输入格式描述，例如 "Tensor of shape (B, H, W)"。
            message: 自定义错误消息；若未提供，将根据 ``expected`` 与 ``actual`` 自动生成。
        """
        if message is None:
            if expected is None and actual is None:
                message = "Invalid input detected."
            else:
                message = f"Invalid input. Expected: {expected}, Actual: {actual}."
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class FusionError(AnalyzerException):
    """融合计算异常。

    在执行像素级信息融合（见 :mod:`fusion.strategies` 和
    :class:`fusion.composer.FusionComposer`）过程中，如果输入不匹配、
    数值范围异常或内部实现错误，均可能抛出该异常。
    """

    def __init__(self, message: str = "Fusion computation failed.") -> None:
        """初始化异常。

        Args:
            message: 错误消息描述。
        """
        super().__init__(message)
