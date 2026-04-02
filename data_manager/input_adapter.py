"""
多输入模型适配器模块

根据 design.md §3.7.4，InputAdapter 负责将多输入模型包装为标准单输入接口 model(x)，
使分析流水线无需感知模型签名差异。

背景：在测试 MultiModalECGTransformer 时发现，该模型 forward 函数接受两个输入
(ecg_signal, meta_data)，而分析流水线默认假设单输入接口 model(x)。
InputAdapter 通过绑定辅助输入或字典展开，将多输入统一为单输入调用。

设计亮点：
- from_signature() 工厂方法可自动从 forward 签名推断适配策略
- BIND_AUXILIARY 模式会自动同步辅助张量到主输入所在设备
- to_device() 同时迁移模型本体和所有辅助输入张量
"""

import inspect
import logging
import warnings
from enum import Enum, auto
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaptStrategy(Enum):
    """
    多输入适配策略枚举。

    用于 InputAdapter 决定如何将多输入模型包装为单输入接口：
    - PASSTHROUGH：模型本身只有单输入，不做任何包装
    - BIND_AUXILIARY：将固定辅助输入绑定到模型，仅主输入动态传入
    - DICT_EXPAND：输入为字典，展开为 **kwargs 传入模型
    """
    PASSTHROUGH = auto()
    """单输入直通：模型 forward 仅有一个必选参数，不需要适配。"""

    BIND_AUXILIARY = auto()
    """辅助输入绑定：固定辅助输入为预设张量，主输入动态传入。
    适用场景：ECG 模型 forward(ecg_signal, meta_data)，meta_data 固定为占位张量。"""

    DICT_EXPAND = auto()
    """字典展开：输入为 dict，将各 key 展开为 **kwargs 调用。
    适用场景：NLP 模型接收 {"input_ids": ..., "attention_mask": ...} 格式输入。"""


class InputAdapter(nn.Module):
    """
    多输入模型适配器：将多输入模型包装为标准单输入接口 model(x)。

    适配策略说明：
    - PASSTHROUGH：等价于直接调用 model(x)，无任何包装开销。
    - BIND_AUXILIARY：调用时自动将 auxiliary_inputs 中的所有张量迁移到
      与 x 相同的设备，然后以关键字参数方式传入模型。
    - DICT_EXPAND：x 必须为字典，调用时等价于 model(**x)。

    常见使用场景：
    - ECG 多模态模型（forward(ecg_signal, meta_data) 双输入）
    - 带 attention_mask 的 NLP 模型
    - 任何 forward 签名包含多个必选参数的模型

    注意：
    - Hook 注册应作用于 get_wrapped_model() 返回的原始模型，而非 InputAdapter 本身。
    - DataParallel 包装应在 InputAdapter 外层处理。

    示例::

        # ECG 双输入模型
        model = MultiModalECGTransformer(...)
        meta_placeholder = torch.zeros(1, 16)
        adapter = InputAdapter.from_signature(
            model, auxiliary_inputs={"meta_data": meta_placeholder}
        )
        # 之后 adapter 表现为单输入模型
        output = adapter(ecg_signal_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: AdaptStrategy = AdaptStrategy.PASSTHROUGH,
        auxiliary_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        初始化适配器。

        Args:
            model: 被包装的原始模型（任意 nn.Module）。
            strategy: 适配策略枚举（AdaptStrategy），默认 PASSTHROUGH。
            auxiliary_inputs: BIND_AUXILIARY 策略下的辅助输入字典。
                              字典的 key 对应 model.forward() 的参数名，
                              value 为固定的辅助张量（会随主输入自动迁移设备）。
                              例如：{"meta_data": torch.zeros(1, 16)}。
                              其他策略下此参数被忽略。
        """
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.auxiliary_inputs: Dict[str, Any] = auxiliary_inputs or {}

    def forward(self, x: Any) -> Any:
        """
        统一单输入调用接口。

        根据 strategy 分发到对应的内部方法：
        - PASSTHROUGH → model(x)
        - BIND_AUXILIARY → _forward_bind_auxiliary(x)
        - DICT_EXPAND → _forward_dict_expand(x)

        Args:
            x: 主输入张量（PASSTHROUGH / BIND_AUXILIARY 模式），
               或包含所有输入键值对的字典（DICT_EXPAND 模式）。

        Returns:
            Any: 原始模型的输出（与直接调用 model 一致）。

        Raises:
            ValueError: strategy 为未知值时抛出。
            TypeError: DICT_EXPAND 模式下 x 不是 dict 时抛出。
        """
        if self.strategy == AdaptStrategy.PASSTHROUGH:
            return self.model(x)
        elif self.strategy == AdaptStrategy.BIND_AUXILIARY:
            return self._forward_bind_auxiliary(x)
        elif self.strategy == AdaptStrategy.DICT_EXPAND:
            return self._forward_dict_expand(x)
        else:
            raise ValueError(f"未知的适配策略：{self.strategy}")

    def _forward_bind_auxiliary(self, x: Any) -> Any:
        """
        辅助输入绑定模式的前向传播。

        将 auxiliary_inputs 中所有张量（Tensor 类型）迁移到与 x 相同的设备，
        然后将 x 作为第一个位置参数，auxiliary_inputs 中的键值对作为关键字参数，
        共同传入 model.forward()。

        设备同步逻辑（仅对 torch.Tensor 类型的辅助输入生效）：
            if isinstance(v, Tensor): v = v.to(x.device)

        调用等价于：
            model(x, meta_data=self.auxiliary_inputs["meta_data"].to(x.device), ...)

        Args:
            x: 主输入张量（动态传入，每次调用可不同）。

        Returns:
            Any: 模型输出。
        """
        raise NotImplementedError("待实现")

    def _forward_dict_expand(self, x: Dict[str, Any]) -> Any:
        """
        字典展开模式的前向传播。

        将输入字典 x 展开为 **kwargs 传入模型：
            model(**x)

        Args:
            x: 包含模型所有必要输入的字典，key 对应 forward() 参数名。
               例如：{"input_ids": ..., "attention_mask": ...}

        Returns:
            Any: 模型输出。

        Raises:
            TypeError: x 不是 dict 时抛出，错误消息说明实际类型。
        """
        raise NotImplementedError("待实现")

    @classmethod
    def from_signature(
        cls,
        model: nn.Module,
        auxiliary_inputs: Optional[Dict[str, Any]] = None,
    ) -> "InputAdapter":
        """
        工厂方法：从模型 forward 签名自动推断适配策略并创建实例。

        推断规则（按优先级）：
        1. 若 forward 只有 1 个必选参数（除 self 外）→ PASSTHROUGH
        2. 若 forward 有 2+ 个必选参数，且传入了 auxiliary_inputs → BIND_AUXILIARY
        3. 若 forward 有 2+ 个必选参数，但未传入 auxiliary_inputs →
           记录警告（"模型有多个必选参数但未提供 auxiliary_inputs，回退到 PASSTHROUGH"），
           默认 PASSTHROUGH

        参数计数规则：仅统计无默认值的必选参数（排除 *args、**kwargs）。

        Args:
            model: 被包装的模型实例（支持 DataParallel 包装，内部会先 unwrap）。
            auxiliary_inputs: 辅助输入字典（可选）。
                              有多个必选参数且提供此参数时，触发 BIND_AUXILIARY 策略。

        Returns:
            InputAdapter: 根据签名自动配置的适配器实例。

        Note:
            推断结果会通过 logging.info() 输出选中的策略，
            低置信度情况（多参数无辅助输入）会通过 warnings.warn() 打印警告。
        """
        raise NotImplementedError("待实现")

    def get_wrapped_model(self) -> nn.Module:
        """
        获取被包装的原始模型。

        用于需要直接访问原始模型的场景，例如：
        - HookManager 注册 Hook（Hook 应注册在原始模型上）
        - ArchitectureDetector 探测架构（探测原始模型结构）
        - 手动访问模型权重

        Returns:
            nn.Module: 被包装的原始模型实例（未经 InputAdapter 包装）。
        """
        return self.model

    def to_device(self, device: Union[str, torch.device]) -> "InputAdapter":
        """
        将适配器（含模型本体和所有辅助输入张量）迁移到指定设备。

        迁移逻辑：
        1. 调用 self.model.to(device) 迁移模型参数。
        2. 遍历 auxiliary_inputs，将所有 torch.Tensor 类型的值 .to(device)。

        Args:
            device: 目标设备，可为字符串（如 "cuda:0"、"cpu"）
                    或 torch.device 对象。

        Returns:
            InputAdapter: self（支持链式调用）。
        """
        raise NotImplementedError("待实现")
