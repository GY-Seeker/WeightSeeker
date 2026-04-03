"""Forward propagation tracking module.

本模块提供前向传播追踪功能，负责执行前向传播并从 HookManager 中提取
注意力矩阵和隐藏状态。
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..core.types import Tensor
from ..model_adapter.hooks import HookManager


class ForwardTracker:
    """前向传播追踪器。

    负责执行前向传播并从 HookManager 中提取注意力矩阵和隐藏状态。
    在 track 方法中会清空 hook_manager 的存储，执行前向传播，
    然后提取并缓存注意力矩阵和隐藏状态。

    Attributes:
        _hook_manager: HookManager 实例，用于捕获中间层输出
        _last_attention_maps: 上次 track 提取的注意力矩阵缓存
        _last_hidden_states: 上次 track 提取的隐藏状态缓存
    """

    def __init__(self, hook_manager: HookManager) -> None:
        """初始化前向追踪器。

        Args:
            hook_manager: HookManager 实例，用于捕获中间层输出
        """
        self._hook_manager = hook_manager
        self._last_attention_maps: Dict[int, Tensor] = {}
        self._last_hidden_states: Dict[int, Tensor] = {}

    def track(self, model: nn.Module, input_data: Tensor) -> Dict[str, Any]:
        """执行前向传播并提取注意力矩阵和隐藏状态。

        流程：
        1. 清空 hook_manager 的存储（调用 clear_storage）
        2. 设置 model 为 eval 模式
        3. 确保 input_data 需要梯度（用于后续反向传播）
        4. 执行前向传播（保留计算图）
        5. 从 hook_manager 提取所有层的注意力输出和隐藏状态
        6. 缓存到内部变量

        Args:
            model: 模型实例
            input_data: 输入数据 (B, C, H, W) 或 (B, L, D)

        Returns:
            Dict: 包含以下键的字典：
                - "attention": Dict[int, Tensor]，各层注意力矩阵
                - "hidden_state": Dict[int, Tensor]，各层隐藏状态
                - "output": Tensor，模型输出
        """
        # 清空 hook_manager 的存储
        self._hook_manager.clear_storage()

        # 设置 model 为 train 模式以保留所有梯度（用于分析）
        # 注意：虽然 eval 模式也能前向传播，但某些层（如 Dropout）在 eval 模式下行为不同
        # 且为了确保参数梯度能被正确计算，使用 train 模式
        model.train()

        # 确保 input_data 需要梯度（用于后续反向传播）
        if not input_data.requires_grad:
            input_data.requires_grad_(True)

        # 执行前向传播（保留计算图，不使用 no_grad）
        output = model(input_data)

        # 从 hook_manager 提取所有层的注意力输出和隐藏状态
        attention_maps: Dict[int, Tensor] = {}
        hidden_states: Dict[int, Tensor] = {}

        # 遍历所有已捕获的层
        for layer_idx in self._hook_manager._hook_storage["attention"].keys():
            try:
                attention_maps[layer_idx] = self._hook_manager.get_attention_output(layer_idx)
            except KeyError:
                pass

        for layer_idx in self._hook_manager._hook_storage["hidden_state"].keys():
            try:
                hidden_states[layer_idx] = self._hook_manager.get_hidden_state(layer_idx)
            except KeyError:
                pass

        # 缓存到内部变量
        self._last_attention_maps = attention_maps
        self._last_hidden_states = hidden_states

        return {
            "attention": attention_maps,
            "hidden_state": hidden_states,
            "output": output,
        }

    def extract_attention_matrices(self) -> Dict[int, Tensor]:
        """提取所有层的注意力矩阵。

        返回上次 track 调用时提取的注意力矩阵。

        Returns:
            Dict[int, Tensor]: 层索引到注意力矩阵的映射
        """
        return self._last_attention_maps.copy()

    def extract_hidden_states(self) -> Dict[int, Tensor]:
        """提取所有层的隐藏状态。

        返回上次 track 调用时提取的隐藏状态。

        Returns:
            Dict[int, Tensor]: 层索引到隐藏状态的映射
        """
        return self._last_hidden_states.copy()
