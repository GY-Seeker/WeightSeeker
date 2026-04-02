"""Backward propagation tracking module.

本模块提供反向传播追踪功能，负责执行反向传播并计算各类梯度，
包括输入梯度、隐藏状态梯度和注意力梯度。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.types import Tensor
from ..model_adapter.hooks import HookManager
from .metrics import MetricsCalculator


class BackwardTracker:
    """反向传播追踪器。

    负责执行反向传播并计算各类梯度，包括输入梯度、隐藏状态梯度和注意力梯度。
    梯度计算采用 L2 范数，对 batch 维度取平均。

    Attributes:
        _model: 模型实例
        _hook_manager: HookManager 实例，用于获取前向阶段的中间层信息
        _gradients_cache: 梯度缓存字典
    """

    def __init__(self, model: nn.Module, hook_manager: HookManager) -> None:
        """初始化反向追踪器。

        Args:
            model: 模型实例
            hook_manager: Hook管理器，用于获取前向阶段的中间层信息
        """
        self._model = model
        self._hook_manager = hook_manager
        self._gradients_cache: Dict[str, Any] = {}

    def track(
        self, loss: Tensor, input_data: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """执行反向传播并计算梯度。

        流程：
        1. 执行反向传播：loss.backward(retain_graph=True)
        2. 收集各类梯度：
           - 输入梯度：如果 input_data 有 grad，提取 input_data.grad
           - 隐藏状态梯度：从 hook_manager 获取各层隐藏状态，计算其梯度
           - 注意力梯度：从 hook_manager 获取各层注意力，计算其梯度
        3. 梯度计算方式：平方和开根号（L2范数），对batch维度取平均

        Args:
            loss: 损失张量
            input_data: 可选的输入数据张量，用于提取输入梯度

        Returns:
            Dict: 包含以下键的字典：
                - "input": Tensor，输入梯度（L2范数，batch平均）
                - "hidden": Dict[int, Tensor]，各层隐藏状态梯度
                - "attention": Dict[int, Tensor]，各层注意力梯度
        """
        # 清除之前的梯度
        self._model.zero_grad()
        if input_data is not None and input_data.grad is not None:
            input_data.grad.zero_()

        # 执行反向传播（保留计算图）
        loss.backward(retain_graph=True)

        # 收集各类梯度
        result: Dict[str, Any] = {
            "input": None,
            "hidden": {},
            "attention": {},
        }

        # 输入梯度
        if input_data is not None:
            result["input"] = self.compute_input_gradient(input_data)

        # 隐藏状态梯度
        for layer_idx in self._hook_manager._hook_storage["hidden_state"].keys():
            hidden_grad = self.compute_hidden_gradient(layer_idx)
            if hidden_grad is not None:
                result["hidden"][layer_idx] = hidden_grad

        # 注意力梯度
        for layer_idx in self._hook_manager._hook_storage["attention"].keys():
            attn_grad = self.compute_attention_gradient(layer_idx)
            if attn_grad is not None:
                result["attention"][layer_idx] = attn_grad

        self._gradients_cache = result
        return result

    def compute_input_gradient(self, input_data: Tensor) -> Tensor:
        """计算输入数据的梯度。

        如果 input_data.grad 存在，返回其 L2 范数（对 batch 维平均）。
        否则返回零张量。

        Args:
            input_data: 输入数据张量

        Returns:
            Tensor: 输入梯度的 L2 范数（标量或按batch平均后的值）
        """
        if input_data.grad is None:
            # 返回零张量
            return torch.tensor(0.0, device=input_data.device, dtype=input_data.dtype)

        # 计算 L2 范数
        grad_norm = MetricsCalculator.compute_l2_norm(input_data.grad)
        # 对 batch 维度取平均
        if grad_norm.dim() > 0:
            grad_norm = grad_norm.mean()
        return grad_norm

    def compute_hidden_gradient(self, layer_idx: int) -> Optional[Tensor]:
        """计算隐藏状态梯度。

        从 hook_manager 获取该层的隐藏状态，如果隐藏状态有 grad，
        返回梯度的 L2 范数（对 batch 维平均）。

        Args:
            layer_idx: 层索引

        Returns:
            Tensor: 梯度的 L2 范数（batch 平均后），如果无梯度则返回 None
        """
        try:
            hidden_state = self._hook_manager.get_hidden_state(layer_idx)
        except KeyError:
            return None

        if hidden_state.grad is None:
            return None

        # 计算 L2 范数
        grad_norm = MetricsCalculator.compute_l2_norm(hidden_state.grad)
        # 对 batch 维度取平均（如果有多维）
        if grad_norm.dim() > 0:
            grad_norm = grad_norm.mean()
        return grad_norm

    def compute_attention_gradient(self, layer_idx: int) -> Optional[Tensor]:
        """计算注意力梯度。

        从 hook_manager 获取该层的注意力输出，如果注意力张量有 grad，
        返回梯度的 L2 范数（对 batch 维平均）。

        Args:
            layer_idx: 层索引

        Returns:
            Tensor: 梯度的 L2 范数（batch 平均后），如果无梯度则返回 None
        """
        try:
            attention = self._hook_manager.get_attention_output(layer_idx)
        except KeyError:
            return None

        if attention.grad is None:
            return None

        # 计算 L2 范数
        grad_norm = MetricsCalculator.compute_l2_norm(attention.grad)
        # 对 batch 维度取平均（如果有多维）
        if grad_norm.dim() > 0:
            grad_norm = grad_norm.mean()
        return grad_norm

    def aggregate_to_patch_level(self, gradients: Tensor, patch_size: int) -> Tensor:
        """将像素级梯度聚合为 Patch 级向量。

        使用 unfold 操作将 (B, C, H, W) 的梯度张量分割为 patches，
        然后对每个 patch 内的梯度取平均，得到 (B, num_patches_h, num_patches_w) 的结果。

        Args:
            gradients: 像素级梯度张量，形状 (B, C, H, W)
            patch_size: Patch 大小

        Returns:
            Tensor: Patch 级梯度向量，形状 (B, num_patches_h, num_patches_w)
        """
        if gradients.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {gradients.dim()}D")

        B, C, H, W = gradients.shape

        # 确保 H, W 能被 patch_size 整除
        if H % patch_size != 0 or W % patch_size != 0:
            # 裁剪到可整除的大小
            H = (H // patch_size) * patch_size
            W = (W // patch_size) * patch_size
            gradients = gradients[:, :, :H, :W]

        # 使用 unfold 提取 patches: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = gradients.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches 形状: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)

        # 对每个 patch 内的所有值取平均
        # 先对最后两个维度（patch_size, patch_size）和通道维度取平均
        patch_gradients = patches.mean(dim=(1, 4, 5))  # (B, num_patches_h, num_patches_w)

        return patch_gradients
