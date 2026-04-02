"""Cross-sample accumulator module.

本模块提供跨样本持久化累积功能，维护与模型结构平行的统计缓冲区，
用于累积多个样本的统计信息，支持头激活频率、注意力集中度、梯度范数等指标。
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ..core.types import Tensor, ModelInfo, AccumulatorState, ModelArchitecture
from ..core.exceptions import AccumulatorOverflowError
from .metrics import MetricsCalculator


class CrossSampleAccumulator:
    """跨样本持久化累积器。

    维护与模型结构平行的统计缓冲区，累积多个样本的统计信息：
    - 每个注意力头的"激活频率"
    - 每个注意力头的"平均注意力集中度"
    - 每个隐藏状态层的"梯度L2范数"
    - 每个注意力头的"梯度L2范数"
    - 每个专家的"被选中计数"（MoE架构）

    Attributes:
        _model_info: 模型信息
        _limit: 样本上限
        _sample_count: 已处理的样本数
        _head_activation_freq: 头激活频率计数，形状 (num_layers, num_heads)
        _head_concentration_sum: 集中度累加，形状 (num_layers, num_heads)
        _layer_gradient_norm_sum: 层梯度范数累加，形状 (num_layers,)
        _attention_gradient_norm_sum: 注意力梯度范数累加，形状 (num_layers, num_heads)
        _expert_selection_count: 专家选中计数，形状 (num_experts,)，MoE架构
    """

    def __init__(self, model_info: ModelInfo, limit: int = 100000, device: Optional[str] = None) -> None:
        """初始化累积器。

        Args:
            model_info: 模型信息，包含层数、头数等
            limit: 样本上限，超过此值将抛出 AccumulatorOverflowError
            device: 设备字符串（如 'cuda:0', 'cpu'），默认从 model_info 推断
        """
        self._model_info = model_info
        self._limit = limit
        self._sample_count = 0
        
        # 确定设备
        if device is None:
            if hasattr(model_info, 'device') and model_info.device:
                self._device = model_info.device
            else:
                self._device = 'cpu'  # 默认使用 CPU
        else:
            self._device = device

        num_layers = model_info.num_layers
        num_heads = model_info.num_heads

        # 初始化统计缓冲区（全部为零张量，移到正确设备）
        # 头激活频率计数 (num_layers, num_heads)
        self._head_activation_freq = torch.zeros(num_layers, num_heads, dtype=torch.float32, device=self._device)

        # 集中度累加 (num_layers, num_heads)
        self._head_concentration_sum = torch.zeros(num_layers, num_heads, dtype=torch.float32, device=self._device)

        # 层梯度范数累加 (num_layers,)
        self._layer_gradient_norm_sum = torch.zeros(num_layers, dtype=torch.float32, device=self._device)

        # 注意力梯度范数累加 (num_layers, num_heads)
        self._attention_gradient_norm_sum = torch.zeros(num_layers, num_heads, dtype=torch.float32, device=self._device)

        # 专家选中计数 (num_experts,)，如果是 MoE 架构
        if model_info.num_experts is not None and model_info.num_experts > 0:
            self._expert_selection_count = torch.zeros(model_info.num_experts, dtype=torch.float32, device=self._device)
        else:
            self._expert_selection_count = None

    def update(
        self,
        attention_maps: Dict[int, Tensor],
        input_gradients: Optional[Tensor] = None,
        hidden_gradients: Optional[Dict[int, Tensor]] = None,
        attention_gradients: Optional[Dict[int, Tensor]] = None,
        expert_assignments: Optional[Tensor] = None,
    ) -> None:
        """更新累积器状态。

        检查是否已满，如满则抛出 AccumulatorOverflowError。
        否则调用各个 _update 方法更新统计信息。

        Args:
            attention_maps: 注意力矩阵字典 {layer_idx: tensor}
            input_gradients: 输入梯度张量 (B, C, H, W) 或 (B, L, D)，可选
            hidden_gradients: 隐藏状态梯度字典 {layer_idx: tensor}，可选
            attention_gradients: 注意力梯度字典 {layer_idx: tensor}，可选
            expert_assignments: 专家分配索引 (MoE)，可选

        Raises:
            AccumulatorOverflowError: 当样本数超过上限时抛出
        """
        # 检查是否已满
        if self._sample_count >= self._limit:
            raise AccumulatorOverflowError(
                current_count=self._sample_count,
                limit=self._limit,
            )

        # 获取 batch size（从 attention_maps 推断）
        batch_size = 1
        if attention_maps:
            first_attention = next(iter(attention_maps.values()))
            batch_size = first_attention.shape[0] if first_attention.dim() > 0 else 1

        # 更新各项统计
        if attention_maps:
            self._update_head_activation_freq(attention_maps)
            self._update_head_concentration(attention_maps)

        if hidden_gradients:
            self._update_layer_gradient_norm(hidden_gradients)

        if attention_gradients:
            self._update_attention_gradient_norm(attention_gradients)

        if expert_assignments is not None and self._expert_selection_count is not None:
            self._update_expert_count(expert_assignments)

        # 递增样本计数（按 batch size 递增）
        self._sample_count += batch_size

    def _update_head_activation_freq(self, attention_maps: Dict[int, Tensor]) -> None:
        """更新头的激活频率。

        对每层每个头，检查该 batch 中是否有非零注意力权重。
        使用 MetricsCalculator.compute_activation_frequency 计算。

        Args:
            attention_maps: 注意力矩阵字典 {layer_idx: (B, H, N, N)}
        """
        # 计算当前 batch 的激活频率
        activation_freq = MetricsCalculator.compute_activation_frequency(attention_maps)
        # 累加到缓冲区
        self._head_activation_freq += activation_freq

    def _update_head_concentration(self, attention_maps: Dict[int, Tensor]) -> None:
        """更新头的注意力集中度。

        对每层每个头计算注意力集中度（1-归一化熵）。
        使用 MetricsCalculator.compute_attention_concentration 计算。
        对 batch 维取平均后累加到 _head_concentration_sum。

        Args:
            attention_maps: 注意力矩阵字典 {layer_idx: (B, H, N, N)}
        """
        for layer_idx, attention in attention_maps.items():
            if layer_idx >= self._model_info.num_layers:
                continue

            # attention 形状: (B, H, N, N) 或 (B, H, N)
            if attention.dim() == 4:
                # (B, H, N, N) -> 对每个头的注意力矩阵计算集中度
                # 对每个 query 位置计算集中度，然后平均
                B, H, N, _ = attention.shape
                # 对最后一个维度（key 维度）计算集中度
                concentration = MetricsCalculator.compute_attention_concentration(attention)
                # concentration 形状: (B, H, N)
                # 对 N 维度取平均
                concentration = concentration.mean(dim=-1)  # (B, H)
            elif attention.dim() == 3:
                # (B, H, N) -> 直接计算集中度
                concentration = MetricsCalculator.compute_attention_concentration(attention)
                # concentration 形状: (B, H)
            else:
                continue

            # 对 batch 维度取平均
            concentration_mean = concentration.mean(dim=0)  # (H,)

            # 累加到缓冲区
            num_heads = min(H, self._model_info.num_heads)
            self._head_concentration_sum[layer_idx, :num_heads] += concentration_mean[:num_heads]

    def _update_layer_gradient_norm(self, hidden_gradients: Dict[int, Tensor]) -> None:
        """更新层的梯度L2范数。

        对每层计算隐藏状态梯度的 L2 范数。
        使用 MetricsCalculator.compute_l2_norm 计算。
        累加到 _layer_gradient_norm_sum。

        Args:
            hidden_gradients: 隐藏状态梯度字典 {layer_idx: tensor}
        """
        for layer_idx, gradient in hidden_gradients.items():
            if layer_idx >= self._model_info.num_layers:
                continue

            # 计算 L2 范数
            grad_norm = MetricsCalculator.compute_l2_norm(gradient)
            # 累加到缓冲区
            self._layer_gradient_norm_sum[layer_idx] += grad_norm.item() if grad_norm.dim() == 0 else grad_norm.mean().item()

    def _update_attention_gradient_norm(self, attention_gradients: Dict[int, Tensor]) -> None:
        """更新注意力梯度范数。

        对每层计算注意力梯度的 L2 范数，并按头聚合。

        Args:
            attention_gradients: 注意力梯度字典 {layer_idx: tensor}
        """
        for layer_idx, gradient in attention_gradients.items():
            if layer_idx >= self._model_info.num_layers:
                continue

            # gradient 形状可能是 (H,) 或标量
            if gradient.dim() == 0:
                # 标量，均匀分配到所有头
                self._attention_gradient_norm_sum[layer_idx] += gradient.item()
            else:
                # 向量，假设是 (H,)
                num_heads = min(gradient.shape[0], self._model_info.num_heads)
                self._attention_gradient_norm_sum[layer_idx, :num_heads] += gradient[:num_heads].detach().cpu()

    def _update_expert_count(self, expert_assignments: Tensor) -> None:
        """更新专家选中计数。

        统计每个专家被选中的次数，使用 torch.bincount。
        累加到 _expert_selection_count。

        Args:
            expert_assignments: 专家分配索引，形状 (B, L, top_k) 或 (B, L)
        """
        if self._expert_selection_count is None:
            return

        # 展平专家分配索引
        flat_assignments = expert_assignments.view(-1)

        # 使用 bincount 统计每个专家的选中次数
        counts = torch.bincount(
            flat_assignments.long(),
            minlength=len(self._expert_selection_count),
        ).float()

        # 累加到缓冲区
        self._expert_selection_count += counts

    def reset(self) -> None:
        """清空累积器。

        清零所有缓冲区和 _sample_count。
        """
        self._sample_count = 0
        self._head_activation_freq.zero_()
        self._head_concentration_sum.zero_()
        self._layer_gradient_norm_sum.zero_()
        self._attention_gradient_norm_sum.zero_()
        if self._expert_selection_count is not None:
            self._expert_selection_count.zero_()

    def save(self, path: str) -> None:
        """持久化统计状态到磁盘。

        使用 torch.save 保存所有统计状态到文件，
        包含 model_info、所有缓冲区、sample_count。

        Args:
            path: 保存路径
        """
        state = {
            "model_info": self._model_info,
            "sample_count": self._sample_count,
            "head_activation_freq": self._head_activation_freq,
            "head_concentration_sum": self._head_concentration_sum,
            "layer_gradient_norm_sum": self._layer_gradient_norm_sum,
            "attention_gradient_norm_sum": self._attention_gradient_norm_sum,
            "expert_selection_count": self._expert_selection_count,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """从磁盘加载统计状态。

        使用 torch.load 恢复统计状态。

        Args:
            path: 加载路径
        """
        state = torch.load(path, map_location="cpu")

        self._model_info = state["model_info"]
        self._sample_count = state["sample_count"]
        self._head_activation_freq = state["head_activation_freq"]
        self._head_concentration_sum = state["head_concentration_sum"]
        self._layer_gradient_norm_sum = state["layer_gradient_norm_sum"]
        self._attention_gradient_norm_sum = state["attention_gradient_norm_sum"]
        self._expert_selection_count = state.get("expert_selection_count")

    def get_statistics(self) -> AccumulatorState:
        """获取当前统计状态。

        计算并返回当前累积的统计信息：
        - 频率 = _head_activation_freq / _sample_count
        - 平均集中度 = _head_concentration_sum / _sample_count
        - 平均梯度范数 = _layer_gradient_norm_sum / _sample_count

        Returns:
            AccumulatorState: 累积器状态数据类
        """
        if self._sample_count == 0:
            # 避免除零
            return AccumulatorState(
                head_activation_freq=torch.zeros_like(self._head_activation_freq),
                head_attention_concentration=torch.zeros_like(self._head_concentration_sum),
                layer_gradient_norm=torch.zeros_like(self._layer_gradient_norm_sum),
                attention_gradient_norm=torch.zeros_like(self._attention_gradient_norm_sum),
                expert_selection_count=self._expert_selection_count.clone() if self._expert_selection_count is not None else None,
                sample_count=0,
            )

        # 计算平均值
        head_activation_freq = self._head_activation_freq / self._sample_count
        head_attention_concentration = self._head_concentration_sum / self._sample_count
        layer_gradient_norm = self._layer_gradient_norm_sum / self._sample_count
        attention_gradient_norm = self._attention_gradient_norm_sum / self._sample_count

        return AccumulatorState(
            head_activation_freq=head_activation_freq,
            head_attention_concentration=head_attention_concentration,
            layer_gradient_norm=layer_gradient_norm,
            attention_gradient_norm=attention_gradient_norm,
            expert_selection_count=self._expert_selection_count.clone() if self._expert_selection_count is not None else None,
            sample_count=self._sample_count,
        )

    def is_full(self) -> bool:
        """检查是否达到样本上限。

        Returns:
            bool: 如果 _sample_count >= limit 返回 True
        """
        return self._sample_count >= self._limit
