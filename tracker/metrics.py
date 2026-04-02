"""Metrics calculation module.

本模块提供各类指标计算工具，包括注意力熵、集中度、L2范数、激活频率等。
所有方法均为静态方法，可直接调用无需实例化。
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from ..core.types import Tensor


class MetricsCalculator:
    """指标计算工具类。

    提供注意力熵、集中度、L2范数、激活频率等静态计算方法。
    所有方法均为 @staticmethod，无需实例化即可使用。
    """

    @staticmethod
    def compute_attention_entropy(attention: Tensor, eps: float = 1e-10) -> Tensor:
        """计算注意力熵。

        信息熵公式：H = -sum(p * log(p))，其中 p 是注意力权重（概率分布）。
        为避免 log(0)，对 p=0 的情况加 eps 处理。

        Args:
            attention: 注意力矩阵，最后一个维度是概率分布，形状为 (..., seq_len)
            eps: 数值稳定性的小常数，避免 log(0)

        Returns:
            Tensor: 熵值，形状与输入去掉最后一个维度后相同 (...)
        """
        # 确保注意力权重非负且加 eps 避免 log(0)
        p = attention.clamp(min=eps)
        # 计算熵: H = -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p), dim=-1)
        return entropy

    @staticmethod
    def compute_attention_concentration(attention: Tensor, eps: float = 1e-10) -> Tensor:
        """计算注意力集中度 (1 - 归一化熵)。

        注意力集中度 = 1 - H/H_max，其中 H_max = log(seq_len)（均匀分布时的最大熵）。
        值越接近1表示越聚焦，越接近0表示越均匀分布。

        Args:
            attention: 注意力矩阵，最后一个维度是概率分布，形状为 (..., seq_len)
            eps: 数值稳定性的小常数

        Returns:
            Tensor: 集中度值，范围 [0, 1]，形状与输入去掉最后一个维度后相同 (...)
        """
        # 获取序列长度（最后一个维度）
        seq_len = attention.shape[-1]
        # 计算最大熵（均匀分布时的熵）
        h_max = torch.log(torch.tensor(seq_len, dtype=attention.dtype, device=attention.device))
        # 计算实际熵
        entropy = MetricsCalculator.compute_attention_entropy(attention, eps=eps)
        # 归一化熵并计算集中度
        normalized_entropy = entropy / (h_max + eps)
        concentration = 1.0 - normalized_entropy
        # 裁剪到 [0, 1] 范围
        return concentration.clamp(min=0.0, max=1.0)

    @staticmethod
    def compute_l2_norm(
        tensor: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Tensor:
        """计算L2范数。

        L2范数公式：sqrt(sum(x^2))，即欧几里得范数。

        Args:
            tensor: 输入张量
            dim: 求范数的维度，None 表示对整个张量求范数

        Returns:
            Tensor: L2范数结果
        """
        return torch.norm(tensor, p=2, dim=dim)

    @staticmethod
    def compute_activation_frequency(
        attention_maps: Dict[int, Tensor], threshold: float = 1e-6
    ) -> Tensor:
        """计算注意力头的激活频率。

        对每个头，检查是否有非零（>threshold）的注意力权重。
        返回二值张量，1表示该头在当前样本中被激活。

        Args:
            attention_maps: 各层注意力矩阵字典 {layer_idx: (B, H, N, N) 或 (B, H, N)}
            threshold: 非零阈值，小于此值视为未激活

        Returns:
            Tensor: 二值张量，形状 (num_layers, num_heads)，
                1 表示该层该头在对应 batch 中有非零注意力权重
        """
        if not attention_maps:
            return torch.empty(0)

        # 获取层数和头数
        num_layers = len(attention_maps)
        # 从第一层获取头数
        first_layer_idx = min(attention_maps.keys())
        first_attention = attention_maps[first_layer_idx]
        # 注意力形状: (B, H, N, N) 或 (B, H, N)
        num_heads = first_attention.shape[1] if first_attention.dim() >= 2 else 1

        # 初始化激活频率张量 (num_layers, num_heads)，与输入设备一致
        device = first_attention.device
        activation_freq = torch.zeros(num_layers, num_heads, dtype=torch.float32, device=device)

        for layer_idx, attention in attention_maps.items():
            # attention 形状: (B, H, N, N) 或 (B, H, N)
            # 检查每个头是否有非零值
            if attention.dim() == 4:
                # (B, H, N, N) -> 对 N, N 维度检查最大值
                max_per_head = attention.abs().amax(dim=(2, 3))  # (B, H)
            elif attention.dim() == 3:
                # (B, H, N) -> 对 N 维度检查最大值
                max_per_head = attention.abs().amax(dim=2)  # (B, H)
            else:
                continue

            # 检查是否超过阈值 (B, H)
            activated = (max_per_head > threshold).float()  # (B, H)
            # 对 batch 维度取平均，得到该层各头的激活频率
            activation_freq[layer_idx] = activated.mean(dim=0)  # (H,)

        return activation_freq
