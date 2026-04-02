"""MoE routing handling module."""

from typing import Any, Callable, List

import torch
import torch.nn as nn

from ..core.types import Tensor


class MoEHandler:
    """MoE-Transformer路由处理器。

    该处理器负责：
    - 在模型中的路由门控层（Gate/Router模块）注册Hook，捕获路由输出；
    - 从路由输出中提取每个Token/Patch被分配的专家索引；
    - 统计专家被选中的次数，并计算负载均衡相关指标。
    """

    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        """初始化MoE处理器。

        Args:
            num_experts: 专家数量。
            top_k: 每个token选择的专家数（top-k routing）。
        """
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        # 路由历史记录：保存最近一次或多次前向的专家索引，形状 (B, L, top_k)
        self._routing_history: List[Tensor] = []

    def register_router_hook(self, model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
        """在路由门控层注册Hook。

        遍历模型的所有子模块，查找类名或名称中包含 ``Router``/``Gate``/
        ``Routing`` 等关键字的模块，在其上注册前向Hook，捕获路由输出。

        Hook回调会自动调用 :meth:`capture_expert_assignment`，将每次
        前向的专家分配结果追加到 ``_routing_history`` 中。

        Args:
            model: 包含MoE路由层的模型实例。

        Returns:
            List[RemovableHandle]: 所有注册的Hook句柄列表。
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def _make_hook() -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
            def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:  # noqa: D401
                # 路由输出通常为 (B, L, num_experts) 的logits或概率
                if isinstance(output, torch.Tensor):
                    router_out = output
                elif isinstance(output, (tuple, list)) and output:
                    first = output[0]
                    if isinstance(first, torch.Tensor):
                        router_out = first
                    else:
                        return
                else:
                    return

                assignment = self.capture_expert_assignment(router_out)
                # detach + clone 以避免保留计算图
                self._routing_history.append(assignment.detach().clone())

            return _hook

        for name, module in model.named_modules():
            class_name = type(module).__name__.lower()
            name_lower = name.lower()
            if any(k in class_name for k in ["router", "gate", "routing"]) or any(
                k in name_lower for k in ["router", "gate", "routing"]
            ):
                handle = module.register_forward_hook(_make_hook())
                handles.append(handle)

        return handles

    def capture_expert_assignment(self, router_output: Tensor) -> Tensor:
        """捕获每个Token/Patch被分配的专家索引。

        Args:
            router_output: 路由门控输出张量，形状通常为 ``(B, L, num_experts)``，
                其中最后一维为每个专家的logits或概率。

        Returns:
            Tensor: 每个Token/Patch被选中的top-k专家索引，形状 ``(B, L, top_k)``。
        """
        if router_output.dim() != 3:
            raise ValueError(
                f"router_output 期望为3维张量 (B, L, num_experts)，实际维度为 {router_output.dim()}"
            )

        B, L, num_experts = router_output.shape
        if num_experts != self.num_experts:
            # 为保持稳健性，允许num_experts与配置不符，但仍使用实际值
            self.num_experts = num_experts

        # 使用topk获取索引
        _, topk_idx = torch.topk(router_output, k=min(self.top_k, num_experts), dim=-1)
        return topk_idx

    def get_expert_load_distribution(self) -> Tensor:
        """获取各专家的负载分布。

        统计 ``_routing_history`` 中所有记录里专家被选中的次数总和，
        返回形状为 ``(num_experts,)`` 的计数张量。

        Returns:
            Tensor: 各专家的被选中计数，形状 ``(num_experts,)``。若尚未
            有任何路由记录，则返回全零向量。
        """
        if not self._routing_history:
            return torch.zeros(self.num_experts, dtype=torch.long)

        # 拼接所有历史记录: (N, B, L, top_k) -> (N*B*L*top_k,)
        assignments = torch.stack(self._routing_history, dim=0)
        # 展平并统计直方图
        flat_idx = assignments.view(-1)
        counts = torch.bincount(flat_idx, minlength=self.num_experts)
        return counts

    def compute_load_balance_loss(self, expert_counts: Tensor) -> Tensor:
        """计算负载均衡损失（用于评估MoE路由的均衡性）。

        使用变异系数(Coefﬁcient of Variation, CV)作为不均衡程度度量：

        ``CV = std / mean``

        CV越大，表示负载越不均衡。为避免除零问题，当均值为0时返回0。

        Args:
            expert_counts: 各专家的被选中计数，形状 ``(num_experts,)``。

        Returns:
            Tensor: 标量张量，表示负载不均衡程度，值越大越不均衡。
        """
        if expert_counts.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        counts = expert_counts.to(dtype=torch.float32)
        mean = counts.mean()
        if mean <= 0:
            return torch.tensor(0.0, dtype=torch.float32)

        std = counts.std(unbiased=False)
        cv = std / mean
        return cv
