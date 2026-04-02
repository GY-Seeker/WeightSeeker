"""异常识别器 - 检测全局权重使用中的异常模式。"""

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor


class AnomalyDetector:
    """基于频率-重要性交叉分析，识别冗余头、稀疏关键头及 MoE 负载偏斜。"""

    def __init__(
        self,
        freq_threshold: float = 0.5,
        importance_threshold: float = 0.5,
    ) -> None:
        """初始化，freq_threshold/importance_threshold 均为分位阈值，范围 (0,1)。"""
        if not (0.0 < freq_threshold < 1.0):
            raise ValueError(f"freq_threshold 必须在 (0,1)，当前: {freq_threshold}")
        if not (0.0 < importance_threshold < 1.0):
            raise ValueError(f"importance_threshold 必须在 (0,1)，当前: {importance_threshold}")
        self.freq_threshold = freq_threshold
        self.importance_threshold = importance_threshold

    def detect_redundant_heads(
        self,
        frequency_ranking: List[Dict],
        importance_ranking: List[Dict],
    ) -> List[Tuple[int, int]]:
        """识别高频低效头（冗余头），返回 [(layer_idx, head_idx), ...]。"""
        # 计算频率和重要性的分位阈值
        freqs = [item["freq"] for item in frequency_ranking]
        grad_norms = [item["grad_norm"] for item in importance_ranking]
        if not freqs or not grad_norms:
            return []

        freq_cut = _quantile_value(freqs, self.freq_threshold)
        imp_cut = _quantile_value(grad_norms, self.importance_threshold)

        # 构建重要性查找表
        imp_lookup: Dict[Tuple[int, int], float] = {
            (item["layer_idx"], item["head_idx"]): item["grad_norm"]
            for item in importance_ranking
        }

        result = []
        for item in frequency_ranking:
            key = (item["layer_idx"], item["head_idx"])
            if item["freq"] > freq_cut:
                imp = imp_lookup.get(key, 0.0)
                if imp < imp_cut:
                    result.append(key)
        return result

    def detect_sparse_critical_heads(
        self,
        frequency_ranking: List[Dict],
        importance_ranking: List[Dict],
    ) -> List[Tuple[int, int]]:
        """识别低频高效头（稀疏关键头），返回 [(layer_idx, head_idx), ...]。"""
        freqs = [item["freq"] for item in frequency_ranking]
        grad_norms = [item["grad_norm"] for item in importance_ranking]
        if not freqs or not grad_norms:
            return []

        freq_cut = _quantile_value(freqs, self.freq_threshold)
        imp_cut = _quantile_value(grad_norms, self.importance_threshold)

        imp_lookup: Dict[Tuple[int, int], float] = {
            (item["layer_idx"], item["head_idx"]): item["grad_norm"]
            for item in importance_ranking
        }

        result = []
        for item in frequency_ranking:
            key = (item["layer_idx"], item["head_idx"])
            if item["freq"] < freq_cut:
                imp = imp_lookup.get(key, 0.0)
                if imp > imp_cut:
                    result.append(key)
        return result

    def detect_moe_load_imbalance(
        self,
        expert_counts: Tensor,
        imbalance_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """检测 MoE 专家负载偏斜，返回偏斜度及过载/低负载专家列表。"""
        skewness = self.compute_skewness(expert_counts)
        mean_val = expert_counts.float().mean().item()
        std_val = expert_counts.float().std().item() if expert_counts.numel() > 1 else 0.0

        overloaded = []
        underloaded = []
        for i, cnt in enumerate(expert_counts.tolist()):
            if std_val > 0:
                z = (cnt - mean_val) / std_val
                if z > 1.0:
                    overloaded.append(i)
                elif z < -1.0:
                    underloaded.append(i)

        return {
            "is_imbalanced": abs(skewness) > imbalance_threshold,
            "skewness": skewness,
            "overloaded_experts": overloaded,
            "underloaded_experts": underloaded,
        }

    def compute_skewness(self, distribution: Tensor) -> float:
        """计算分布偏斜度 E[(x-μ)³] / σ³，标准差为零时抛出 ValueError。"""
        x = distribution.float()
        mu = x.mean()
        sigma = x.std(unbiased=False)
        if sigma.item() < 1e-8:
            raise ValueError("标准差为零，无法计算偏斜度（所有值相等）。")
        skew = ((x - mu) ** 3).mean() / (sigma ** 3)
        return skew.item()


def _quantile_value(values: List[float], q: float) -> float:
    """计算列表的 q 分位数（线性插值）。"""
    t = torch.tensor(values, dtype=torch.float32)
    return torch.quantile(t, q).item()
