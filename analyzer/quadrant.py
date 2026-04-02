"""四象限分析器 - 注意力-梯度联合分类。"""

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..core.types import Quadrant


class QuadrantAnalyzer:
    """将注意力图和梯度图逐点映射到四个语义象限。"""

    def __init__(self, threshold_method: str = "median") -> None:
        """初始化四象限分析器，threshold_method 支持 'median'/'mean'/'otsu'。"""
        if threshold_method not in ("median", "mean", "otsu"):
            raise ValueError(f"不支持的 threshold_method: {threshold_method}")
        self.threshold_method = threshold_method

    def compute_threshold(
        self,
        attention: Tensor,
        gradient: Tensor,
    ) -> Tuple[float, float]:
        """分别计算注意力和梯度的阈值，返回 (attn_threshold, grad_threshold)。"""
        def _calc(t: Tensor) -> float:
            f = t.float().flatten()
            if self.threshold_method == "median":
                return torch.median(f).item()
            elif self.threshold_method == "mean":
                return f.mean().item()
            else:  # otsu
                return _otsu_threshold(f)

        return _calc(attention), _calc(gradient)

    def classify_quadrant(
        self,
        attention_value: float,
        gradient_value: float,
        attn_threshold: float,
        grad_threshold: float,
    ) -> Quadrant:
        """将单点 (attention_value, gradient_value) 分配到对应象限。"""
        high_a = attention_value >= attn_threshold
        high_g = gradient_value >= grad_threshold
        if high_a and high_g:
            return Quadrant.CORE_DISCRIMINATIVE
        elif high_a and not high_g:
            return Quadrant.REDUNDANT_ATTENTION
        elif not high_a and high_g:
            return Quadrant.POTENTIAL_INFLUENCE
        else:
            return Quadrant.IRRELEVANT

    def generate_quadrant_map(
        self,
        attention_map: Tensor,
        gradient_map: Tensor,
    ) -> Tensor:
        """生成逐点象限分类图，返回 (H, W) 整数张量（Quadrant.value）。"""
        if attention_map.shape != gradient_map.shape:
            raise ValueError(
                f"形状不一致: {attention_map.shape} vs {gradient_map.shape}"
            )
        attn_t, grad_t = self.compute_threshold(attention_map, gradient_map)

        a = attention_map.float()
        g = gradient_map.float()

        # 向量化判断四个象限
        high_a = a >= attn_t
        high_g = g >= grad_t

        # 按优先级赋值：CORE=1, REDUNDANT=2, POTENTIAL=3, IRRELEVANT=4
        result = torch.full(a.shape, Quadrant.IRRELEVANT.value, dtype=torch.long)
        result[high_a & ~high_g] = Quadrant.REDUNDANT_ATTENTION.value
        result[~high_a & high_g] = Quadrant.POTENTIAL_INFLUENCE.value
        result[high_a & high_g] = Quadrant.CORE_DISCRIMINATIVE.value
        return result

    def compute_quadrant_statistics(
        self,
        quadrant_map: Tensor,
    ) -> Dict[Quadrant, float]:
        """统计各象限面积占比，返回 {Quadrant: ratio}，总和为 1.0。"""
        total = quadrant_map.numel()
        stats: Dict[Quadrant, float] = {}
        for q in Quadrant:
            count = (quadrant_map == q.value).sum().item()
            stats[q] = count / total if total > 0 else 0.0
        return stats


def _otsu_threshold(flat: Tensor) -> float:
    """Otsu 双峰阈值：最大化类间方差。"""
    # 使用 256 档离散化
    num_bins = 256
    min_v = flat.min().item()
    max_v = flat.max().item()
    if abs(max_v - min_v) < 1e-8:
        return min_v
    bins = torch.linspace(min_v, max_v, num_bins + 1)
    hist = torch.histc(flat, bins=num_bins, min=min_v, max=max_v)
    hist = hist / hist.sum()

    best_thresh = min_v
    best_var = -1.0
    cumsum = torch.cumsum(hist, dim=0)
    cummean = torch.cumsum(hist * bins[:-1], dim=0)
    total_mean = cummean[-1].item()

    for i in range(1, num_bins - 1):
        w0 = cumsum[i].item()
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = cummean[i].item() / w0
        m1 = (total_mean - cummean[i].item()) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = bins[i].item()
    return best_thresh
