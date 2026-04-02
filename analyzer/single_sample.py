"""单样本解释器 - 轨道A：对单个前向/反向传播结果进行多维分析。"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..tracker.metrics import MetricsCalculator


class SingleSampleAnalyzer:
    """对单样本的注意力图和梯度图执行深度分析（层分组、头聚类、层重要性）。"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        threshold_method: str = "median",
        attention_threshold: Optional[float] = None,
        gradient_threshold: Optional[float] = None,
    ) -> None:
        """初始化，threshold_method 支持 'median'/'mean'/'otsu'。"""
        if threshold_method not in ("median", "mean", "otsu"):
            raise ValueError(f"不支持的 threshold_method: {threshold_method}")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.threshold_method = threshold_method
        self.attention_threshold = attention_threshold
        self.gradient_threshold = gradient_threshold

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def analyze(
        self,
        attention_maps: Dict[int, Tensor],
        gradient_maps: Dict[int, Tensor],
        normalized_data: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        """综合分析入口：返回 depth_groups、head_clusters、layer_importance、summary。"""
        depth_groups = self.split_by_depth(attention_maps)
        head_clusters = self.cluster_attention_heads(attention_maps)
        layer_importance = self.compute_layer_importance(gradient_maps)

        # 摘要统计
        importances = list(layer_importance.values())
        summary: Dict[str, Any] = {
            "num_layers_analyzed": len(attention_maps),
            "num_heads": self.num_heads,
            "max_layer_importance": max(importances) if importances else 0.0,
            "min_layer_importance": min(importances) if importances else 0.0,
            "mean_layer_importance": (
                sum(importances) / len(importances) if importances else 0.0
            ),
            "num_clusters": (
                len(set(head_clusters.values())) if head_clusters else 0
            ),
        }

        return {
            "depth_groups": depth_groups,
            "head_clusters": head_clusters,
            "layer_importance": layer_importance,
            "summary": summary,
        }

    def split_by_depth(
        self,
        data: Dict[int, Tensor],
    ) -> Dict[str, Dict[int, Tensor]]:
        """按深度三等分为 shallow/middle/deep 三组。"""
        n = self.num_layers
        third = n // 3
        shallow_end = third
        middle_end = 2 * third

        shallow: Dict[int, Tensor] = {}
        middle: Dict[int, Tensor] = {}
        deep: Dict[int, Tensor] = {}

        for idx, tensor in data.items():
            if idx < shallow_end:
                shallow[idx] = tensor
            elif idx < middle_end:
                middle[idx] = tensor
            else:
                deep[idx] = tensor

        return {"shallow": shallow, "middle": middle, "deep": deep}

    def cluster_attention_heads(
        self,
        attention_maps: Dict[int, Tensor],
    ) -> Dict[Tuple[int, int], int]:
        """K-means 聚类所有注意力头，返回 {(layer_idx, head_idx): cluster_id}。"""
        # 提取每个头的平均注意力分布作为特征向量
        features: list = []
        keys: list = []

        for layer_idx in sorted(attention_maps.keys()):
            attn = attention_maps[layer_idx]  # (B, H, N, N) 或 (H, N, N)
            if attn.dim() == 4:
                attn = attn.mean(dim=0)  # (H, N, N)
            elif attn.dim() == 3:
                pass
            else:
                continue

            num_heads = attn.shape[0]
            for h in range(num_heads):
                head_attn = attn[h]  # (N, N)
                # 对每个 query 求平均，得到 key 维的注意力分布 (N,)
                feat = head_attn.mean(dim=0).float()
                features.append(feat)
                keys.append((layer_idx, h))

        if not features:
            return {}

        # 统一长度（截断或补零到最短）
        min_len = min(f.shape[0] for f in features)
        feat_matrix = torch.stack([f[:min_len] for f in features])  # (total_heads, N)

        # 简单 K-means（k = sqrt(total_heads) 取整，最少2）
        k = max(2, int(len(features) ** 0.5))
        k = min(k, len(features))
        cluster_ids = _kmeans(feat_matrix, k=k, n_iter=20)

        return {keys[i]: cluster_ids[i] for i in range(len(keys))}

    def compute_layer_importance(
        self,
        gradient_maps: Dict[int, Tensor],
    ) -> Dict[int, float]:
        """计算各层梯度 L2 范数均值作为重要性得分。"""
        result: Dict[int, float] = {}
        for layer_idx, grad in gradient_maps.items():
            norm = MetricsCalculator.compute_l2_norm(grad)
            # 若有 batch 维则取均值
            if norm.dim() > 0:
                result[layer_idx] = norm.mean().item()
            else:
                result[layer_idx] = norm.item()
        return result


# ------------------------------------------------------------------
# 内部工具：简单 K-means
# ------------------------------------------------------------------

def _kmeans(data: Tensor, k: int, n_iter: int = 20) -> list:
    """对 (N, D) 张量执行 K-means，返回长度 N 的聚类 ID 列表。"""
    n = data.shape[0]
    if n <= k:
        return list(range(n))

    # 随机初始化中心（选前 k 个不重复样本）
    indices = torch.randperm(n)[:k]
    centers = data[indices].clone().float()
    data_f = data.float()

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(n_iter):
        # 分配步骤：计算每个点到中心的距离
        dists = torch.cdist(data_f, centers)  # (n, k)
        labels = dists.argmin(dim=1)

        # 更新步骤
        new_centers = torch.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = data_f[mask].mean(dim=0)
            else:
                new_centers[c] = centers[c]
        if (new_centers - centers).abs().max().item() < 1e-6:
            break
        centers = new_centers

    return labels.tolist()
