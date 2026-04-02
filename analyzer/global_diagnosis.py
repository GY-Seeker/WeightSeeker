"""全局权重诊断引擎 - 轨道B：基于跨样本累积统计进行全局诊断。"""

from typing import Any, Dict, List, Tuple

from ..core.types import DiagnosisReport, HeadClassification
from ..tracker.accumulator import CrossSampleAccumulator
from .anomaly_detector import AnomalyDetector


class GlobalDiagnosisEngine:
    """消费 CrossSampleAccumulator 累积统计，输出结构化 DiagnosisReport。"""

    def __init__(self, accumulator: CrossSampleAccumulator) -> None:
        """初始化，接收已完成累积的 CrossSampleAccumulator 实例。"""
        self.accumulator = accumulator

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def diagnose(self) -> DiagnosisReport:
        """执行完整诊断，返回 DiagnosisReport。"""
        freq_ranking = self.rank_by_activation_frequency()
        grad_ranking = self.rank_by_gradient_importance()
        categories = self.categorize_heads_by_frequency()

        # 异常检测
        detector = AnomalyDetector()
        head_ranking = grad_ranking.get("head_ranking", [])
        redundant = detector.detect_redundant_heads(freq_ranking, head_ranking)
        sparse_critical = detector.detect_sparse_critical_heads(freq_ranking, head_ranking)

        state = self.accumulator.get_statistics()
        moe_result: Dict[str, Any] = {}
        if state.expert_selection_count is not None:
            try:
                moe_result = detector.detect_moe_load_imbalance(
                    state.expert_selection_count
                )
            except ValueError:
                moe_result = {"is_imbalanced": False, "skewness": 0.0}

        anomaly_analysis: Dict[str, Any] = {
            "redundant_heads": redundant,
            "sparse_critical_heads": sparse_critical,
            "moe_load_imbalance": moe_result,
        }

        # 构建 head_classification（每个头附带 HeadClassification 数据类）
        head_classification: Dict[str, List[HeadClassification]] = {
            "high_freq_high_focus": [],
            "high_freq_low_focus": [],
            "low_freq": [],
        }

        # 构建梯度重要性查找表
        grad_lookup: Dict[Tuple[int, int], float] = {
            (item["layer_idx"], item["head_idx"]): item["grad_norm"]
            for item in head_ranking
        }

        freq_lookup: Dict[Tuple[int, int], float] = {
            (item["layer_idx"], item["head_idx"]): item["freq"]
            for item in freq_ranking
        }
        conc_lookup: Dict[Tuple[int, int], float] = {
            (item["layer_idx"], item["head_idx"]): item["concentration"]
            for item in freq_ranking
        }

        for cat, pairs in categories.items():
            for (layer_idx, head_idx) in pairs:
                key = (layer_idx, head_idx)
                hc = HeadClassification(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    activation_freq=freq_lookup.get(key, 0.0),
                    concentration=conc_lookup.get(key, 0.0),
                    importance_score=grad_lookup.get(key, 0.0),
                    category=cat,
                )
                head_classification[cat].append(hc)

        return DiagnosisReport(
            activation_frequency_ranking=freq_ranking,
            gradient_importance_ranking=grad_ranking,
            anomaly_analysis=anomaly_analysis,
            head_classification=head_classification,
        )

    # ------------------------------------------------------------------
    # 子方法
    # ------------------------------------------------------------------

    def rank_by_activation_frequency(self) -> List[Dict[str, Any]]:
        """按激活频率降序排名所有注意力头，附带集中度字段。"""
        state = self.accumulator.get_statistics()
        freq = state.head_activation_freq  # (num_layers, num_heads)
        conc = state.head_attention_concentration  # (num_layers, num_heads)

        entries = []
        num_layers, num_heads = freq.shape
        for l in range(num_layers):
            for h in range(num_heads):
                entries.append({
                    "layer_idx": l,
                    "head_idx": h,
                    "freq": freq[l, h].item(),
                    "concentration": conc[l, h].item(),
                })

        entries.sort(key=lambda x: (x["freq"], x["concentration"]), reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank
        return entries

    def rank_by_gradient_importance(self) -> Dict[str, List[Dict[str, Any]]]:
        """按梯度范数降序排名各层和各注意力头。"""
        state = self.accumulator.get_statistics()
        layer_norm = state.layer_gradient_norm  # (num_layers,)

        # 层级排名
        layer_ranking = []
        for l, norm_val in enumerate(layer_norm.tolist()):
            layer_ranking.append({"layer_idx": l, "grad_norm": norm_val})
        layer_ranking.sort(key=lambda x: x["grad_norm"], reverse=True)
        for rank, entry in enumerate(layer_ranking, start=1):
            entry["rank"] = rank

        # 头级排名（若有注意力梯度范数）
        head_ranking: List[Dict[str, Any]] = []
        attn_norm = state.attention_gradient_norm
        if attn_norm is not None and attn_norm.numel() > 0:
            num_layers, num_heads = attn_norm.shape
            for l in range(num_layers):
                for h in range(num_heads):
                    head_ranking.append({
                        "layer_idx": l,
                        "head_idx": h,
                        "grad_norm": attn_norm[l, h].item(),
                    })
            head_ranking.sort(key=lambda x: x["grad_norm"], reverse=True)
            for rank, entry in enumerate(head_ranking, start=1):
                entry["rank"] = rank

        return {"layer_ranking": layer_ranking, "head_ranking": head_ranking}

    def categorize_heads_by_frequency(self) -> Dict[str, List[Tuple[int, int]]]:
        """按激活频率和集中度将头分为三类，阈值取各指标中位数。"""
        import torch

        state = self.accumulator.get_statistics()
        freq = state.head_activation_freq  # (num_layers, num_heads)
        conc = state.head_attention_concentration  # (num_layers, num_heads)

        freq_median = torch.median(freq).item()
        conc_median = torch.median(conc).item()

        high_freq_high_focus: List[Tuple[int, int]] = []
        high_freq_low_focus: List[Tuple[int, int]] = []
        low_freq: List[Tuple[int, int]] = []

        num_layers, num_heads = freq.shape
        for l in range(num_layers):
            for h in range(num_heads):
                f = freq[l, h].item()
                c = conc[l, h].item()
                if f >= freq_median:
                    if c >= conc_median:
                        high_freq_high_focus.append((l, h))
                    else:
                        high_freq_low_focus.append((l, h))
                else:
                    low_freq.append((l, h))

        return {
            "high_freq_high_focus": high_freq_high_focus,
            "high_freq_low_focus": high_freq_low_focus,
            "low_freq": low_freq,
        }
