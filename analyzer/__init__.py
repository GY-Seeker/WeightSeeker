"""单样本解释与全局权重诊断引擎模块

本模块实现分析流水线的模块4，提供两条分析轨道：
    - 轨道A（单样本解释）：针对单个输入样本，分析注意力头行为、
      层重要性、四象限分布，生成可解释性报告。
    - 轨道B（全局诊断）：基于跨样本累积统计，识别冗余头、
      稀疏关键头、MoE 负载偏斜等全局模式。

融合功能已从原 fusion/ 模块并入本模块的 fusion_utils.py，
以简化的工具函数形式提供加权融合和 GradCAM 式融合。

公开接口：
    - :class:`SingleSampleAnalyzer`：单样本解释器（轨道A）
    - :class:`GlobalDiagnosisEngine`：全局诊断引擎（轨道B）
    - :class:`QuadrantAnalyzer`：四象限划分分析器
    - :class:`AnomalyDetector`：异常模式识别器
    - :func:`weighted_sum_fusion`：加权求和融合工具函数
    - :func:`gradcam_fusion`：GradCAM 式融合工具函数
    - :func:`normalize_for_fusion`：融合前归一化工具函数
"""

from .single_sample import SingleSampleAnalyzer
from .global_diagnosis import GlobalDiagnosisEngine
from .quadrant import QuadrantAnalyzer
from .anomaly_detector import AnomalyDetector
from .fusion_utils import weighted_sum_fusion, gradcam_fusion, normalize_for_fusion
from .token_importance import compute_token_importance_with_fallback

__all__ = [
    "SingleSampleAnalyzer",
    "GlobalDiagnosisEngine",
    "QuadrantAnalyzer",
    "AnomalyDetector",
    "weighted_sum_fusion",
    "gradcam_fusion",
    "normalize_for_fusion",
    "compute_token_importance_with_fallback",
]
