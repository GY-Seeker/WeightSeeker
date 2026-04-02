"""
推理编排器：提供一键式分析接口

根据 design.md §3.9，pipeline.py 是整个分析系统的主入口，
负责编排所有模块的初始化与执行顺序，支持通过 PipelineConfig
按需跳过特定 Stage（spatial/ 空间重构、全局诊断、融合计算）。

核心设计动机（来自 ECG 模型测试反思）：
- 1D 序列模型（ECG / NLP）无需空间重构，应跳过 spatial/ 阶段
- 多输入模型可通过 InputAdapter 在加载阶段完成适配，无需修改后续所有阶段
- 全局诊断开销较大，单次快速分析时可跳过
- detector_override 用于修正多模态模型的架构探测误判
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .core.config import Config
from .core.types import ModelArchitecture


@dataclass
class PipelineConfig:
    """
    流水线运行时配置（各 Stage 跳过开关及相关参数）。

    使用 PipelineConfig 的好处：
    - 局部修改不影响全局：所有跳过逻辑均通过此配置控制
    - 对 1D 序列模型（ECG / NLP），设 skip_spatial=True 即可跳过空间重构
    - 对多模态模型，通过 detector_override 修正架构探测误判
    - 对快速单样本分析，设 skip_global_diagnosis=True 可大幅提升速度

    示例（ECG 模型配置）::

        config = PipelineConfig(
            skip_spatial=True,
            skip_global_diagnosis=False,
            skip_fusion=False,
            detector_override={"architecture": ModelArchitecture.TRANSFORMER, "num_heads": 8},
            input_adapter_auxiliary={"meta_data": torch.zeros(1, 16)},
        )
    """

    # ------------------------------------------------------------------ #
    # Stage 跳过开关
    # ------------------------------------------------------------------ #
    skip_spatial: bool = False
    """
    跳过空间重构阶段（spatial/ 模块）。

    适用场景：输入为 1D 序列的模型（ECG、NLP 等），
    注意力矩阵本身就是最终结果，无需重构到 2D 像素空间。
    图像模型（ViT、Swin）应保持 False。
    """

    skip_global_diagnosis: bool = False
    """
    跳过全局诊断阶段（CrossSampleAccumulator + GlobalDiagnosisEngine）。

    适用场景：快速单样本分析，不需要跨样本统计信息时设为 True，
    可显著减少运行时间和内存消耗。
    """

    skip_fusion: bool = False
    """
    跳过融合阶段（analyzer/fusion_utils 中的加权融合 / GradCAM 融合）。

    适用场景：只需要原始注意力图或梯度图，不需要融合重要性图时设为 True。
    对于 ECG 等非图像模型，注意力矩阵本身往往已足够，融合不是必需步骤。
    """

    # ------------------------------------------------------------------ #
    # 架构探测覆盖
    # ------------------------------------------------------------------ #
    detector_override: Optional[Dict[str, Any]] = None
    """
    传入 ArchitectureDetector.detect(override=...) 的覆盖参数字典（可选）。

    用途：修正多模态 / 混合架构的自动探测误判。
    当 ArchitectureDetector 置信度 < 0.6 时建议手动指定。

    支持的键（与 ModelInfo 字段对应）：
    - "architecture": ModelArchitecture 枚举值（如 ModelArchitecture.TRANSFORMER）
    - "num_layers": int，层数
    - "num_heads": int，注意力头数
    - "patch_size": int，Patch 大小（图像模型）
    - "hidden_dim": int，隐藏维度
    - "window_size": int（Swin 特有）
    - "num_experts": int（MoE 特有）

    示例::

        detector_override={
            "architecture": ModelArchitecture.TRANSFORMER,
            "num_heads": 8,
            "num_layers": 6,
        }
    """

    # ------------------------------------------------------------------ #
    # 输入适配器配置
    # ------------------------------------------------------------------ #
    input_adapter_auxiliary: Optional[Dict[str, Any]] = None
    """
    传入 InputAdapter 的辅助输入字典（可选，用于 BIND_AUXILIARY 策略）。

    若为 None，AnalysisPipeline 会调用 InputAdapter.from_signature(model)，
    此时若模型有多个必选参数但没有提供辅助输入，会触发警告并回退到 PASSTHROUGH。

    示例（ECG 双输入模型）::

        input_adapter_auxiliary={"meta_data": torch.zeros(1, 16)}
    """

    # ------------------------------------------------------------------ #
    # 输出配置
    # ------------------------------------------------------------------ #
    output_dir: str = "./output"
    """分析结果的输出目录，若不存在会自动创建。"""

    save_visualizations: bool = True
    """是否保存可视化图表到 output_dir。"""

    save_raw_tensors: bool = False
    """是否保存原始注意力/梯度张量到 output_dir（.pt 文件）。文件较大，默认关闭。"""


class AnalysisPipeline:
    """
    分析流程编排器：提供一键式分析接口。

    负责协调以下所有模块的初始化与执行：
    - data_manager/: 模型加载 + InputAdapter 适配 + 数据加载
    - model_adapter/: 架构探测 + Hook 注册
    - tracker/: 前向 / 反向追踪 + 跨样本累积
    - spatial/: [可选] 空间重构（skip_spatial=True 时跳过）
    - analyzer/: 单样本分析 + 全局诊断（可按需跳过）
    - visualization/: HeatmapRenderer + plot_utils 输出

    推荐使用方式（极简）::

        pipeline = AnalysisPipeline(
            model="path/to/model.pth",
            data_path="path/to/data/",
            pipeline_config=PipelineConfig(skip_spatial=True),
        )
        results = pipeline.run()

    高级使用方式（预包装模型）::

        adapter = InputAdapter.from_signature(model, auxiliary_inputs={...})
        pipeline = AnalysisPipeline(
            model=adapter,
            data_path="path/to/data/",
            pipeline_config=PipelineConfig(skip_spatial=True),
        )
        results = pipeline.run()
    """

    def __init__(
        self,
        model: Union[str, "nn.Module"],
        data_path: str,
        config: Optional[Config] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        初始化分析流程编排器。

        若 model 为字符串路径，内部会调用 ModelLoader.load_from_checkpoint() 加载。
        若 model 为 nn.Module 实例（包括已包装的 InputAdapter），直接使用。

        初始化时仅保存参数，不执行加载/探测/注册等耗时操作，
        这些操作在 _init_components() 中延迟执行（run() 时调用）。

        Args:
            model: 模型来源，支持以下形式：
                   - str: 本地模型文件路径（.pth / .pt）或 HuggingFace 模型名
                   - nn.Module: 模型实例（可以是 InputAdapter 包装后的模型）
            data_path: 数据路径，支持：
                       - 图像目录路径（含分类子目录）
                       - 序列数据文件路径（.pt / .npy / .csv）
            config: 全局配置对象（Config）。若为 None，使用默认 Config()。
            pipeline_config: 流水线运行时配置（PipelineConfig）。
                             控制各 Stage 的跳过开关、架构探测覆盖等。
                             若为 None，使用默认 PipelineConfig()（所有 Stage 均执行）。
        """
        raise NotImplementedError("待实现")

    def run(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整分析流程，支持按 PipelineConfig 条件跳过指定 Stage。

        Stage 执行顺序：
        1. 初始化组件（_init_components）：
           - 模型加载（含 InputAdapter 自动适配）
           - 架构探测（应用 detector_override）
           - Hook 注册（内置 TransformerEncoderLayer need_weights 补丁）
        2. 前向 + 反向传播追踪（_run_forward_stage / _run_backward_stage）
        3. 跨样本累积统计（若 skip_global_diagnosis=False）
        4. [可选] 空间重构（若 skip_spatial=False）：_run_spatial_stage
        5. 分析阶段（_run_analysis_stage）：
           - 单样本分析（SingleSampleAnalyzer）
           - [可选] 全局诊断（若 skip_global_diagnosis=False）
           - [可选] 融合计算（若 skip_fusion=False）
        6. 可视化输出（_run_visualization_stage）

        Args:
            output_dir: 覆盖 PipelineConfig.output_dir 的输出目录（可选）。
                        若为 None，使用 pipeline_config.output_dir。

        Returns:
            Dict[str, Any]: 包含所有分析结果的字典，键包括：
                - "single_sample_results": Dict，单样本分析结果
                - "global_diagnosis": Dict（若 skip_global_diagnosis=False）
                - "attention_maps": Dict[int, Tensor]，各层注意力矩阵
                - "gradient_maps": Dict[int, Tensor]，各层梯度矩阵
                - "fusion_maps": Dict（若 skip_fusion=False）
                - "output_dir": str，可视化结果保存目录
        """
        raise NotImplementedError("待实现")

    def _init_components(self) -> None:
        """
        初始化所有分析组件（延迟初始化，在 run() 首次调用时执行）。

        操作顺序：
        1. 加载模型（若 model 为路径字符串，调用 ModelLoader）
        2. 检查 forward 签名，必要时创建 InputAdapter：
           - 若用户已传入 InputAdapter 实例，直接使用
           - 否则调用 InputAdapter.from_signature(model, pipeline_config.input_adapter_auxiliary)
        3. 初始化 ArchitectureDetector，应用 detector_override 探测架构
           - 若置信度 < 0.6，打印警告，建议用户手动指定 detector_override
        4. 初始化 HookManager，调用 register_all_hooks()
           - 自动处理 TransformerEncoderLayer 的 need_weights 问题
        5. 初始化 ForwardTracker 和 BackwardTracker
        6. 初始化 CrossSampleAccumulator（若 skip_global_diagnosis=False）
        7. 初始化 SingleSampleAnalyzer 和 GlobalDiagnosisEngine
        8. 初始化 SpatialReshaper（若 skip_spatial=False）
        9. 初始化 HeatmapRenderer 和 plot_utils 相关组件
        10. 调用 DataManager 加载数据
        """
        raise NotImplementedError("待实现")

    def _run_forward_stage(self, batch_data: Tensor) -> Dict[int, Tensor]:
        """
        前向传播阶段：通过 ForwardTracker 捕获各层注意力矩阵。

        通过 InputAdapter（或原始模型）执行 forward，
        ForwardTracker 从 HookManager 的缓存中提取注意力矩阵。

        Args:
            batch_data: 当前批次输入张量。
                        - 图像：形状 (B, C, H, W)
                        - 序列：形状 (B, L) 或 (B, C, L)

        Returns:
            Dict[int, Tensor]: 各层注意力矩阵，{layer_idx: tensor}。
                               tensor 形状：(B, num_heads, seq_len, seq_len)
        """
        raise NotImplementedError("待实现")

    def _run_backward_stage(self, output: Tensor, input_data: Tensor) -> Dict[str, Any]:
        """
        反向传播阶段：通过 BackwardTracker 计算各类梯度。

        先通过 _compute_loss() 计算损失，再调用 BackwardTracker.track(loss)
        提取输入梯度、隐藏状态梯度和注意力梯度。

        Args:
            output: 模型前向输出张量。
            input_data: 对应的输入批次张量（用于损失计算）。

        Returns:
            Dict[str, Any]: 梯度信息字典，包含：
                - "input": Tensor，输入梯度 (B, C, H, W) 或 (B, L, D)
                - "hidden": Dict[int, Tensor]，各层隐藏状态梯度
                - "attention": Dict[Tuple[int, int], Tensor]，各 (layer, head) 注意力梯度
        """
        raise NotImplementedError("待实现")

    def _run_spatial_stage(self, attention_maps: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """
        空间重构阶段（仅图像模型需要，1D 序列模型通过 skip_spatial=True 跳过）。

        将注意力矩阵从 Patch 级 (num_patches, num_patches) 重构到像素级 (H, W)，
        使用 SpatialReshaper.patch_to_grid() + Normalizer.normalize_for_visualization()。

        Args:
            attention_maps: 各层注意力矩阵，{layer_idx: tensor}。
                            形状：(B, num_heads, num_patches, num_patches)

        Returns:
            Dict[int, Tensor]: 重构后的像素级注意力图，{layer_idx: tensor}。
                               形状：(B, num_heads, H, W)
        """
        raise NotImplementedError("待实现")

    def _run_analysis_stage(
        self,
        attention_maps: Dict[int, Tensor],
        gradient_maps: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        分析阶段：执行单样本分析、全局诊断（可选）和融合计算（可选）。

        执行顺序：
        1. SingleSampleAnalyzer.analyze() → 单样本解释结果
        2. QuadrantAnalyzer.generate_quadrant_map() → 四象限分类图
        3. [skip_global_diagnosis=False] 更新 CrossSampleAccumulator，
           GlobalDiagnosisEngine.diagnose() → 全局诊断报告
        4. [skip_fusion=False] gradcam_fusion() 或 weighted_sum_fusion() → 融合重要性图

        Args:
            attention_maps: 前向阶段捕获的注意力矩阵（经空间重构后，若 skip_spatial=False）。
            gradient_maps: 反向阶段计算的梯度信息字典（含 "input"、"hidden"、"attention"）。

        Returns:
            Dict[str, Any]: 分析结果字典，包含：
                - "single_sample": Dict，单样本分析结果（头聚类、层重要性等）
                - "quadrant_map": Tensor，四象限分类图
                - "fusion_map": Tensor（若 skip_fusion=False）
                - "global_diagnosis": DiagnosisReport（若 skip_global_diagnosis=False）
        """
        raise NotImplementedError("待实现")

    def _run_visualization_stage(
        self,
        analysis_results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        可视化输出阶段：生成热力图和统计图表，保存到 output_dir。

        调用的可视化组件：
        - HeatmapRenderer.render_attention() → 注意力热力图
        - HeatmapRenderer.render_gradient() → 梯度热力图
        - HeatmapRenderer.render_multi_layer() → 多层注意力面板
        - plot_layer_importance() → 层重要性柱状图
        - plot_head_scatter() → 头"频率-重要性"散点图
        - plot_accumulator_stats() → 累积器统计摘要图（若 skip_global_diagnosis=False）

        Args:
            analysis_results: _run_analysis_stage() 的输出结果。
            attention_maps: 各层注意力矩阵（用于多层面板渲染）。
            output_dir: 可视化图表的保存目录。

        Returns:
            Dict[str, str]: 各图表的保存路径字典，例如：
                {
                    "attention_heatmap": "output/attention.png",
                    "layer_importance": "output/layer_importance.png",
                    ...
                }
        """
        raise NotImplementedError("待实现")

    def analyze_batch(self, batch_data: Tensor) -> Dict[str, Any]:
        """
        单批次分析：完整执行前向 + 反向 + 注意力/梯度捕获。

        此方法封装了 _run_forward_stage() + _run_backward_stage() 两个阶段，
        用于逐批次迭代分析数据集。

        Args:
            batch_data: 批次数据张量，形状 (B, ...) 取决于数据类型。

        Returns:
            Dict[str, Any]: 批次分析结果，包含：
                - "attention_maps": Dict[int, Tensor]，各层注意力矩阵
                - "gradient_maps": Dict[str, Any]，梯度信息
                - "output": Tensor，模型输出
        """
        raise NotImplementedError("待实现")

    def _compute_loss(self, output: Tensor, input_data: Tensor) -> Tensor:
        """
        计算用于反向传播的损失。

        支持两种场景：
        - 分类任务：若 output 形状为 (B, num_classes) 且有标签，使用交叉熵损失。
        - 无监督 / 无标签场景：使用 output.sum()（或 output.norm()）作为代理损失，
          仅用于触发梯度回传，不代表真实的训练损失。

        Args:
            output: 模型前向输出张量。支持分类输出 (B, C) 或特征输出 (B, D)。
            input_data: 输入数据张量（分类场景下标签从此提取，若存在）。

        Returns:
            Tensor: 标量损失张量（可直接 .backward()）。
        """
        raise NotImplementedError("待实现")

    def get_single_sample_result(self, sample_idx: int) -> Dict[str, Any]:
        """
        获取指定样本的单样本分析结果（run() 执行后调用）。

        Args:
            sample_idx: 样本在数据集中的索引（从 0 开始）。

        Returns:
            Dict[str, Any]: 单样本分析结果，包含以下键：
                - "attention_maps": Dict[int, Tensor]，各层注意力图
                - "gradient_maps": Dict[int, Tensor]，各层梯度图
                - "fusion_map": Tensor，融合重要性图（若 skip_fusion=False）
                - "quadrant_map": Tensor，四象限分类图

        Raises:
            IndexError: sample_idx 超出数据集范围时抛出。
            RuntimeError: run() 尚未执行时抛出。
        """
        raise NotImplementedError("待实现")

    def get_global_diagnosis(self) -> Dict[str, Any]:
        """
        获取全局诊断报告（run() 执行后调用，且 skip_global_diagnosis=False 时有效）。

        Returns:
            Dict[str, Any]: 全局诊断报告字典，包含：
                - "activation_frequency_ranking": List[Dict]，按激活频率排名的注意力头列表
                - "gradient_importance_ranking": Dict，按梯度范数排名的层/头列表
                - "anomaly_analysis": Dict，异常模式识别结果
                  （高频低效头、低频高效头、MoE 负载偏斜等）
                - "head_classification": Dict，头分类结果
                  （"high_freq_high_focus" / "high_freq_low_focus" / "low_freq"）

        Raises:
            RuntimeError: skip_global_diagnosis=True 或 run() 尚未执行时抛出。
        """
        raise NotImplementedError("待实现")

    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """
        生成完整可视化报告（run() 执行后调用）。

        将所有分析结果的可视化图表和 JSON 摘要保存到指定目录，
        并返回输出目录路径供用户查看。

        Args:
            output_dir: 报告输出目录。若为 None，使用 pipeline_config.output_dir。
                        目录不存在时自动创建。

        Returns:
            str: 报告输出目录的绝对路径。

        Raises:
            RuntimeError: run() 尚未执行时抛出。
        """
        raise NotImplementedError("待实现")
