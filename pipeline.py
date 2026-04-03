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

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .core.config import Config
from .core.types import AccumulatorState, ModelArchitecture, ModelInfo, Quadrant

logger = logging.getLogger(__name__)


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

    # Stage 跳过开关
    skip_spatial: bool = True
    skip_global_diagnosis: bool = False
    skip_fusion: bool = False
    skip_visualization: bool = False

    # 架构探测覆盖
    detector_override: Optional[Dict[str, Any]] = None

    # 输入适配器配置
    input_adapter_auxiliary: Optional[Dict[str, Any]] = None

    # 输出配置
    output_dir: str = "./output"
    save_visualizations: bool = True
    save_raw_tensors: bool = False

    # 运行时配置
    precision: str = "fp32"
    device: Optional[str] = None  # None=自动检测
    accumulator_limit: int = 100000
    loss_fn: Optional[Callable] = None  # 自定义 loss，默认 output.sum()


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
            model=my_model,
            config=PipelineConfig(skip_spatial=True),
        )
        results = pipeline.run_single(input_tensor)

    高级使用方式（批量分析）::

        results = pipeline.run_batch(data_loader, max_samples=500)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        初始化分析流程编排器。

        Args:
            model: 模型实例（可以是 InputAdapter 包装后的模型）。
            config: 流水线运行时配置（PipelineConfig）。
                    控制各 Stage 的跳过开关、架构探测覆盖等。
                    若为 None，使用默认 PipelineConfig()。
        """
        self.config = config or PipelineConfig()
        self._original_model = model

        # 确定运行设备
        if self.config.device is not None:
            self._device = self.config.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 延迟初始化的组件（在 _init_components 中完成）
        self._model: Optional[nn.Module] = None           # 可能被 InputAdapter 包装
        self._raw_model: Optional[nn.Module] = None       # 原始模型（用于 hook/detect）
        self._model_info: Optional[ModelInfo] = None
        self._hook_manager = None
        self._forward_tracker = None
        self._backward_tracker = None
        self._accumulator = None
        self._single_analyzer = None
        self._global_engine = None
        self._spatial_reshaper = None
        self._normalizer = None
        self._heatmap_renderer = None
        self._initialized = False

        # 分析结果存储
        self._last_results: Dict[str, Any] = {}
        self._sample_results: List[Dict[str, Any]] = []

        logger.info(
            "AnalysisPipeline 创建完成：device=%s, skip_spatial=%s, "
            "skip_global_diagnosis=%s, skip_fusion=%s",
            self._device,
            self.config.skip_spatial,
            self.config.skip_global_diagnosis,
            self.config.skip_fusion,
        )

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """初始化所有分析组件（延迟初始化，在首次 run 时执行）。"""
        if self._initialized:
            return

        from .model_adapter.detector import ArchitectureDetector
        from .model_adapter.hooks import HookManager
        from .tracker.forward_tracker import ForwardTracker
        from .tracker.backward_tracker import BackwardTracker
        from .tracker.accumulator import CrossSampleAccumulator
        from .analyzer.single_sample import SingleSampleAnalyzer
        from .analyzer.global_diagnosis import GlobalDiagnosisEngine
        from .data_manager.input_adapter import InputAdapter

        # 1. 处理模型：若有辅助输入，用 InputAdapter 包装
        model = self._original_model
        if self.config.input_adapter_auxiliary is not None:
            if not isinstance(model, InputAdapter):
                logger.info("使用 BIND_AUXILIARY 策略包装模型（辅助输入：%s）",
                            list(self.config.input_adapter_auxiliary.keys()))
                model = InputAdapter.from_signature(
                    model,
                    auxiliary_inputs=self.config.input_adapter_auxiliary,
                )
        else:
            if not isinstance(model, InputAdapter):
                model = InputAdapter.from_signature(model)

        self._model = model.to(self._device) if hasattr(model, 'to') else model

        # 获取原始模型（用于探测和 hook）
        if isinstance(model, InputAdapter):
            self._raw_model = model.get_wrapped_model()
        else:
            self._raw_model = model

        # 2. 架构探测
        detector = ArchitectureDetector()
        detection_result = detector.detect_with_confidence(self._raw_model)
        if detection_result.confidence < ArchitectureDetector.CONFIDENCE_WARNING_THRESHOLD:
            for w in detection_result.warnings:
                logger.warning("[ArchitectureDetector] %s", w)

        self._model_info = detector.detect(
            self._raw_model,
            override=self.config.detector_override,
        )
        logger.info(
            "架构探测完成：%s，层数=%d，头数=%d",
            self._model_info.architecture.name,
            self._model_info.num_layers,
            self._model_info.num_heads,
        )

        # 3. 注册 Hook
        self._hook_manager = HookManager(
            model=self._raw_model,
            model_info=self._model_info,
        )
        try:
            self._hook_manager.register_all_hooks()
            logger.info("Hook 注册成功")
        except Exception as e:
            logger.error("Hook 注册失败：%s", e)
            raise

        # 4. 初始化 Tracker
        self._forward_tracker = ForwardTracker(hook_manager=self._hook_manager)
        self._backward_tracker = BackwardTracker(
            model=self._raw_model,
            hook_manager=self._hook_manager,
        )

        # 5. 初始化累积器
        if not self.config.skip_global_diagnosis:
            self._accumulator = CrossSampleAccumulator(
                model_info=self._model_info,
                limit=self.config.accumulator_limit,
                device=self._device,  # 传入当前设备，确保张量在同一设备上
            )

        # 6. 初始化分析器
        self._single_analyzer = SingleSampleAnalyzer(
            num_layers=self._model_info.num_layers,
            num_heads=self._model_info.num_heads,
        )
        if not self.config.skip_global_diagnosis and self._accumulator is not None:
            self._global_engine = GlobalDiagnosisEngine(accumulator=self._accumulator)

        # 7. 初始化空间重构（图像模型）
        if not self.config.skip_spatial:
            from .spatial.reshaper import SpatialReshaper
            from .spatial.normalizer import Normalizer
            patch_size = max(self._model_info.patch_size, 1)
            image_size = (224, 224)  # 默认图像尺寸
            num_stages = None
            if self._model_info.architecture == ModelArchitecture.SWIN:
                num_stages = 4
            self._spatial_reshaper = SpatialReshaper(
                patch_size=patch_size,
                image_size=image_size,
                architecture=self._model_info.architecture,
                num_stages=num_stages,
            )
            self._normalizer = Normalizer()

        # 8. 初始化可视化（骨架，暂时跳过未实现部分）
        if not self.config.skip_visualization and self.config.save_visualizations:
            try:
                from .visualization.heatmap import HeatmapRenderer
                self._heatmap_renderer = HeatmapRenderer()
            except NotImplementedError:
                logger.warning("HeatmapRenderer 尚未实现，跳过可视化初始化")
                self._heatmap_renderer = None

        self._initialized = True
        logger.info("AnalysisPipeline 组件初始化完成")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_single(self, input_data: Tensor) -> Dict[str, Any]:
        """
        单次分析（轨道A）：前向 → loss → 反向 → 分析。

        Args:
            input_data: 输入张量，形状 (B, ...) 取决于模型类型。

        Returns:
            Dict[str, Any]: 包含注意力、梯度、分析结果的字典。
        """
        self._init_components()
        input_data = input_data.to(self._device)

        results: Dict[str, Any] = {}

        # 前向阶段
        try:
            forward_result = self._run_forward_stage(input_data)
            attention_maps = forward_result["attention_maps"]
            hidden_states = forward_result["hidden_states"]
            output = forward_result["output"]
            results["attention_maps"] = attention_maps
            results["hidden_states"] = hidden_states
            results["output"] = output.detach()
        except Exception as e:
            logger.error("前向阶段失败：%s", e)
            attention_maps = {}
            hidden_states = {}
            output = torch.tensor(0.0, device=self._device, requires_grad=True)
            results["attention_maps"] = attention_maps

        # 反向阶段
        try:
            gradient_maps = self._run_backward_stage(output, input_data)
            results["gradient_maps"] = gradient_maps
        except Exception as e:
            logger.error("反向阶段失败：%s", e)
            gradient_maps = {"input": None, "hidden": {}, "attention": {}}
            results["gradient_maps"] = gradient_maps

        # 空间重构阶段
        try:
            spatial_attention = self._run_spatial_stage(attention_maps, gradient_maps)
            results["spatial_attention"] = spatial_attention
            effective_attention = spatial_attention if spatial_attention else attention_maps
        except Exception as e:
            logger.error("空间重构阶段失败：%s", e)
            effective_attention = attention_maps

        # 分析阶段
        try:
            analysis = self._run_analysis_stage(effective_attention, gradient_maps)
            results.update(analysis)
        except Exception as e:
            logger.error("分析阶段失败：%s", e)

        # 更新累积器
        if not self.config.skip_global_diagnosis and self._accumulator is not None:
            try:
                self._accumulator.update(
                    attention_maps=attention_maps,
                    hidden_gradients=gradient_maps.get("hidden", {}),
                    attention_gradients=gradient_maps.get("attention", {}),
                )
            except Exception as e:
                logger.error("累积器更新失败：%s", e)

        # 可视化阶段
        if not self.config.skip_visualization:
            try:
                vis_paths = self._run_visualization_stage(
                    results, effective_attention, self.config.output_dir
                )
                results["visualization_paths"] = vis_paths
            except Exception as e:
                logger.error("可视化阶段失败：%s", e)

        # 保存原始张量
        if self.config.save_raw_tensors and attention_maps:
            try:
                raw_path = os.path.join(self.config.output_dir, "raw_tensors.pt")
                torch.save({
                    "attention_maps": {k: v.cpu() for k, v in attention_maps.items()},
                }, raw_path)
                results["raw_tensors_path"] = raw_path
            except Exception as e:
                logger.error("保存原始张量失败：%s", e)

        self._last_results = results
        self._sample_results.append(results)
        return results

    def run_batch(
        self,
        data_loader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        批量分析（轨道B）：遍历 data_loader，累积统计，最后执行全局诊断。

        Args:
            data_loader: PyTorch DataLoader，每次 yield 一个 batch。
            max_samples: 最大处理样本数，None 表示处理全部。

        Returns:
            Dict[str, Any]: 累积统计和诊断结果字典。
        """
        self._init_components()

        processed = 0
        batch_results_list = []

        for batch in data_loader:
            if max_samples is not None and processed >= max_samples:
                break

            # 提取 tensor（DataLoader 可能返回 (data, label) 或纯 tensor）
            if isinstance(batch, (list, tuple)):
                input_data = batch[0]
            else:
                input_data = batch

            if not isinstance(input_data, Tensor):
                logger.warning("跳过非 Tensor batch（类型：%s）", type(input_data).__name__)
                continue

            try:
                batch_result = self.run_single(input_data)
                batch_results_list.append(batch_result)
                batch_size = input_data.shape[0]
                processed += batch_size
                logger.info("已处理 %d 个样本", processed)
            except Exception as e:
                logger.error("处理 batch 失败：%s", e)
                continue

        # 全局诊断
        global_diag = {}
        if not self.config.skip_global_diagnosis and self._accumulator is not None:
            try:
                global_diag = self._run_global_diagnosis_stage(self._accumulator)
            except Exception as e:
                logger.error("全局诊断失败：%s", e)

        results = {
            "batch_results": batch_results_list,
            "global_diagnosis": global_diag,
            "accumulator_state": self.get_accumulator_state(),
            "total_samples_processed": processed,
        }
        self._last_results = results
        return results

    # ------------------------------------------------------------------
    # Stage 实现
    # ------------------------------------------------------------------

    def _run_forward_stage(self, input_data: Tensor) -> Dict[str, Any]:
        """前向传播阶段：调用 ForwardTracker.track()。"""
        assert self._forward_tracker is not None, "组件未初始化"
        assert self._model is not None, "模型未初始化"

        # 保存输入数据用于后续可视化
        self._last_input_data = input_data.detach().cpu()

        forward_result = self._forward_tracker.track(
            model=self._model,
            input_data=input_data,
        )
        return {
            "attention_maps": forward_result["attention"],
            "hidden_states": forward_result["hidden_state"],
            "output": forward_result["output"],
        }

    def _run_backward_stage(self, output: Tensor, input_data: Tensor) -> Dict[str, Any]:
        """反向传播阶段：计算 loss 并调用 BackwardTracker.track()。"""
        assert self._backward_tracker is not None, "组件未初始化"

        loss = self._compute_loss(output, input_data)
        gradients = self._backward_tracker.track(loss=loss, input_data=input_data)
        return gradients

    def _run_spatial_stage(
        self,
        attention_maps: Dict[int, Tensor],
        gradient_maps: Dict[str, Any],
    ) -> Dict[int, Tensor]:
        """
        空间重构阶段。

        skip_spatial=True 时直接返回原始注意力图；
        否则通过 SpatialReshaper + Normalizer 重构。
        """
        if self.config.skip_spatial:
            return attention_maps

        if self._spatial_reshaper is None or self._normalizer is None:
            logger.warning("空间重构器未初始化，跳过空间重构")
            return attention_maps

        reshaped: Dict[int, Tensor] = {}
        for layer_idx, attn in attention_maps.items():
            try:
                # attn: (B, H, N, N) → reshape 到 2D 网格
                grid = self._spatial_reshaper.reshape_attention(attn, layer_idx=layer_idx)
                # 归一化
                norm = self._normalizer.normalize_for_visualization(grid)
                reshaped[layer_idx] = norm
            except Exception as e:
                logger.error("层 %d 空间重构失败：%s", layer_idx, e)
                reshaped[layer_idx] = attn

        return reshaped

    def _run_analysis_stage(
        self,
        attention_maps: Dict[int, Tensor],
        gradient_maps: Dict[str, Any],
    ) -> Dict[str, Any]:
        """分析阶段：SingleSampleAnalyzer + 可选 fusion。"""
        from .analyzer.fusion_utils import gradcam_fusion, weighted_sum_fusion, normalize_for_fusion

        results: Dict[str, Any] = {}

        # 准备梯度图（取 hidden 梯度）
        hidden_grads: Dict[int, Tensor] = gradient_maps.get("hidden", {})
        attn_grads: Dict[int, Tensor] = gradient_maps.get("attention", {})

        # 选用注意力梯度（若有），否则用隐藏状态梯度
        grad_for_analysis = attn_grads if attn_grads else hidden_grads

        # SingleSampleAnalyzer 分析
        if self._single_analyzer is not None:
            try:
                single_result = self._single_analyzer.analyze(
                    attention_maps=attention_maps,
                    gradient_maps=grad_for_analysis,
                    normalized_data={},
                )
                results["single_sample"] = single_result
            except Exception as e:
                logger.error("SingleSampleAnalyzer 失败：%s", e)
                results["single_sample"] = {}

        # 融合阶段（可选）
        if not self.config.skip_fusion and attention_maps and grad_for_analysis:
            try:
                # 取第一层做示例融合（可扩展为逐层融合）
                common_layers = sorted(
                    set(attention_maps.keys()) & set(grad_for_analysis.keys())
                )
                fusion_maps: Dict[int, Tensor] = {}
                for layer_idx in common_layers:
                    attn = attention_maps[layer_idx]
                    grad = grad_for_analysis[layer_idx]

                    # 对齐维度：将梯度标量/1D 扩展到与 attn 相同形状
                    if grad.dim() == 0:
                        grad = grad.expand_as(attn)
                    elif grad.shape != attn.shape:
                        # 尝试广播
                        try:
                            grad = grad.expand_as(attn)
                        except RuntimeError:
                            continue

                    attn_norm = normalize_for_fusion(attn)
                    grad_norm = normalize_for_fusion(grad)
                    fused = gradcam_fusion(attn_norm, grad_norm)
                    fusion_maps[layer_idx] = fused

                results["fusion_maps"] = fusion_maps
            except Exception as e:
                logger.error("融合阶段失败：%s", e)
                results["fusion_maps"] = {}

        return results

    def _run_visualization_stage(
        self,
        results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
    ) -> Dict[str, str]:
        """可视化输出阶段：生成热力图和统计图表。"""
        if self.config.skip_visualization:
            return {}

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: Dict[str, str] = {}

        # HeatmapRenderer - 生成所有头的全景热力图
        if self._heatmap_renderer is not None:
            try:
                # 1. 所有层所有头的完整热力图面板（每行 6 个）
                if attention_maps:
                    all_heads_path = os.path.join(output_dir, "all_heads_attention_heatmap.png")
                    self._heatmap_renderer.render_multihead_all_heads(
                        attention_maps=attention_maps,
                        title="All Attention Heads Heatmap (6 columns)",
                        save_path=all_heads_path,
                        num_cols=6,
                    )
                    saved_paths["all_heads_heatmap"] = all_heads_path
                    logger.info("已生成所有头的热力图：%s", all_heads_path)
                
                # ❌【已弃用】传统的单头热力图（无实际意义）
                # attn_path = os.path.join(output_dir, "attention_heatmap.png")
                # if attention_maps:
                #     first_attn = next(iter(attention_maps.values()))
                #     # 取第一个 batch、第一个 head
                #     if first_attn.dim() == 4:
                #         render_attn = first_attn[0, 0]
                #     elif first_attn.dim() == 3:
                #         render_attn = first_attn[0]
                #     else:
                #         render_attn = first_attn
                #     self._heatmap_renderer.render_attention(
                #         render_attn, title="Attention Heatmap (Layer 0, Head 0)", save_path=attn_path
                #     )
                #     saved_paths["attention_heatmap"] = attn_path
                #     logger.info("已生成单头热力图：%s", attn_path)
                    
                # 3. 多层对比面板
                if len(attention_maps) > 1:
                    multi_layer_path = os.path.join(output_dir, "multi_layer_attention.png")
                    self._heatmap_renderer.render_multi_layer(
                        layer_maps=attention_maps,
                        title="Multi-Layer Attention Comparison",
                        save_path=multi_layer_path,
                        num_cols=4,
                    )
                    saved_paths["multi_layer_comparison"] = multi_layer_path
                    logger.info("已生成多层对比图：%s", multi_layer_path)
                    
            except Exception as e:
                logger.error("热力图渲染失败：%s", e)

        # ========== 新增：时序数据融合可视化 ==========
        if self.config.skip_spatial:  # ECG/NLP/时序模型
            try:
                from .analyzer.fusion_utils import compute_token_importance
                from .visualization.timeseries_visualizer import TimeSeriesVisualizer
                
                # 1. 计算 token 重要性（融合注意力和梯度）
                # 改进版本：BackwardTracker 现在返回完整的 (B, L) 梯度而不是标量范数
                gradient_maps = results.get("gradient_maps", {})
                hidden_gradients = gradient_maps.get('hidden', {})
                
                logger.info("开始时序数据融合可视化...")
                logger.info(f"  - 注意力图层数：{len(attention_maps)}")
                logger.info(f"  - 可用梯度层数：{len(hidden_gradients)}")
                
                # 检查是否有可用的完整梯度（(B, L) 形状）
                has_complete_gradients = False
                for grad in hidden_gradients.values():
                    if grad.dim() == 2:  # 完整的 (B, L) 梯度
                        has_complete_gradients = True
                        break
                
                token_importance = None
                
                # 尝试使用真实梯度融合
                if has_complete_gradients:
                    logger.info("  ✓ 检测到完整的 (B, L) 梯度，使用真实梯度融合")
                    try:
                        token_importance = compute_token_importance(
                            attention_maps=attention_maps,
                            hidden_gradients=hidden_gradients,
                            method='gradcam',  # Grad-CAM 融合：梯度 × 注意力
                        )  # (B, L)
                        logger.info(f"  ✓ token_importance 计算成功（梯度×注意力融合），形状：{token_importance.shape}")
                    except Exception as e:
                        logger.error("梯度融合失败，将回退到纯注意力：%s", e)
                        import traceback
                        traceback.print_exc()
                        token_importance = None
                else:
                    logger.warning("  - 未检测到完整的 (B, L) 梯度，将使用纯注意力作为重要性得分")
                
                # Fallback: 如果真实梯度不可用，使用纯注意力对角线
                if token_importance is None:
                    logger.info("  - 使用纯注意力对角线作为 token 重要性")
                    diagonals_per_layer = []
                    for layer_idx, attn in attention_maps.items():
                        if attn.dim() == 4:  # (B, H, L, L)
                            B, H, L, _ = attn.shape
                            diagonal = attn[:, :, range(L), range(L)].mean(dim=1)  # (B, L)
                            diagonals_per_layer.append(diagonal)
                        elif attn.dim() == 3:  # (B, L, L)
                            B, L, _ = attn.shape
                            diagonal = attn[:, range(L), range(L)]  # (B, L)
                            diagonals_per_layer.append(diagonal)
                    
                    if diagonals_per_layer:
                        stacked = torch.stack(diagonals_per_layer, dim=0)  # (num_layers, B, L)
                        token_importance = stacked.mean(dim=0)  # (B, L)
                        # 归一化到 [0, 1]
                        from .analyzer.fusion_utils import normalize_for_fusion
                        token_importance = normalize_for_fusion(token_importance)
                        logger.info(f"  ✓ token_importance 计算成功（纯注意力），形状：{token_importance.shape}")
                
                # 2. 获取原始输入数据（从 forward_tracker 或外部传入）
                # 只要有 token_importance 就执行可视化（不管是梯度融合还是纯注意力）
                logger.info(f"  - 检查可视化条件：hasattr(_last_input_data)={hasattr(self, '_last_input_data')}, token_importance is not None={token_importance is not None}")
                if hasattr(self, '_last_input_data') and token_importance is not None:
                        sequence = self._last_input_data  # (C, L) 或 (L,)
                        
                        # 如果 sequence 是 batch 数据 (B, C, L)，取第一个样本
                        if sequence.dim() == 3:
                            sequence = sequence[0]  # (C, L)
                        logger.info(f"  - 可视化序列形状：{sequence.shape}")
                        logger.info(f"  - token_importance 形状：{token_importance.shape}")
                        
                        # 3. 创建可视化器（统一配置）
                        ts_vis = TimeSeriesVisualizer(
                            sampling_rate=None,
                            time_unit='steps',
                        )
                        
                        # 4. 生成融合可视化图
                        fusion_path = os.path.join(output_dir, "token_importance_fusion.png")
                        try:
                            channel_names = None
                            if hasattr(self, '_get_channel_names'):
                                try:
                                    channel_names = self._get_channel_names()
                                except Exception:
                                    pass
                            
                            logger.info(f"  - 准备生成融合图：sequence.shape={sequence.shape}, token_importance.shape={token_importance[0].shape}")
                            fig = ts_vis.render_sequence_with_attention(
                                sequence=sequence,
                                attention_weights=token_importance[0],  # 取第一个样本
                                channel_names=channel_names,
                                title="Token Importance (Attention × Gradient Fusion)",
                                save_path=fusion_path,
                            )
                            saved_paths["token_importance_fusion"] = fusion_path
                            logger.info("已生成融合可视化：%s", fusion_path)
                        except Exception as e:
                            logger.error("render_sequence_with_attention 失败：%s", e, exc_info=True)
                        
                        # 5. 检测关键位置并标注
                        sequence_np = sequence.cpu().numpy()
                        if sequence_np.ndim == 2:
                            signal_for_detection = sequence_np[0]  # 取第一个通道
                        else:
                            signal_for_detection = sequence_np
                        
                        key_positions = ts_vis.detect_key_positions(
                            signal_for_detection,
                            method='auto'
                        )
                        
                        annotation_path = os.path.join(output_dir, "key_segments_annotation.png")
                        fig = ts_vis.render_sequence_with_attention(
                            sequence=sequence,
                            attention_weights=token_importance[0],
                            key_positions=key_positions,
                            title="Key Segments Annotation",
                            save_path=annotation_path,
                        )
                        saved_paths["key_segments_annotation"] = annotation_path
                        logger.info("已生成关键区段标注：%s", annotation_path)
                        
                        # 6. 多层注意力对比
                        if len(attention_maps) > 1:
                            multi_layer_seq_path = os.path.join(output_dir, "multi_layer_attention_seq.png")
                            layer_sequences = {
                                layer_idx: attn.mean(dim=(0, 1))[0]  # 平均所有头和 batch
                                for layer_idx, attn in attention_maps.items()
                            }
                            fig = ts_vis.render_multi_layer_comparison(
                                layer_sequences=layer_sequences,
                                sequence=sequence,
                                save_path=multi_layer_seq_path,
                            )
                            saved_paths["multi_layer_attention_seq"] = multi_layer_seq_path
                            logger.info("已生成多层注意力对比：%s", multi_layer_seq_path)
                    
            except Exception as e:
                logger.error("时序数据融合可视化失败：%s", e)

        # charts（骨架未实现时跳过）
        try:
            from .visualization.charts import plot_layer_importance, plot_head_scatter
            single = results.get("single_sample", {})
            layer_importance = single.get("layer_importance", {})
            if layer_importance:
                li_path = os.path.join(output_dir, "layer_importance.png")
                fig = plot_layer_importance(layer_importance, save_path=li_path)
                saved_paths["layer_importance"] = li_path
            
            # 新增：头频率 - 重要性散点图（需要累积器数据）
            if not self.config.skip_global_diagnosis and self._accumulator is not None:
                acc_state = self._accumulator.get_statistics()
                scatter_path = os.path.join(output_dir, "head_frequency_importance_scatter.png")
                fig = plot_head_scatter(
                    frequency_data=[],
                    importance_data=[],
                    head_freq=acc_state.head_activation_freq,
                    head_concentration=acc_state.head_attention_concentration,
                    title="Head Frequency vs Importance (Quadrant Analysis)",
                    save_path=scatter_path,
                )
                saved_paths["head_scatter"] = scatter_path
        except NotImplementedError:
            logger.debug("charts 尚未实现，跳过统计图生成")
        except Exception as e:
            logger.error("统计图生成失败：%s", e)

        # 四象限分析（新增）
        if self._heatmap_renderer is not None:
            try:
                from .analyzer.quadrant import QuadrantAnalyzer
                quadrant_analyzer = QuadrantAnalyzer(threshold_method="median")
                
                # 从结果中提取隐藏状态梯度
                hidden_states = results.get("gradient_maps", {}).get("hidden", {})
                
                # 取第一层做示例
                if attention_maps and hidden_states:
                    first_layer = min(attention_maps.keys())
                    attn = attention_maps[first_layer]
                    grad = hidden_states.get(first_layer, attn)
                    
                    # 如果是序列数据，将注意力图视为 1D 空间
                    if self.config.skip_spatial:
                        # 1D 序列模型：使用平均注意力图 (B, L)
                        logger.info("生成序列数据的四象限分析...")
                        # 从 (B, H, L, L) 提取对角线作为 1D 注意力分布
                        if attn.dim() == 4:  # (B, H, L, L)
                            B, H, L, _ = attn.shape
                            # 取每个头的自注意力对角线平均
                            diagonal_attn = torch.stack([
                                attn[:, h, range(L), range(L)].mean(dim=0) 
                                for h in range(H)
                            ], dim=0).mean(dim=0)  # (L,)
                        else:
                            diagonal_attn = attn.mean(dim=(0, 1)) if attn.dim() > 1 else attn.flatten()
                        
                        # 梯度也转为 1D
                        if grad.dim() == 2:  # (B, L)
                            grad_1d = grad.mean(dim=0)  # (L,)
                        else:
                            grad_1d = diagonal_attn.clone()  # fallback
                        
                        # 生成 1D 四象限分类图
                        quad_map_1d = quadrant_analyzer.generate_quadrant_map(
                            diagonal_attn.unsqueeze(0).unsqueeze(-1),  # (1, L, 1)
                            grad_1d.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
                        ).squeeze()  # (L,)
                        
                        # 保存四象限统计到文本文件
                        quad_txt_path = os.path.join(output_dir, "quadrant_analysis.txt")
                        with open(quad_txt_path, "w", encoding="utf-8") as f:
                            f.write("四象限分析报告\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"分析层：Layer {first_layer}\n")
                            f.write(f"阈值方法：中位数\n\n")
                            quad_stats = quadrant_analyzer.compute_quadrant_statistics(quad_map_1d.view(1, -1, 1))
                            for quadrant, ratio in quad_stats.items():
                                f.write(f"{quadrant.name}: {ratio*100:.2f}%\n")
                            f.write("\n说明：\n")
                            f.write("- 核心判别区：高注意力 + 高梯度（真正重要的区域）\n")
                            f.write("- 冗余关注区：高注意力 + 低梯度（可能是噪声）\n")
                            f.write("- 潜在影响区：低注意力 + 高梯度（隐性影响因素）\n")
                            f.write("- 无关区域：低注意力 + 低梯度\n")
                        saved_paths["quadrant_analysis"] = quad_txt_path
                        logger.info("已生成四象限分析：%s", quad_txt_path)
                        
                        # 生成四象限可视化热力图（在原始序列空间上标注）
                        quad_viz_path = os.path.join(output_dir, "quadrant_map_seq.png")
                        fig = self._heatmap_renderer.render_quadrant_map(
                            quad_map_1d.unsqueeze(0).unsqueeze(-1),  # 转为 (1, L, 1) 用于渲染
                            title=f"Four Quadrant Analysis (Sequence Data, Layer {first_layer})",
                            save_path=quad_viz_path
                        )
                        saved_paths["quadrant_viz_seq"] = quad_viz_path
                        logger.info("已生成序列四象限热力图：%s", quad_viz_path)
                    else:
                        # 图像模型：正常的 2D 四象限
                        logger.info("生成图像数据的四象限分析...")
                        
                        # 计算四象限分布
                        quad_map = quadrant_analyzer.generate_quadrant_map(
                            attn[0, 0], 
                            grad[0] if grad.dim() > 0 else attn[0, 0]
                        )
                        quad_stats = quadrant_analyzer.compute_quadrant_statistics(quad_map)
                        
                        # 保存四象限统计到文本文件
                        quad_txt_path = os.path.join(output_dir, "quadrant_analysis.txt")
                        with open(quad_txt_path, "w", encoding="utf-8") as f:
                            f.write("四象限分析报告\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"分析层：Layer {first_layer}\n")
                            f.write(f"阈值方法：中位数\n\n")
                            for quadrant, ratio in quad_stats.items():
                                f.write(f"{quadrant.name}: {ratio*100:.2f}%\n")
                            f.write("\n说明：\n")
                            f.write("- 核心判别区：高注意力 + 高梯度（真正重要的区域）\n")
                            f.write("- 冗余关注区：高注意力 + 低梯度（可能是噪声）\n")
                            f.write("- 潜在影响区：低注意力 + 高梯度（隐性影响因素）\n")
                            f.write("- 无关区域：低注意力 + 低梯度\n")
                        saved_paths["quadrant_analysis"] = quad_txt_path
                        logger.info("已生成四象限分析：%s", quad_txt_path)
                        
                        # 生成四象限可视化热力图
                        quad_viz_path = os.path.join(output_dir, "quadrant_map.png")
                        fig = self._heatmap_renderer.render_quadrant_map(
                            quad_map,
                            title=f"Four Quadrant Analysis (Layer {first_layer})",
                            save_path=quad_viz_path
                        )
                        saved_paths["quadrant_viz"] = quad_viz_path
                        logger.info("已生成四象限可视化图：%s", quad_viz_path)
            except Exception as e:
                logger.error("四象限分析失败：%s", e)
        if not self.config.skip_global_diagnosis and self._accumulator is not None:
            try:
                from .visualization.charts import plot_accumulator_stats
                stats_path = os.path.join(output_dir, "accumulator_stats.png")
                acc_state = self._accumulator.get_statistics()
                fig = plot_accumulator_stats(acc_state, save_path=stats_path)
                saved_paths["accumulator_stats"] = stats_path
            except NotImplementedError:
                logger.debug("plot_accumulator_stats 尚未实现")
            except Exception as e:
                logger.error("累积器统计图生成失败：%s", e)
        
        # 生成综合分析报告（新增）
        if not self.config.skip_global_diagnosis and self._accumulator is not None:
            try:
                from .utils.report_generator import generate_analysis_report
                report_path = os.path.join(output_dir, "analysis_report.md")
                acc_state = self._accumulator.get_statistics()
                generate_analysis_report(results, acc_state, report_path)
                saved_paths["analysis_report"] = report_path
                logger.info("已生成分析报告：%s", report_path)
            except Exception as e:
                logger.error("报告生成失败：%s", e)

        return saved_paths

    def _run_global_diagnosis_stage(self, accumulator) -> Dict[str, Any]:
        """全局诊断阶段：调用 GlobalDiagnosisEngine.diagnose()。"""
        if self.config.skip_global_diagnosis:
            return {}

        if self._global_engine is None:
            logger.warning("GlobalDiagnosisEngine 未初始化，跳过全局诊断")
            return {}

        try:
            report = self._global_engine.diagnose()
            return {
                "activation_frequency_ranking": report.activation_frequency_ranking,
                "gradient_importance_ranking": report.gradient_importance_ranking,
                "anomaly_analysis": report.anomaly_analysis,
                "head_classification": report.head_classification,
            }
        except Exception as e:
            logger.error("全局诊断失败：%s", e)
            return {}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _compute_loss(self, output: Tensor, input_data: Tensor) -> Tensor:
        """计算用于反向传播的损失。"""
        if self.config.loss_fn is not None:
            try:
                return self.config.loss_fn(output)
            except Exception as e:
                logger.warning("自定义 loss_fn 失败（%s），回退到 output.sum()", e)

        # 默认：output.sum() 作为代理损失
        if isinstance(output, (tuple, list)):
            # 取第一个张量
            out = output[0] if isinstance(output[0], Tensor) else output
        else:
            out = output

        if isinstance(out, Tensor):
            return out.sum()
        else:
            return torch.tensor(0.0, requires_grad=True, device=self._device)

    def cleanup(self) -> None:
        """移除所有 Hook，重置累积器，清理资源。"""
        if self._hook_manager is not None:
            try:
                self._hook_manager.remove_all_hooks()
                logger.info("Hook 已全部移除")
            except Exception as e:
                logger.error("移除 Hook 失败：%s", e)

        if self._accumulator is not None:
            try:
                self._accumulator.reset()
                logger.info("累积器已重置")
            except Exception as e:
                logger.error("重置累积器失败：%s", e)

        self._last_results = {}
        self._sample_results.clear()
        self._initialized = False
        logger.info("AnalysisPipeline cleanup 完成")

    def get_accumulator_state(self) -> AccumulatorState:
        """返回当前累积状态。"""
        if self._accumulator is None:
            return AccumulatorState()
        return self._accumulator.get_statistics()

    def save_results(self, results: Dict, path: str) -> None:
        """将结果保存到文件（torch.save）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # 只保存可序列化的部分（tensor 和基本类型）
        serializable: Dict[str, Any] = {}
        for k, v in results.items():
            if isinstance(v, Tensor):
                serializable[k] = v.cpu()
            elif isinstance(v, dict):
                serializable[k] = {
                    sk: sv.cpu() if isinstance(sv, Tensor) else sv
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v
        torch.save(serializable, path)
        logger.info("结果已保存到：%s", path)

    def _render_quadrant_bar_chart(
        self,
        quad_stats: Dict[Quadrant, float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制四象限分布条形图（适用于 1D 序列数据）。
        
        Args:
            quad_stats: 四象限统计字典 {Quadrant: ratio}
            save_path: 保存路径，None 则不保存
        """
        import matplotlib.pyplot as plt
        
        # 准备数据
        labels = []
        ratios = []
        colors = []
        
        color_map = {
            Quadrant.CORE_DISCRIMINATIVE: "#d62728",  # 红色
            Quadrant.REDUNDANT_ATTENTION: "#2ca02c",   # 绿色
            Quadrant.POTENTIAL_INFLUENCE: "#1f77b4",   # 蓝色
            Quadrant.IRRELEVANT: "#7f7f7f",            # 灰色
        }
        
        label_map = {
            Quadrant.CORE_DISCRIMINATIVE: "Core Discriminative\n(核心判别区)",
            Quadrant.REDUNDANT_ATTENTION: "Redundant Attention\n(冗余关注区)",
            Quadrant.POTENTIAL_INFLUENCE: "Potential Influence\n(潜在影响区)",
            Quadrant.IRRELEVANT: "Irrelevant\n(无关区域)",
        }
        
        for quadrant in [Quadrant.CORE_DISCRIMINATIVE, Quadrant.REDUNDANT_ATTENTION,
                         Quadrant.POTENTIAL_INFLUENCE, Quadrant.IRRELEVANT]:
            labels.append(label_map[quadrant])
            ratios.append(quad_stats.get(quadrant, 0.0) * 100)
            colors.append(color_map[quadrant])
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        bars = ax.bar(labels, ratios, color=colors, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Four Quadrant Distribution (Sequence Data)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(ratios) * 1.3 if max(ratios) > 0 else 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"四象限条形图已保存到：{save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # 兼容旧接口（design.md §3.9 中的接口签名）
    # ------------------------------------------------------------------

    def run(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整分析流程（兼容旧接口，需外部提供数据）。

        此方法依赖外部通过 data_loader 或直接调用 run_single/run_batch。
        若无数据，则仅完成组件初始化并返回空结果。

        Args:
            output_dir: 覆盖 PipelineConfig.output_dir 的输出目录（可选）。

        Returns:
            Dict[str, Any]: 最近一次分析结果，或初始化后的空字典。
        """
        if output_dir is not None:
            self.config.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        self._init_components()

        if self._last_results:
            return self._last_results

        return {"status": "initialized", "output_dir": self.config.output_dir}

    def get_single_sample_result(self, sample_idx: int) -> Dict[str, Any]:
        """获取指定样本的单样本分析结果（run_batch 执行后调用）。"""
        if not self._sample_results:
            raise RuntimeError("尚未执行任何分析，请先调用 run_single 或 run_batch")
        if sample_idx < 0 or sample_idx >= len(self._sample_results):
            raise IndexError(
                f"sample_idx={sample_idx} 超出范围，已处理 {len(self._sample_results)} 个样本"
            )
        return self._sample_results[sample_idx]

    def get_global_diagnosis(self) -> Dict[str, Any]:
        """获取全局诊断报告（run_batch 执行后调用）。"""
        if self.config.skip_global_diagnosis:
            raise RuntimeError("skip_global_diagnosis=True，全局诊断已被跳过")
        if self._accumulator is None or self._accumulator.get_statistics().sample_count == 0:
            raise RuntimeError("尚未累积任何样本，请先调用 run_batch")
        return self._run_global_diagnosis_stage(self._accumulator)

    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """生成完整可视化报告（run 执行后调用）。"""
        if not self._last_results and not self._sample_results:
            raise RuntimeError("尚未执行任何分析，请先调用 run_single 或 run_batch")

        out = output_dir or self.config.output_dir
        os.makedirs(out, exist_ok=True)

        # 获取最近一次分析的注意力图
        last = self._last_results
        attention_maps = last.get("attention_maps", {})

        try:
            vis_paths = self._run_visualization_stage(last, attention_maps, out)
            logger.info("报告已生成，输出目录：%s", out)
        except Exception as e:
            logger.error("生成报告失败：%s", e)

        return os.path.abspath(out)
