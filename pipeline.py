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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .core.config import Config
from .core.types import AccumulatorState, ModelArchitecture, ModelInfo

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
    precision: str = Config.PRECISION
    device: Optional[str] = None  # None=自动检测
    accumulator_limit: int = Config.ACCUMULATOR_LIMIT
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
        self._vis_manager = None  # VisualizationManager
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

        # 1. 处理模型：用 InputAdapter 包装（若未包装）
        model = self._original_model
        if not isinstance(model, InputAdapter):
            if self.config.input_adapter_auxiliary is not None:
                logger.info("使用 BIND_AUXILIARY 策略包装模型（辅助输入：%s）",
                            list(self.config.input_adapter_auxiliary.keys()))
            model = InputAdapter.from_signature(
                model,
                auxiliary_inputs=self.config.input_adapter_auxiliary,
            )

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
            # 处理 patch_size 可能是 tuple 的情况
            patch_size_raw = self._model_info.patch_size
            if isinstance(patch_size_raw, tuple):
                patch_size = max(patch_size_raw)  # 取最大值
            else:
                patch_size = max(patch_size_raw, 1)
            image_size = Config.DEFAULT_IMAGE_SIZE
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
            
            # 保存 patch_size 用于后续可视化
            self._patch_size = patch_size

        # 8. 初始化可视化管理器
        heatmap_renderer = None
        image_visualizer = None
        patch_size = 16

        if not self.config.skip_visualization and self.config.save_visualizations:
            try:
                from .visualization.heatmap import HeatmapRenderer
                heatmap_renderer = HeatmapRenderer()
            except NotImplementedError:
                logger.warning("HeatmapRenderer 尚未实现，跳过可视化初始化")

            # 初始化 ImageVisualizer（图像模型专用）
            # 注意：即使 skip_spatial=True，如果是 4D 图像数据也需要 ImageVisualizer
            try:
                from .visualization.image_visualizer import ImageVisualizer
                image_visualizer = ImageVisualizer()
                logger.info("ImageVisualizer 初始化成功（图像模型可视化）")
            except Exception as e:
                logger.warning("ImageVisualizer 初始化失败：%s", e)

        if not self.config.skip_spatial and hasattr(self, '_patch_size'):
            patch_size = self._patch_size

        from .visualization.manager import VisualizationManager
        self._vis_manager = VisualizationManager(
            config=self.config,
            heatmap_renderer=heatmap_renderer,
            image_visualizer=image_visualizer,
            model_info=self._model_info,
            device=self._device,
        )
        if hasattr(self, '_patch_size'):
            self._vis_manager.set_patch_size(self._patch_size)

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
            import traceback
            traceback.print_exc()
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
        if self._vis_manager is not None:
            self._vis_manager.set_last_data(input_data=input_data)

        forward_result = self._forward_tracker.track(
            model=self._model,
            input_data=input_data,
        )

        # 保存输出数据用于图像质量对比
        if not self.config.skip_spatial and self._vis_manager is not None:  # 图像模型
            output = forward_result["output"]
            self._vis_manager.set_last_data(output_data=output)

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
        """可视化输出阶段：委托 VisualizationManager 执行。"""
        if self._vis_manager is None:
            return {}
        return self._vis_manager.run(
            results, attention_maps, output_dir,
            accumulator=self._accumulator,
        )

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
