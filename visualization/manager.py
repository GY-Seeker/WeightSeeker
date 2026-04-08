"""可视化管理器：统一编排所有可视化子模块。

将 pipeline 中的可视化逻辑集中到此类，Pipeline 仅通过 run() 接口调用。
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..analyzer.token_importance import compute_token_importance_with_fallback
from ..spatial.shape_utils import (
    extract_attn_2d,
    extract_grad_2d,
    process_multi_layer_attention,
)

logger = logging.getLogger(__name__)


class VisualizationManager:
    """可视化全流程管理器。

    负责协调 HeatmapRenderer / ImageVisualizer / TimeSeriesVisualizer /
    QuadrantAnalyzer / charts 等子模块，生成所有可视化输出。

    Args:
        config: PipelineConfig 实例（读取 skip_* 开关等配置）。
        heatmap_renderer: HeatmapRenderer 实例（可为 None）。
        image_visualizer: ImageVisualizer 实例（可为 None）。
        model_info: ModelInfo 实例。
        device: 运行设备。
    """

    def __init__(
        self,
        config,
        heatmap_renderer,
        image_visualizer,
        model_info,
        device: str,
    ) -> None:
        self.config = config
        self._heatmap_renderer = heatmap_renderer
        self._image_visualizer = image_visualizer
        self._model_info = model_info
        self._device = device

        # 由 Pipeline 在前向/反向阶段设置
        self._last_input_data: Optional[Tensor] = None
        self._last_output_data: Optional[Tensor] = None
        self._patch_size: int = 16

    def set_last_data(
        self,
        input_data: Optional[Tensor] = None,
        output_data: Optional[Tensor] = None,
    ) -> None:
        """保存最近一次前向传播的输入/输出数据，供可视化使用。"""
        if input_data is not None:
            self._last_input_data = input_data.detach().cpu()
        if output_data is not None:
            self._last_output_data = output_data.detach().cpu()

    def set_patch_size(self, patch_size: int) -> None:
        """设置 patch 大小（由 Pipeline 初始化阶段调用）。"""
        self._patch_size = patch_size

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
        accumulator=None,
    ) -> Dict[str, str]:
        """可视化全流程：协调各子模块生成热力图、统计图表。

        Args:
            results: 分析结果字典（含 attention_maps, gradient_maps 等）。
            attention_maps: 各层注意力图。
            output_dir: 输出目录。
            accumulator: CrossSampleAccumulator 实例（可为 None）。
        """
        if self.config.skip_visualization:
            return {}

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: Dict[str, str] = {}

        # 判断数据类型
        is_image_data, is_timeseries_data = self._classify_data_type()

        # 1. 热力图渲染
        try:
            self._visualize_heatmaps(attention_maps, output_dir, saved_paths)
        except Exception as e:
            logger.error("热力图渲染阶段失败：%s", e)

        # 2. 图像模型可视化
        if (not self.config.skip_spatial or is_image_data) and self._image_visualizer is not None:
            try:
                self._visualize_image_model(results, attention_maps, output_dir, saved_paths)
            except Exception as e:
                logger.error("图像模型专用可视化失败：%s", e)
                import traceback
                traceback.print_exc()

        # 3. 时序数据可视化
        if self.config.skip_spatial and is_timeseries_data:
            try:
                self._visualize_timeseries(results, attention_maps, output_dir, saved_paths)
            except Exception as e:
                logger.error("时序数据融合可视化失败：%s", e)

        # 4. 统计图表
        try:
            self._visualize_charts(results, output_dir, saved_paths, accumulator)
        except Exception as e:
            logger.error("统计图生成阶段失败：%s", e)

        # 5. 四象限分析
        try:
            self._visualize_quadrant_analysis(results, attention_maps, output_dir, saved_paths, is_image_data)
        except Exception as e:
            logger.error("四象限分析失败：%s", e)

        # 6. 累积器统计
        try:
            self._visualize_reports(results, output_dir, saved_paths, accumulator)
        except Exception as e:
            logger.error("报告生成阶段失败：%s", e)

        return saved_paths

    # ------------------------------------------------------------------
    # 数据类型判断
    # ------------------------------------------------------------------

    def _classify_data_type(self) -> Tuple[bool, bool]:
        """判断当前输入是图像还是时序数据。

        Returns:
            (is_image_data, is_timeseries_data)
        """
        is_image_data = False
        is_timeseries_data = False
        if self._last_input_data is not None:
            input_dim = self._last_input_data.dim()
            is_image_data = input_dim == 4       # (B, C, H, W)
            is_timeseries_data = input_dim in [2, 3]  # (C, L) 或 (B, C, L)
        return is_image_data, is_timeseries_data

    # ------------------------------------------------------------------
    # 可视化子方法
    # ------------------------------------------------------------------

    def _visualize_heatmaps(
        self,
        attention_maps: Dict[int, Tensor],
        output_dir: str,
        saved_paths: Dict[str, str],
    ) -> None:
        """生成所有头的全景热力图和多层对比面板。"""
        if self._heatmap_renderer is None:
            return

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

            # 2. 多层对比面板
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

    def _visualize_image_model(
        self,
        results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
        saved_paths: Dict[str, str],
    ) -> None:
        """图像模型专用可视化：patches / 质量对比 / 注意力叠加 / 多层对比 / 梯度叠加。"""
        from .image_visualizer import compute_psnr, compute_ssim

        # 1. Patch 划分可视化
        if self._last_input_data is not None:
            input_data_vis = self._last_input_data
            if input_data_vis.dim() == 4:  # (B, C, H, W)
                input_data_vis = input_data_vis[0]  # 取第一个样本

            patch_viz_path = os.path.join(output_dir, "patch_visualization.png")

            # 计算 token_importance
            token_importance_for_viz = None
            gradient_maps = results.get("gradient_maps", {})
            hidden_gradients = gradient_maps.get('hidden', {})

            if attention_maps and hidden_gradients:
                token_importance_for_viz = compute_token_importance_with_fallback(
                    attention_maps, hidden_gradients
                )

            self._image_visualizer.visualize_patches(
                input_data_vis,
                patch_size=self._patch_size,
                token_importance=token_importance_for_viz,
                save_path=patch_viz_path
            )
            saved_paths["patch_visualization"] = patch_viz_path
            logger.info("已生成 Patch 可视化：%s", patch_viz_path)

        # 2. 图像质量对比（原图 vs 输出）
        if self._last_input_data is not None and self._last_output_data is not None:
            original = self._last_input_data
            reconstructed = self._last_output_data

            if original.dim() == 4:
                original = original[0]
            if reconstructed.dim() == 4:
                reconstructed = reconstructed[0]

            # 计算质量指标
            psnr = compute_psnr(original, reconstructed)
            ssim = compute_ssim(original, reconstructed)
            logger.info("  图像质量指标 - PSNR: %.2f dB, SSIM: %.4f", psnr, ssim)

            comparison_path = os.path.join(output_dir, "image_quality_comparison.png")
            metrics = {'PSNR': psnr, 'SSIM': ssim}
            self._image_visualizer.visualize_image_comparison(
                original, reconstructed,
                metrics=metrics,
                save_path=comparison_path
            )
            saved_paths["image_quality_comparison"] = comparison_path
            logger.info("已生成图像质量对比：%s", comparison_path)

        # 3. 注意力叠加到原始图像
        if attention_maps and self._last_input_data is not None:
            try:
                input_data_vis = self._last_input_data
                if input_data_vis.dim() == 4:
                    input_data_vis = input_data_vis[0]

                # 取第一层注意力
                first_layer = min(attention_maps.keys())
                attn = attention_maps[first_layer]

                # 处理注意力图：(B, H, L, L) → (H, W)
                attn_2d = extract_attn_2d(attn)

                if attn_2d is not None:
                    overlay_path = os.path.join(output_dir, "attention_overlay_on_image.png")
                    self._image_visualizer.overlay_attention_on_image(
                        attn_2d,
                        input_data_vis,
                        alpha=0.5,
                        save_path=overlay_path
                    )
                    saved_paths["attention_overlay"] = overlay_path
                    logger.info("已生成注意力叠加图：%s", overlay_path)
            except Exception as e:
                logger.warning("  注意力叠加生成失败：%s，跳过", e)

        # 4. 多层注意力对比（仅适用于标准注意力，窗口注意力不适合此可视化）
        if len(attention_maps) > 1:
            processed_attention = process_multi_layer_attention(attention_maps)

            if processed_attention:
                multi_layer_path = os.path.join(output_dir, "multi_layer_attention_image.png")
                self._image_visualizer.visualize_multi_layer_attention(
                    processed_attention,
                    num_cols=4,
                    save_path=multi_layer_path
                )
                saved_paths["multi_layer_attention_image"] = multi_layer_path
                logger.info("已生成多层注意力对比（图像）：%s", multi_layer_path)
            else:
                logger.info("  所有层都是窗口注意力，跳过多层对比可视化")

        # 5. 梯度热力图叠加
        gradient_maps = results.get("gradient_maps", {})
        hidden_gradients = gradient_maps.get('hidden', {})

        if hidden_gradients and self._last_input_data is not None:
            input_data_vis = self._last_input_data
            if input_data_vis.dim() == 4:
                input_data_vis = input_data_vis[0]

            # 取第一层梯度
            first_grad_layer = min(hidden_gradients.keys())
            grad = hidden_gradients[first_grad_layer]

            grad_2d = extract_grad_2d(grad)

            if grad_2d is not None:
                grad_overlay_path = os.path.join(output_dir, "gradient_overlay_on_image.png")
                self._image_visualizer.overlay_attention_on_image(
                    grad_2d,
                    input_data_vis,
                    alpha=0.5,
                    save_path=grad_overlay_path
                )
                saved_paths["gradient_overlay"] = grad_overlay_path
                logger.info("已生成梯度叠加图：%s", grad_overlay_path)

    def _visualize_timeseries(
        self,
        results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
        saved_paths: Dict[str, str],
    ) -> None:
        """时序数据融合可视化：token重要性融合 / 关键区段标注 / 多层对比。"""
        from .timeseries_visualizer import TimeSeriesVisualizer

        # 1. 计算 token 重要性（统一方法）
        gradient_maps = results.get("gradient_maps", {})
        hidden_gradients = gradient_maps.get('hidden', {})
        token_importance = compute_token_importance_with_fallback(
            attention_maps, hidden_gradients
        )

        # 2. 获取原始输入数据
        if not (self._last_input_data is not None and token_importance is not None):
            return

        sequence = self._last_input_data  # (C, L) 或 (L,)

        # 如果 sequence 是 batch 数据 (B, C, L)，取第一个样本
        if sequence.dim() == 3:
            sequence = sequence[0]  # (C, L)

        # 3. 创建可视化器（统一配置）
        ts_vis = TimeSeriesVisualizer(
            sampling_rate=None,
            time_unit='steps',
        )

        # 4. 生成融合可视化图
        fusion_path = os.path.join(output_dir, "token_importance_fusion.png")
        try:
            channel_names = None

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
            layer_sequences = {}
            for layer_idx, attn in attention_maps.items():
                if attn.dim() == 4:  # (B, H, L, L)
                    B, H, L, _ = attn.shape
                    # 取自注意力对角线：每个 token 对自身的注意力强度
                    layer_sequences[layer_idx] = attn[:, :, range(L), range(L)].mean(dim=(0, 1))  # (L,)
                elif attn.dim() == 3:  # (B, L, L)
                    B, L, _ = attn.shape
                    layer_sequences[layer_idx] = attn[:, range(L), range(L)].mean(dim=0)  # (L,)
                else:
                    layer_sequences[layer_idx] = attn.flatten()
            fig = ts_vis.render_multi_layer_comparison(
                layer_sequences=layer_sequences,
                sequence=sequence,
                save_path=multi_layer_seq_path,
            )
            saved_paths["multi_layer_attention_seq"] = multi_layer_seq_path
            logger.info("已生成多层注意力对比：%s", multi_layer_seq_path)

    def _visualize_charts(
        self,
        results: Dict[str, Any],
        output_dir: str,
        saved_paths: Dict[str, str],
        accumulator=None,
    ) -> None:
        """统计图表：层重要性图 / 头频率-重要性散点图。"""
        try:
            from .charts import plot_layer_importance, plot_head_scatter
            single = results.get("single_sample", {})
            layer_importance = single.get("layer_importance", {})
            if layer_importance:
                li_path = os.path.join(output_dir, "layer_importance.png")
                fig = plot_layer_importance(layer_importance, save_path=li_path)
                saved_paths["layer_importance"] = li_path

            # 头频率 - 重要性散点图（需要累积器数据）
            if not self.config.skip_global_diagnosis and accumulator is not None:
                acc_state = accumulator.get_statistics()
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

    def _visualize_quadrant_analysis(
        self,
        results: Dict[str, Any],
        attention_maps: Dict[int, Tensor],
        output_dir: str,
        saved_paths: Dict[str, str],
        is_image_data: bool,
    ) -> None:
        """四象限分析：1D序列模型 + 2D图像模型两种路径。"""
        if self._heatmap_renderer is None:
            return

        from ..analyzer.quadrant import QuadrantAnalyzer
        quadrant_analyzer = QuadrantAnalyzer(threshold_method="median")

        # 从结果中提取隐藏状态梯度
        hidden_states = results.get("gradient_maps", {}).get("hidden", {})

        # 取第一层做示例
        if not (attention_maps and hidden_states):
            return

        first_layer = min(attention_maps.keys())
        attn = attention_maps[first_layer]
        grad = hidden_states.get(first_layer, attn)

        # 判断数据类型
        is_sequence_data = not is_image_data and self.config.skip_spatial

        if is_sequence_data:
            self._quadrant_1d(
                quadrant_analyzer, attn, grad, first_layer, output_dir, saved_paths
            )
        else:
            self._quadrant_2d(
                quadrant_analyzer, attn, grad, first_layer, output_dir, saved_paths
            )

    def _quadrant_1d(
        self,
        quadrant_analyzer,
        attn: Tensor,
        grad: Tensor,
        first_layer: int,
        output_dir: str,
        saved_paths: Dict[str, str],
    ) -> None:
        """1D 序列模型的四象限分析。"""
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

        quad_stats = quadrant_analyzer.compute_quadrant_statistics(quad_map_1d.view(1, -1, 1))
        logger.info("四象限统计（序列数据，Layer %d）：%s", first_layer, quad_stats)

        # 生成四象限可视化热力图（在原始序列空间上标注）
        quad_viz_path = os.path.join(output_dir, "quadrant_map_seq.png")
        fig = self._heatmap_renderer.render_quadrant_map(
            quad_map_1d.unsqueeze(0).unsqueeze(-1),  # 转为 (1, L, 1) 用于渲染
            title=f"Four Quadrant Analysis (Sequence Data, Layer {first_layer})",
            save_path=quad_viz_path,
            original_signal=self._last_input_data
        )
        saved_paths["quadrant_viz_seq"] = quad_viz_path
        logger.info("已生成序列四象限热力图：%s", quad_viz_path)

    def _quadrant_2d(
        self,
        quadrant_analyzer,
        attn: Tensor,
        grad: Tensor,
        first_layer: int,
        output_dir: str,
        saved_paths: Dict[str, str],
    ) -> None:
        """2D 图像模型的四象限分析。"""
        # 检查注意力矩阵是否为标准的 (B, H, L, L) 格式
        attn_2d = None
        grad_2d = None

        if attn.dim() == 4:
            B_dim, H_dim, L_dim1, L_dim2 = attn.shape

            # 情况1: 标准全局注意力 (B, H, L, L)
            if L_dim1 == L_dim2:
                # 对 batch 和 heads 取平均
                attn_2d = attn.mean(dim=[0, 1])  # (L, L)
                # 尝试 reshape 到方形网格
                L = attn_2d.shape[0]
                sqrt_L = int(L ** 0.5)
                if sqrt_L * sqrt_L == L:
                    attn_2d = attn_2d.view(sqrt_L, sqrt_L)
                else:
                    logger.warning("  L=%d 不是完美平方数，使用 1D 分析", L)
                    attn_2d = torch.diag(attn_2d) if L_dim1 > 1 else attn_2d.flatten()
            # 情况2: Swin 窗口注意力 (num_windows*B, H, ws², ws²)
            elif L_dim1 != L_dim2:
                # 对窗口内部维度取平均，得到每个窗口的注意力分数
                attn_mean = attn.mean(dim=[1, 2, 3])  # (num_windows*B,)

                # 尝试 reshape 到 2D 网格
                num_windows = B_dim
                sqrt_win = int(num_windows ** 0.5)
                if sqrt_win * sqrt_win == num_windows:
                    attn_2d = attn_mean.view(sqrt_win, sqrt_win)
                else:
                    logger.warning(
                        "  窗口数 %d 不是完美平方数 (sqrt=%.2f)，跳过四象限分析",
                        num_windows, sqrt_win,
                    )
                    attn_2d = None

        # 处理梯度图
        if grad.dim() == 4:  # (B, C, H, W) - 已经是空间维度
            grad_2d = grad[0].mean(dim=0)  # 对通道取平均 (H, W)
        elif grad.dim() == 2:  # (B, L) - 序列/patch 梯度
            grad_1d = grad.mean(dim=0)  # (L,)
            L = grad_1d.shape[0]
            sqrt_L = int(L ** 0.5)
            if sqrt_L * sqrt_L == L:
                grad_2d = grad_1d.view(sqrt_L, sqrt_L)
            else:
                logger.warning("  梯度长度 %d 不是完美平方数", L)
                grad_2d = None

        # 检查是否都能转换为 2D
        if attn_2d is None or grad_2d is None:
            logger.warning(
                "  无法转换为 2D 格式 (attn_2d=%s, grad_2d=%s)，跳过四象限分析",
                attn_2d is not None, grad_2d is not None,
            )
            return

        # 确保形状完全一致
        if attn_2d.shape != grad_2d.shape:
            logger.warning(
                "  形状不一致 %s vs %s，进行插值对齐",
                attn_2d.shape, grad_2d.shape,
            )
            grad_2d = F.interpolate(
                grad_2d.unsqueeze(0).unsqueeze(0).float(),
                size=attn_2d.shape,
                mode='bilinear',
                align_corners=True
            ).squeeze()

        # 计算四象限分布
        quad_map = quadrant_analyzer.generate_quadrant_map(attn_2d, grad_2d)
        quad_stats = quadrant_analyzer.compute_quadrant_statistics(quad_map)
        logger.info("四象限统计（图像数据，Layer %d）：%s", first_layer, quad_stats)

        # 生成四象限可视化热力图
        quad_viz_path = os.path.join(output_dir, "quadrant_map.png")

        # 如果有原图数据，作为背景传入
        background_img = None
        quad_map_resized = quad_map

        if self._last_input_data is not None and self._last_input_data.dim() == 4:
            background_img = self._last_input_data[0]  # (C, H, W)

            # 将四象限图插值到原图尺寸
            if background_img.dim() == 3:
                orig_h, orig_w = background_img.shape[1], background_img.shape[2]
            else:
                orig_h, orig_w = background_img.shape[0], background_img.shape[1]

            quad_h, quad_w = quad_map.shape
            if quad_h != orig_h or quad_w != orig_w:
                quad_map_float = quad_map.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                quad_map_resized = F.interpolate(
                    quad_map_float,
                    size=(orig_h, orig_w),
                    mode='nearest'  # 使用最近邻插值保持离散值
                ).squeeze().long()  # (H, W)

        fig = self._heatmap_renderer.render_quadrant_map(
            quad_map_resized,
            title=f"Four Quadrant Analysis (Layer {first_layer})",
            save_path=quad_viz_path,
            background_image=background_img,
            alpha=0.6
        )
        saved_paths["quadrant_viz"] = quad_viz_path
        logger.info("已生成四象限可视化图：%s", quad_viz_path)

    def _visualize_reports(
        self,
        results: Dict[str, Any],
        output_dir: str,
        saved_paths: Dict[str, str],
        accumulator=None,
    ) -> None:
        """累积器统计图。"""
        if not self.config.skip_global_diagnosis and accumulator is not None:
            try:
                from .charts import plot_accumulator_stats
                stats_path = os.path.join(output_dir, "accumulator_stats.png")
                acc_state = accumulator.get_statistics()
                fig = plot_accumulator_stats(acc_state, save_path=stats_path)
                saved_paths["accumulator_stats"] = stats_path
            except NotImplementedError:
                logger.debug("plot_accumulator_stats 尚未实现")
            except Exception as e:
                logger.error("累积器统计图生成失败：%s", e)
