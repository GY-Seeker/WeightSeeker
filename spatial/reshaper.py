"""空间重构器 - Patch 粒度转换与上采样

可选模块 — 仅图像输入场景需要。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.types import ModelArchitecture
from ..core.exceptions import InvalidInputError


class SpatialReshaper:
    """空间重构器：Patch 粒度转换与上采样。

    将 Patch 级向量/注意力矩阵重塑为 2D 网格并上采样至原图尺寸。
    支持 ViT 标准 Patch 网格与 Swin Transformer 多 stage 窗口注意力重组。
    """

    def __init__(
        self,
        patch_size: int,
        image_size: Tuple[int, int],
        architecture: ModelArchitecture = ModelArchitecture.VIT,
        num_stages: Optional[int] = None,
    ) -> None:
        """初始化空间重构器。

        Args:
            patch_size: Patch 边长（像素）。
            image_size: 原始图像尺寸 (H, W)。
            architecture: 模型架构，Swin 有额外处理逻辑。
            num_stages: Swin 架构的 stage 数量，非 Swin 可为 None。

        Raises:
            InvalidInputError: architecture=Swin 且 num_stages 未提供时。
        """
        if architecture == ModelArchitecture.SWIN and num_stages is None:
            raise InvalidInputError(
                expected="num_stages != None for Swin architecture",
                actual="num_stages=None",
            )
        self.patch_size = patch_size
        self.image_size = image_size  # (H, W)
        self.architecture = architecture
        self.num_stages = num_stages

        # 基础 Patch 网格尺寸（ViT 场景）
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def patch_to_grid(
        self,
        patch_vector: Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """将 Patch 级一维向量重塑为二维网格。

        Args:
            patch_vector: (B, num_patches) 或 (num_patches,)。
            num_patches_h: 网格高度。
            num_patches_w: 网格宽度。

        Returns:
            (B, num_patches_h, num_patches_w) 或 (num_patches_h, num_patches_w)。

        Raises:
            InvalidInputError: patch 数量与 h*w 不匹配时。
        """
        expected = num_patches_h * num_patches_w
        has_batch = patch_vector.dim() == 2

        n = patch_vector.shape[-1]
        if n != expected:
            raise InvalidInputError(
                expected=f"num_patches={expected}",
                actual=f"num_patches={n}",
            )

        if has_batch:
            return patch_vector.view(patch_vector.shape[0], num_patches_h, num_patches_w)
        else:
            return patch_vector.view(num_patches_h, num_patches_w)

    def upsample_to_image(self, grid: Tensor, method: str = "bilinear") -> Tensor:
        """将 Patch 级二维网格上采样至原图尺寸。

        Args:
            grid: (H, W) 或 (B, H, W)。
            method: "bilinear" 或 "gaussian"。

        Returns:
            (image_h, image_w) 或 (B, image_h, image_w)。

        Raises:
            InvalidInputError: method 不支持时。
        """
        if method not in ("bilinear", "gaussian"):
            raise InvalidInputError(
                expected="method in ['bilinear', 'gaussian']",
                actual=f"method='{method}'",
            )

        target_h, target_w = self.image_size
        has_batch = grid.dim() == 3

        # 统一升维为 (B, 1, H, W) 再插值
        x = grid.unsqueeze(0) if not has_batch else grid
        x = x.unsqueeze(1).float()  # (B, 1, h, w)

        if method == "bilinear":
            out = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
        else:
            # gaussian: 先双线性上采样，再高斯平滑
            out = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            interpolator = Interpolator()
            # 对每个 batch 分别平滑
            out = out.squeeze(1)  # (B, H, W)
            out = interpolator.gaussian_smooth(out)
            out = out.unsqueeze(1)

        out = out.squeeze(1)  # (B, H, W)
        if not has_batch:
            out = out.squeeze(0)
        return out

    def swin_window_reorganize(
        self,
        window_attention: Tensor,
        stage_idx: int,
        feature_h: int,
        feature_w: int,
        window_size: int,
        shift_size: int = 0,
    ) -> Tensor:
        """将 Swin 窗口注意力重组为 (feature_h, feature_w) 的全局注意力图。

        每个 Patch 位置的得分取其所属窗口注意力行的均值（query → keys 均值）。

        Args:
            window_attention: (num_windows, window_size^2, window_size^2)。
            stage_idx: stage 索引（暂留，供多 stage 扩展）。
            feature_h: 当前 stage 特征图高度（Patch 数）。
            feature_w: 当前 stage 特征图宽度（Patch 数）。
            window_size: 窗口大小（Patch 单位）。
            shift_size: 循环位移大小，0 表示不位移。

        Returns:
            (feature_h, feature_w) 的全局注意力图。

        Raises:
            InvalidInputError: feature_h 或 feature_w 不能被 window_size 整除时。
        """
        if feature_h % window_size != 0 or feature_w % window_size != 0:
            raise InvalidInputError(
                expected=f"feature_h,w divisible by window_size={window_size}",
                actual=f"feature_h={feature_h}, feature_w={feature_w}",
            )

        num_win_h = feature_h // window_size
        num_win_w = feature_w // window_size
        ws2 = window_size * window_size

        # window_attention: (num_windows, ws2, ws2)
        # 每个 query token 对所有 key 的注意力均值 → (num_windows, ws2)
        attn_score = window_attention.mean(dim=-1)  # (num_windows, ws2)

        # 重排为 (num_win_h, num_win_w, window_size, window_size)
        attn_score = attn_score.view(num_win_h, num_win_w, window_size, window_size)

        # 拼接回 (feature_h, feature_w)
        global_map = attn_score.permute(0, 2, 1, 3).contiguous()
        global_map = global_map.view(feature_h, feature_w)

        # 若有循环位移，逆位移还原原始坐标
        if shift_size > 0:
            global_map = torch.roll(global_map, shifts=(shift_size, shift_size), dims=(0, 1))

        return global_map

    # ------------------------------------------------------------------
    # 针对注意力矩阵的高阶接口（用户需求扩展）
    # ------------------------------------------------------------------

    def reshape_attention(self, attention: Tensor, layer_idx: int = 0) -> Tensor:
        """将 (B, H, N, N) 注意力矩阵重构为 2D Patch 网格。

        Args:
            attention: (B, num_heads, N, N)，N 可含 CLS token。
            layer_idx: 层索引，Swin 用于确定当前 stage 分辨率。

        Returns:
            (B, num_heads, num_patches_h, num_patches_w)。
        """
        attention = self._handle_cls_token(attention)
        B, H, N, _ = attention.shape
        side = int(math.isqrt(N))
        if side * side != N:
            raise InvalidInputError(
                expected="N is a perfect square",
                actual=f"N={N}",
            )

        # 对每个 query 取其对所有 key 的注意力均值 → (B, H, N)
        attn_mean = attention.mean(dim=-2)  # (B, H, N)
        # reshape 为 2D 网格
        return attn_mean.view(B, H, side, side)

    def reshape_gradient(self, gradient: Tensor) -> Tensor:
        """将梯度重塑为 2D 空间图并取 L2 范数。

        Args:
            gradient: (B, N, D) Patch 级梯度，或 (B, C, H, W) 图像级梯度。

        Returns:
            (B, num_patches_h, num_patches_w) 或 (B, H, W)。
        """
        if gradient.dim() == 3:
            # (B, N, D) → (B, num_patches_h, num_patches_w, D) → L2 → (B, h, w)
            B, N, D = gradient.shape
            side = int(math.isqrt(N))
            grid = gradient.view(B, side, side, D)
            return grid.norm(dim=-1)
        elif gradient.dim() == 4:
            # (B, C, H, W) → L2 over C → (B, H, W)
            return gradient.norm(dim=1)
        else:
            raise InvalidInputError(
                expected="gradient.dim() in [3, 4]",
                actual=f"dim={gradient.dim()}",
            )

    def _handle_cls_token(self, attention: Tensor) -> Tensor:
        """检测并移除 CLS token（若存在）。

        若 N == num_patches_h * num_patches_w + 1，则去掉第 0 个 token。

        Args:
            attention: (B, H, N, N)。

        Returns:
            去除 CLS token 后的 (B, H, N', N')。
        """
        expected = self.num_patches_h * self.num_patches_w
        N = attention.shape[-1]
        if N == expected + 1:
            # 去掉第 0 行和第 0 列
            attention = attention[:, :, 1:, 1:]
        return attention
