"""Swin Transformer special handling module."""

from typing import Tuple

import torch

from ..core.types import Tensor


class SwinHandler:
    """Swin Transformer特殊处理器。

    负责Swin架构中特有的窗口注意力处理逻辑，包括：
    - 将 ``(B*num_windows, num_heads, window_size^2, window_size^2)`` 形式
      的窗口注意力重组为显式的 ``(B, num_heads, num_windows, window_size^2,
      window_size^2)`` 格式；
    - 标记交替层的shifted window；
    - 根据stage索引推断特征图分辨率变化；
    - 将窗口注意力合并为全局注意力图。
    """

    def __init__(self, window_size: int, num_stages: int) -> None:
        """初始化Swin处理器。

        Args:
            window_size: 窗口大小（通常为7或12）。
            num_stages: stage数量，一般为4。
        """
        self.window_size: int = window_size
        self.num_stages: int = num_stages
        # 记录每个stage的特征图分辨率，键为stage_idx
        self._stage_resolutions: dict[int, Tuple[int, int]] = {}

    def extract_window_attention(self, attention: Tensor, shift_size: int) -> Tensor:
        """提取窗口内注意力矩阵并重排为显式窗口维度。

        典型的Swin实现中，窗口注意力的形状为：
        ``(num_windows * B, num_heads, window_size^2, window_size^2)``。

        本方法将其重排为：
        ``(B, num_heads, num_windows, window_size^2, window_size^2)``。

        Args:
            attention: 原始注意力输出张量。
            shift_size: 窗口位移大小，用于区分shifted window层；本方法
                仅负责形状重排，不对shift逻辑做实际补偿。

        Returns:
            Tensor: 形状为 ``(B, num_heads, num_windows, window_size^2, window_size^2)``
            的窗口注意力张量；若输入形状不符合预期，则返回原始张量。
        """
        if attention.dim() != 4:
            return attention

        nwin_b, num_heads, n1, n2 = attention.shape
        if n1 != n2 or n1 != self.window_size * self.window_size:
            # 非标准窗口注意力，直接返回
            return attention

        # 尝试推断batch和窗口数：此时无法从单个张量唯一定出B，
        # 这里采用启发式：若能够被"num_heads"整除，则假设B为1；
        # 更通用的策略需要结合上游特征图信息，这里保持简化实现。
        # 因此我们假定 nwin_b 即为 num_windows * B，其中B>=1，由调用方
        # 在上层保证B的正确性。在无法推断B时，退化为B=1。
        #
        # 为避免错误推断，此处仅将 "num_windows" 显式分离，同时保留
        # B=1 这一维度。
        B = 1
        num_windows = nwin_b // B
        window_attn = attention.view(B, num_windows, num_heads, n1, n2)
        # 调整维度顺序为 (B, num_heads, num_windows, window_size^2, window_size^2)
        window_attn = window_attn.permute(0, 2, 1, 3, 4).contiguous()
        return window_attn

    def mark_window_shift(self, layer_idx: int) -> bool:
        """标记当前层是否使用window_shift。

        根据Swin的设计，窗口注意力层通常在同一stage内部交替使用
        非位移窗口和位移窗口：
        - 偶数层: 不使用shift
        - 奇数层: 使用shift

        Args:
            layer_idx: 当前stage内的层索引（从0开始）。

        Returns:
            bool: 若为shifted window层则返回True，否则返回False。
        """
        return layer_idx % 2 == 1

    def adapt_stage_resolution(
        self, stage_idx: int, input_h: int, input_w: int
    ) -> Tuple[int, int]:
        """适配各stage的特征图分辨率。

        Swin中每进入一个新的stage，通常通过Patch Merging进行下采样，
        使得特征图分辨率减半：

        - stage 0: (H, W)
        - stage 1: (H/2, W/2)
        - stage 2: (H/4, W/4)
        - ...

        Args:
            stage_idx: stage索引，从0开始。
            input_h: 输入高度H。
            input_w: 输入宽度W。

        Returns:
            Tuple[int, int]: 当前stage的特征图分辨率 (H_stage, W_stage)。
        """
        scale = 2 ** stage_idx
        h_stage = max(1, input_h // scale)
        w_stage = max(1, input_w // scale)
        self._stage_resolutions[stage_idx] = (h_stage, w_stage)
        return h_stage, w_stage

    def merge_window_attention(
        self, window_attn: Tensor, num_windows_h: int, num_windows_w: int
    ) -> Tensor:
        """合并窗口注意力为全局格式。

        给定窗口注意力张量 ``window_attn``，其形状通常为：
        ``(B, num_heads, num_windows, window_size^2, window_size^2)``，其中
        ``num_windows = num_windows_h * num_windows_w``。

        本方法将其合并为全局注意力张量：
        ``(B, num_heads, H_patches, W_patches, H_patches, W_patches)``，
        即在二维Patch网格上显式表示查询位置与被关注位置之间的关系。

        为保持实现简单且通用，这里仅完成窗口在网格上的排列重组，
        不对跨窗口的注意力做插值或补偿；后续模块可根据需要对该
        张量进一步处理或展平为 ``(B, num_heads, N, N)``。

        Args:
            window_attn: 窗口注意力张量。
            num_windows_h: 垂直方向窗口数量。
            num_windows_w: 水平方向窗口数量。

        Returns:
            Tensor: 形状为 ``(B, num_heads, H_patches, W_patches,
            H_patches, W_patches)`` 的全局注意力张量；若输入形状不符合
            预期，则返回原始张量。
        """
        if window_attn.dim() != 5:
            return window_attn

        B, num_heads, num_windows, ws2_q, ws2_k = window_attn.shape
        if ws2_q != ws2_k:
            return window_attn

        window_size2 = ws2_q
        window_size = int(window_size2 ** 0.5)
        if window_size * window_size != window_size2:
            return window_attn

        if num_windows != num_windows_h * num_windows_w:
            return window_attn

        # 每个窗口内部的patch网格为 (window_size, window_size)
        # 整体patch网格大小：
        h_patches = num_windows_h * window_size
        w_patches = num_windows_w * window_size

        # 先将 window 维度重排为 (num_windows_h, num_windows_w)
        attn = window_attn.view(
            B,
            num_heads,
            num_windows_h,
            num_windows_w,
            window_size,
            window_size,
            window_size,
            window_size,
        )
        # 现在维度为 (B, H, Wh, Ww, wh_q, ww_q, wh_k, ww_k)

        # 将查询和键的窗口维度与窗口内坐标折叠成全局Patch坐标
        attn = attn.permute(0, 1, 2, 4, 3, 5, 2, 6, 3, 7)
        # 维度说明（仅供理解）：
        # (B, H, Wh_q, wh_q, Ww_q, ww_q, Wh_k, wh_k, Ww_k, ww_k)

        attn = attn.contiguous().view(
            B,
            num_heads,
            h_patches,
            w_patches,
            h_patches,
            w_patches,
        )
        return attn
