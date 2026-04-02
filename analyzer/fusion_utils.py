"""融合工具函数 - 注意力与梯度的简化融合

原 ``fusion/`` 独立模块并入本文件，去除策略模式、抽象基类和工厂模式，
只保留最基础的两种融合方法作为无状态工具函数。

设计说明：
    - 对于 ECG/NLP 等非图像模型，注意力矩阵本身即为最终结果，
      融合不是必需步骤，可在 pipeline 中设置 skip_fusion=True 跳过。
    - 对于图像模型（ViT/Swin 等），融合可将注意力与梯度结合，
      生成更全面的重要性图（Saliency Map）。
    - 使用前，建议先对 attention 和 gradient 调用 normalize_for_fusion，
      将两者统一到 [0, 1] 数值范围，避免量纲差异导致融合偏差。

支持的融合方式：
    1. :func:`weighted_sum_fusion`：加权求和融合（α×注意力 + (1-α)×梯度）
    2. :func:`gradcam_fusion`：GradCAM 式融合（注意力 × 梯度）

工具函数：
    3. :func:`normalize_for_fusion`：百分位裁剪 + Min-Max 归一化预处理
"""

from torch import Tensor


def weighted_sum_fusion(
    attention: Tensor,
    gradient: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """加权求和融合：将注意力图和梯度图按权重线性叠加。

    公式：output = alpha * attention + (1 - alpha) * gradient

    特点：
        - 两者均有贡献，其中任一信号增强时输出随之增强。
        - alpha = 0.5 时等权重融合；alpha → 1 时趋近纯注意力图；
          alpha → 0 时趋近纯梯度图。
        - 适合需要"综合考量"注意力和梯度的场景。

    Args:
        attention: 注意力张量，应归一化至 [0, 1]。
                   推荐先调用 normalize_for_fusion 预处理。
                   形状任意，但应与 gradient 完全一致。
        gradient: 梯度张量，应归一化至 [0, 1]。
                  形状应与 attention 完全一致。
        alpha: 注意力权重系数，范围 [0, 1]，默认 0.5。
               梯度权重自动设为 (1 - alpha)。

    Returns:
        Tensor: 融合后的重要性分数图，值域 [0, 1]，形状与输入相同。

    Raises:
        ValueError: 当 alpha 不在 [0, 1] 范围内时。
        ValueError: 当 attention 和 gradient 形状不一致时。
    """
    raise NotImplementedError("待实现")


def gradcam_fusion(
    attention: Tensor,
    gradient: Tensor,
) -> Tensor:
    """GradCAM 式融合：注意力图与梯度图逐元素相乘。

    公式：output = attention * gradient（再归一化至 [0, 1]）

    特点：
        - 仅当注意力和梯度均高时输出才高，实现"联合确认"效果。
        - 若任一信号为零，输出也为零，具有较强的抑制能力。
        - 适合需要精确定位"模型既关注又依赖"区域的场景。
        - 相比 weighted_sum_fusion，输出更稀疏，背景噪声更少。

    Args:
        attention: 注意力张量，应归一化至 [0, 1]。
                   推荐先调用 normalize_for_fusion 预处理。
                   形状任意，但应与 gradient 完全一致。
        gradient: 梯度张量，应归一化至 [0, 1]。
                  形状应与 attention 完全一致。

    Returns:
        Tensor: 融合后的重要性分数图，值域 [0, 1]，形状与输入相同。
                输出经过 Min-Max 重归一化，以确保最大值为 1.0。

    Raises:
        ValueError: 当 attention 和 gradient 形状不一致时。
    """
    raise NotImplementedError("待实现")


def normalize_for_fusion(
    tensor: Tensor,
    low_percentile: float = 0.01,
    high_percentile: float = 0.99,
) -> Tensor:
    """融合前预处理：百分位裁剪 + Min-Max 归一化至 [0, 1]。

    在执行 weighted_sum_fusion 或 gradcam_fusion 之前，
    应对注意力张量和梯度张量分别调用本函数进行归一化，
    以确保两者处于相同的数值范围，避免融合结果被量纲较大的信号主导。

    处理步骤：
        1. 百分位裁剪：将低于 low_percentile 分位的值截断至该分位值，
           将高于 high_percentile 分位的值截断至该分位值（去除极端值）。
        2. Min-Max 归一化：将裁剪后的张量线性映射至 [0, 1]。

    Args:
        tensor: 原始注意力或梯度张量（任意形状和数值范围）。
        low_percentile: 下百分位裁剪点，范围 [0, 1)，默认 0.01（1% 分位）。
        high_percentile: 上百分位裁剪点，范围 (0, 1]，默认 0.99（99% 分位）。

    Returns:
        Tensor: 归一化后的张量，值域 [0, 1]，形状与输入相同。

    Raises:
        ValueError: 当 low_percentile >= high_percentile 时。
    """
    raise NotImplementedError("待实现")
