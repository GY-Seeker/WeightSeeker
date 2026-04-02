"""
输入预处理器模块

根据 design.md §3.7.3，Preprocessor 针对不同模型架构提供输入预处理管线，
支持图像（PIL / numpy → Tensor）和序列（Tensor → Tensor）两种格式。
"""

from typing import Callable, Tuple, Union
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray as NDArray

from ..core.types import ModelArchitecture, ModelInfo


class Preprocessor:
    """
    输入预处理器：针对不同模型架构的输入预处理管线。

    主要职责：
    - 图像预处理：读取 / 调整尺寸 / 归一化 → (1, C, H, W) Tensor
    - 序列预处理：标准化 / padding / 截断 → Tensor
    - 为各架构提供默认预处理 transform

    使用时先根据 ModelInfo 初始化，再按需调用各预处理方法。
    """

    def __init__(self, model_info: ModelInfo) -> None:
        """
        初始化预处理器。

        根据 model_info.architecture 选择对应的默认预处理参数：
        - VIT / SWIN：默认 224×224，ImageNet 均值/方差归一化
        - TRANSFORMER / MOE_TRANSFORMER：序列 padding / 截断到 max_length

        Args:
            model_info: 模型信息对象（ModelInfo），包含架构类型、patch 大小等。
        """
        raise NotImplementedError("待实现")

    def preprocess_image(
        self,
        image: Union[str, NDArray],
        target_size: Tuple[int, int],
    ) -> Tensor:
        """
        图像预处理，输出标准化的 4D Tensor。

        处理流程：
        1. 若输入为路径字符串，使用 PIL.Image.open() 读取为 RGB 图像。
        2. 调整图像尺寸至 target_size（双线性插值）。
        3. 转换为 torch.Tensor，值域 [0, 1]。
        4. 应用 ImageNet 归一化（mean=[0.485, 0.456, 0.406]，std=[0.229, 0.224, 0.225]）。
        5. 添加 batch 维度，返回形状 (1, C, H, W)。

        Args:
            image: 图像来源。可以是：
                   - 图像文件路径（str），支持 PNG / JPEG / BMP 等常见格式。
                   - numpy 数组（NDArray），形状 (H, W, C) 或 (H, W)，值域 [0, 255]，dtype uint8。
            target_size: 目标尺寸元组 (H, W)，例如 (224, 224)。
                         需在 [Config.MIN_IMAGE_SIZE, Config.MAX_IMAGE_SIZE] 范围内。

        Returns:
            Tensor: 预处理后的图像张量，形状 (1, C, H, W)，dtype float32。

        Raises:
            FileNotFoundError: image 为路径字符串且文件不存在时抛出。
            InvalidInputError: target_size 超出允许范围时抛出。
        """
        raise NotImplementedError("待实现")

    def preprocess_sequence(self, sequence: Tensor) -> Tensor:
        """
        序列预处理（ECG / NLP 等 1D 时序数据）。

        处理流程：
        1. 若序列长度超过 Config.MAX_SEQUENCE_LENGTH，截断至上限。
        2. 若需要 padding（由 model_info 决定），用零填充至固定长度。
        3. 转换 dtype 为 float32（若不是）。
        4. 保证输出至少为 2D（添加 batch 维度，若输入为 1D）。

        Args:
            sequence: 原始序列张量。支持形状：
                      - (L,)：单条序列，自动添加 batch 维度变为 (1, L)
                      - (B, L)：批次序列
                      - (B, C, L)：多通道批次序列（如 12 导联 ECG）

        Returns:
            Tensor: 预处理后的序列张量，dtype float32。
                    形状与输入一致（保持 batch 维度）。
        """
        raise NotImplementedError("待实现")

    def get_default_transform(self, architecture: ModelArchitecture) -> Callable:
        """
        获取指定架构的默认预处理 transform（可直接传入 DataManager）。

        各架构对应的默认 transform：
        - VIT / SWIN：Resize → CenterCrop(224) → ToTensor → Normalize(ImageNet)
        - TRANSFORMER / MOE_TRANSFORMER：仅 ToTensor（序列数据不做额外归一化）

        Args:
            architecture: 模型架构枚举（ModelArchitecture）。

        Returns:
            Callable: 可直接作为 torchvision.transforms 使用的预处理函数。
                      对图像输入为 transforms.Compose 对象；
                      对序列输入为 lambda 函数（恒等变换或简单归一化）。
        """
        raise NotImplementedError("待实现")
