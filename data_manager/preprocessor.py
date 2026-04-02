"""
输入预处理器模块

根据 design.md §3.7.3，Preprocessor 针对不同模型架构提供输入预处理管线，
支持图像（PIL / numpy → Tensor）和序列（Tensor → Tensor）两种格式。
"""

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray as NDArray
from torch import Tensor

from ..core.config import Config
from ..core.exceptions import InvalidInputError
from ..core.types import ModelArchitecture, ModelInfo

logger = logging.getLogger(__name__)

# ImageNet 均值与标准差
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocessor:
    """
    输入预处理器：针对不同模型架构的输入预处理管线。

    主要职责：
    - 图像预处理：读取 / 调整尺寸 / 归一化 → (1, C, H, W) Tensor
    - 序列预处理：标准化 / padding / 截断 → Tensor
    - 为各架构提供默认预处理 transform
    - 统一 batch 维度、dtype 转换和归一化

    使用时先根据 ModelInfo 初始化，再按需调用各预处理方法。
    """

    def __init__(
        self,
        model_info: Optional[ModelInfo] = None,
        config: Optional[Config] = None,
        architecture: Optional[ModelArchitecture] = None,
    ) -> None:
        """
        初始化预处理器。

        根据 model_info.architecture 选择对应的默认预处理参数：
        - VIT / SWIN：默认 224×224，ImageNet 均值/方差归一化
        - TRANSFORMER / MOE_TRANSFORMER：序列 padding / 截断到 max_length

        Args:
            model_info: 模型信息对象（ModelInfo），包含架构类型、patch 大小等。
                        为 None 时使用默认 ModelInfo。
            config: 全局配置对象，为 None 时自动创建 Config()。
            architecture: 架构类型（可选），优先于 model_info.architecture。
        """
        self.config = config if config is not None else Config()
        self.model_info = model_info if model_info is not None else ModelInfo()

        # 确定架构
        if architecture is not None:
            self.architecture = architecture
        else:
            self.architecture = self.model_info.architecture

        # 根据架构设置默认目标尺寸
        if self.architecture in (ModelArchitecture.VIT, ModelArchitecture.SWIN):
            default_size = self.config.get("DEFAULT_IMAGE_SIZE", self.config.DEFAULT_IMAGE_SIZE)
            if isinstance(default_size, (list, tuple)) and len(default_size) == 2:
                self.default_target_size: Tuple[int, int] = tuple(default_size)  # type: ignore
            else:
                self.default_target_size = (224, 224)
        else:
            self.default_target_size = (224, 224)

        # 最大序列长度
        self.max_sequence_length = int(
            self.config.get("MAX_SEQUENCE_LENGTH", self.config.MAX_SEQUENCE_LENGTH)
        )

        logger.info(
            "Preprocessor 初始化：architecture=%s, target_size=%s, max_seq_len=%d",
            self.architecture,
            self.default_target_size,
            self.max_sequence_length,
        )

    def preprocess(self, data: Tensor) -> Tensor:
        """
        统一预处理流程：确保 batch 维度 → dtype 转换 → 归一化。

        根据 precision 配置转换 dtype：
        - fp32：转为 float32
        - fp16：转为 float16
        - bf16：转为 bfloat16

        Args:
            data: 输入张量，任意形状。

        Returns:
            Tensor: 预处理后的张量。
        """
        # 1. 确保 batch 维度
        data = self.ensure_batch_dim(data)

        # 2. dtype 转换
        precision = str(self.config.get("PRECISION", self.config.PRECISION)).lower()
        if precision == "fp16":
            data = data.to(torch.float16)
        elif precision == "bf16":
            data = data.to(torch.bfloat16)
        else:
            data = data.to(torch.float32)

        # 3. 归一化（使用标准归一化）
        data = self.normalize(data, method="standard")
        return data

    def ensure_batch_dim(self, data: Tensor) -> Tensor:
        """
        确保张量包含 batch 维度。

        如果输入没有 batch 维度（ndim < 2），则执行 unsqueeze(0)。
        对于 1D 张量 (L,) → (1, L)。

        Args:
            data: 输入张量。

        Returns:
            Tensor: 至少为 2 维的张量。
        """
        if data.ndim < 2:
            logger.debug("添加 batch 维度：%s → %s", list(data.shape), [1] + list(data.shape))
            return data.unsqueeze(0)
        return data

    def normalize(self, data: Tensor, method: str = "standard") -> Tensor:
        """
        对张量进行归一化处理。

        Args:
            data: 输入张量。
            method: 归一化方法：
                    - "standard"：(x - mean) / (std + eps)，基于全局统计量
                    - "minmax"：(x - min) / (max - min + eps)，映射到 [0, 1]
                    - "none"：不做归一化，直接返回

        Returns:
            Tensor: 归一化后的张量。
        """
        eps = 1e-8
        if method == "standard":
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + eps)
        elif method == "minmax":
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val + eps)
        elif method == "none":
            return data
        else:
            logger.warning("未知的归一化方法：%s，使用 none", method)
            return data

    def pad_or_truncate(
        self,
        data: Tensor,
        target_length: int,
        dim: int = -1,
    ) -> Tensor:
        """
        将张量在指定维度上截断或零填充到目标长度。

        Args:
            data: 输入张量。
            target_length: 目标长度（正整数）。
            dim: 操作的维度，默认 -1（最后一维）。

        Returns:
            Tensor: 截断或填充后的张量，指定维度长度等于 target_length。
        """
        current_length = data.shape[dim]
        if current_length == target_length:
            return data
        elif current_length > target_length:
            # 截断
            slices = [slice(None)] * data.ndim
            slices[dim] = slice(0, target_length)
            return data[tuple(slices)]
        else:
            # 零填充
            pad_size = target_length - current_length
            pad_shape = list(data.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
            return torch.cat([data, padding], dim=dim)

    def to_device(self, data: Tensor, device: Optional[str] = None) -> Tensor:
        """
        将数据移到指定设备。

        Args:
            data: 输入张量。
            device: 目标设备字符串，如 "cpu"、"cuda"、"cuda:0"。
                    为 None 时不做迁移，直接返回原张量。

        Returns:
            Tensor: 已迁移到目标设备的张量。
        """
        if device is None:
            return data
        logger.debug("将数据迁移到设备：%s", device)
        return data.to(device)

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

        Returns:
            Tensor: 预处理后的图像张量，形状 (1, C, H, W)，dtype float32。

        Raises:
            FileNotFoundError: image 为路径字符串且文件不存在时抛出。
            InvalidInputError: target_size 超出允许范围时抛出。
        """
        import os

        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ImportError("图像预处理需要安装 Pillow：pip install Pillow") from exc

        min_img = int(self.config.get("MIN_IMAGE_SIZE", self.config.MIN_IMAGE_SIZE))
        max_img = int(self.config.get("MAX_IMAGE_SIZE", self.config.MAX_IMAGE_SIZE))
        h_target, w_target = target_size
        if not (min_img <= h_target <= max_img):
            raise InvalidInputError(
                expected=f"target height in [{min_img}, {max_img}]",
                actual=str(h_target),
            )
        if not (min_img <= w_target <= max_img):
            raise InvalidInputError(
                expected=f"target width in [{min_img}, {max_img}]",
                actual=str(w_target),
            )

        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"图像文件不存在：{image}")
            pil_image = Image.open(image).convert("RGB")
        else:
            # numpy 数组
            if image.ndim == 2:
                # 灰度图 → RGB
                image = np.stack([image, image, image], axis=-1)
            pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")

        # 调整尺寸
        pil_image = pil_image.resize((w_target, h_target), Image.BILINEAR)  # type: ignore[attr-defined]

        # 转换为张量 [0, 1]
        img_array = np.array(pil_image, dtype=np.float32) / 255.0  # (H, W, C)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)

        # ImageNet 归一化
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # 添加 batch 维度
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        logger.debug("图像预处理完成，输出形状：%s", list(tensor.shape))
        return tensor

    def preprocess_sequence(self, sequence: Tensor) -> Tensor:
        """
        序列预处理（ECG / NLP 等 1D 时序数据）。

        处理流程：
        1. 若序列长度超过 Config.MAX_SEQUENCE_LENGTH，截断至上限。
        2. 转换 dtype 为 float32（若不是）。
        3. 保证输出至少为 2D（添加 batch 维度，若输入为 1D）。

        Args:
            sequence: 原始序列张量。支持形状：
                      - (L,)：单条序列，自动添加 batch 维度变为 (1, L)
                      - (B, L)：批次序列
                      - (B, C, L)：多通道批次序列（如 12 导联 ECG）

        Returns:
            Tensor: 预处理后的序列张量，dtype float32。
                    形状与输入一致（保持 batch 维度）。
        """
        # 确保 batch 维度
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)  # (1, L)

        # 截断过长序列（在最后一个维度上截断）
        max_len = self.max_sequence_length
        if sequence.shape[-1] > max_len:
            logger.warning(
                "序列长度 %d 超过上限 %d，执行截断",
                sequence.shape[-1],
                max_len,
            )
            sequence = self.pad_or_truncate(sequence, max_len, dim=-1)

        # dtype 转换为 float32
        if sequence.dtype != torch.float32:
            sequence = sequence.to(torch.float32)

        logger.debug("序列预处理完成，输出形状：%s", list(sequence.shape))
        return sequence

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

        Raises:
            ImportError: 图像架构下 torchvision 未安装时抛出。
        """
        if architecture in (ModelArchitecture.VIT, ModelArchitecture.SWIN):
            try:
                import torchvision.transforms as T  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "图像预处理需要安装 torchvision：pip install torchvision"
                ) from exc

            target_h, target_w = self.default_target_size
            transform = T.Compose([
                T.Resize((target_h, target_w)),
                T.CenterCrop(min(target_h, target_w)),
                T.ToTensor(),
                T.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
            ])
            logger.info(
                "为架构 %s 创建图像预处理 transform，目标尺寸=%s",
                architecture.name,
                self.default_target_size,
            )
            return transform
        else:
            # 序列架构：恒等变换（数据已是 Tensor）
            logger.info("为架构 %s 创建序列预处理 transform（恒等变换）", architecture.name)
            return lambda x: x.float() if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float32)
