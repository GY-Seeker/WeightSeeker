"""
数据管理器模块

根据 design.md §3.7.2，DataManager 负责：
- 加载图像数据集（支持自定义 transform）
- 加载序列数据集（ECG、NLP 等 1D 时序数据）
- 输入约束校验（尺寸 / 批次大小 / 序列长度）
- 创建标准 DataLoader
"""

from typing import Callable, Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..core.config import Config
from ..core.exceptions import InvalidInputError


class DataManager:
    """
    数据管理器：负责数据加载、输入校验与 DataLoader 创建。

    支持图像（2D，H×W）和序列（1D，长度 L）两种数据格式，
    可通过 validate_input() 在数据进入模型前统一校验约束。
    """

    def __init__(self, config: Config) -> None:
        """
        初始化数据管理器。

        Args:
            config: 全局配置对象（Config），用于读取：
                    - MAX_BATCH_SIZE：批次大小上限（默认 32）
                    - MAX_SEQUENCE_LENGTH：序列长度上限（默认 4096）
                    - MAX_IMAGE_SIZE / MIN_IMAGE_SIZE：图像尺寸范围
        """
        raise NotImplementedError("待实现")

    def load_image_dataset(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
    ) -> DataLoader:
        """
        加载图像数据集，返回 DataLoader。

        支持两种输入形式：
        1. 目录路径：使用 torchvision.datasets.ImageFolder 加载标准分类目录结构。
        2. 文件列表（.txt 文件，每行一个图像路径）：逐行读取并构建数据集。

        Args:
            data_path: 图像目录路径或包含图像路径的文本文件路径。
            transform: 可选的 torchvision transforms，用于图像预处理。
                       若为 None，默认使用 ToTensor()（不归一化）。

        Returns:
            DataLoader: 图像数据加载器，每个 batch 形状为 (B, C, H, W)。
        """
        raise NotImplementedError("待实现")

    def load_sequence_dataset(self, data_path: str) -> DataLoader:
        """
        加载序列数据集（ECG、NLP 时序等），返回 DataLoader。

        支持的数据格式：
        - .pt / .pth 文件：包含张量或字典张量
        - .npy / .npz 文件：numpy 数组格式
        - .csv 文件：每行为一条序列，逗号分隔

        Args:
            data_path: 序列数据文件路径或目录路径。
                       目录模式下会递归查找所有 .pt 文件。

        Returns:
            DataLoader: 序列数据加载器，每个 batch 形状为 (B, L) 或 (B, C, L)。
        """
        raise NotImplementedError("待实现")

    def validate_input(self, tensor: Tensor) -> bool:
        """
        校验输入张量是否满足约束条件。

        约束规则（来自 Config）：
        - 批次大小 B ≤ Config.MAX_BATCH_SIZE（默认 32）
        - 图像模式（4D 张量 B×C×H×W）：
            H, W ∈ [Config.MIN_IMAGE_SIZE, Config.MAX_IMAGE_SIZE]（默认 224 ~ 1024）
        - 序列模式（3D 张量 B×L×D 或 2D 张量 B×L）：
            L ≤ Config.MAX_SEQUENCE_LENGTH（默认 4096）

        Args:
            tensor: 输入张量。支持形状：
                    - (B, C, H, W)：图像输入
                    - (B, L, D)：序列输入（含 embedding 维度）
                    - (B, L)：序列输入（纯 token / 信号）

        Returns:
            bool: True 表示通过校验。

        Raises:
            InvalidInputError: 输入不满足任一约束时抛出，消息中说明违反的约束项。
        """
        raise NotImplementedError("待实现")

    def create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """
        根据数据集创建 DataLoader。

        使用 Config 中的默认参数（num_workers、pin_memory 等）。
        batch_size 不得超过 Config.MAX_BATCH_SIZE。

        Args:
            dataset: torch.utils.data.Dataset 实例。
            batch_size: 每批次样本数量，需 ≥ 1 且 ≤ Config.MAX_BATCH_SIZE。

        Returns:
            DataLoader: 配置好的数据加载器。

        Raises:
            InvalidInputError: batch_size 超出上限时抛出。
        """
        raise NotImplementedError("待实现")
