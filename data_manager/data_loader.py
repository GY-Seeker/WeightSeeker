"""
数据管理器模块

根据 design.md §3.7.2，DataManager 负责：
- 加载图像数据集（支持自定义 transform）
- 加载序列数据集（ECG、NLP 等 1D 时序数据）
- 输入约束校验（尺寸 / 批次大小 / 序列长度）
- 创建标准 DataLoader
"""

import logging
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..core.config import Config
from ..core.exceptions import InvalidInputError

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器：负责数据加载、输入校验与 DataLoader 创建。

    支持图像（2D，H×W）和序列（1D，长度 L）两种数据格式，
    可通过 validate_input() 在数据进入模型前统一校验约束。
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        初始化数据管理器。

        Args:
            config: 全局配置对象（Config），用于读取：
                    - MAX_BATCH_SIZE：批次大小上限（默认 32）
                    - MAX_SEQUENCE_LENGTH：序列长度上限（默认 4096）
                    - MAX_IMAGE_SIZE / MIN_IMAGE_SIZE：图像尺寸范围
                    为 None 时自动创建默认 Config。
        """
        self.config = config if config is not None else Config()
        logger.info("DataManager 初始化完成，MAX_BATCH_SIZE=%s", self.config.MAX_BATCH_SIZE)

    def load_from_tensor(self, data: Tensor) -> DataLoader:
        """
        将单个张量包装为 TensorDataset + DataLoader。

        Args:
            data: 输入张量，形状任意（第 0 维为样本维度）。

        Returns:
            DataLoader: 包装好的数据加载器，batch_size 从 config 获取。
        """
        dataset = TensorDataset(data)
        batch_size = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        logger.info("从张量创建 DataLoader，张量形状=%s，batch_size=%d", list(data.shape), batch_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def load_from_dataset(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        将已有的 Dataset 包装为 DataLoader。

        Args:
            dataset: torch.utils.data.Dataset 实例。
            batch_size: 每批次样本数，None 则取 config.MAX_BATCH_SIZE。
            shuffle: 是否打乱顺序，默认 False。

        Returns:
            DataLoader: 配置好的数据加载器。
        """
        if batch_size is None:
            batch_size = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        max_bs = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        if batch_size > max_bs:
            raise InvalidInputError(
                expected=f"batch_size <= {max_bs}",
                actual=str(batch_size),
            )
        logger.info("从 Dataset 创建 DataLoader，batch_size=%d，shuffle=%s", batch_size, shuffle)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_from_numpy(self, data: np.ndarray) -> DataLoader:
        """
        将 numpy 数组转为 Tensor，再创建 DataLoader。

        Args:
            data: numpy 数组，任意形状，dtype 会被转换为 float32。

        Returns:
            DataLoader: 包装好的数据加载器。
        """
        tensor = torch.from_numpy(data.astype(np.float32))
        logger.info("从 numpy 数组创建 DataLoader，shape=%s", data.shape)
        return self.load_from_tensor(tensor)

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

        Raises:
            ImportError: torchvision 未安装时抛出。
            FileNotFoundError: data_path 不存在时抛出。
        """
        try:
            import torchvision  # type: ignore
            import torchvision.transforms as T  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "加载图像数据集需要安装 torchvision：pip install torchvision"
            ) from exc

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在：{data_path}")

        if transform is None:
            transform = T.ToTensor()

        if os.path.isdir(data_path):
            logger.info("使用 ImageFolder 加载图像目录：%s", data_path)
            dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
        else:
            # .txt 文件列表模式
            logger.info("从文件列表加载图像：%s", data_path)
            dataset = _TextFileImageDataset(data_path, transform=transform)

        batch_size = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def load_from_directory(
        self,
        dir_path: str,
        transform: Optional[Callable] = None,
    ) -> DataLoader:
        """
        使用 torchvision.datasets.ImageFolder 加载图片目录。

        Args:
            dir_path: 标准分类目录结构的根路径。
            transform: 可选的图像预处理 transform。

        Returns:
            DataLoader: 图像数据加载器。

        Raises:
            ImportError: torchvision 未安装时抛出。
        """
        return self.load_image_dataset(dir_path, transform=transform)

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

        Raises:
            FileNotFoundError: data_path 不存在时抛出。
            InvalidInputError: 文件格式不支持或无法解析时抛出。
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"序列数据路径不存在：{data_path}")

        batch_size = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))

        if os.path.isdir(data_path):
            # 递归查找所有 .pt 文件
            pt_files = []
            for root, _, files in os.walk(data_path):
                for f in files:
                    if f.endswith(".pt") or f.endswith(".pth"):
                        pt_files.append(os.path.join(root, f))
            if not pt_files:
                raise InvalidInputError(
                    expected="directory containing .pt files",
                    actual=f"no .pt files found in {data_path}",
                )
            tensors = []
            for pt_file in sorted(pt_files):
                obj = torch.load(pt_file, map_location="cpu")
                if isinstance(obj, dict):
                    # 取第一个张量值
                    for v in obj.values():
                        if isinstance(v, Tensor):
                            tensors.append(v)
                            break
                elif isinstance(obj, Tensor):
                    tensors.append(obj)
            if not tensors:
                raise InvalidInputError(
                    expected="tensors in .pt files",
                    actual="no tensors found",
                )
            data_tensor = torch.cat(tensors, dim=0)
            logger.info("从目录加载序列数据，共 %d 个样本", data_tensor.shape[0])
            dataset = TensorDataset(data_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        ext = os.path.splitext(data_path)[1].lower()
        if ext in (".pt", ".pth"):
            obj = torch.load(data_path, map_location="cpu")
            if isinstance(obj, Tensor):
                tensor = obj
            elif isinstance(obj, dict):
                # 取第一个张量值
                tensor = None
                for v in obj.values():
                    if isinstance(v, Tensor):
                        tensor = v
                        break
                if tensor is None:
                    raise InvalidInputError(
                        expected="dict containing Tensor values",
                        actual="no Tensor found in dict",
                    )
            else:
                raise InvalidInputError(
                    expected="Tensor or dict-of-Tensors in .pt file",
                    actual=type(obj).__name__,
                )
        elif ext == ".npy":
            arr = np.load(data_path)
            tensor = torch.from_numpy(arr.astype(np.float32))
        elif ext == ".npz":
            npz = np.load(data_path)
            # 取第一个数组
            key = list(npz.keys())[0]
            tensor = torch.from_numpy(npz[key].astype(np.float32))
        elif ext == ".csv":
            rows = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        vals = [float(x) for x in line.split(",")]
                        rows.append(vals)
            tensor = torch.tensor(rows, dtype=torch.float32)
        else:
            raise InvalidInputError(
                expected="file with extension .pt/.pth/.npy/.npz/.csv",
                actual=ext,
            )

        logger.info("加载序列数据：%s，形状=%s", data_path, list(tensor.shape))
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        if tensor.ndim < 2:
            raise InvalidInputError(
                expected="tensor with at least 2 dimensions",
                actual=f"tensor with {tensor.ndim} dimensions",
            )

        max_batch = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        batch_size = tensor.shape[0]
        if batch_size > max_batch:
            raise InvalidInputError(
                expected=f"batch_size <= {max_batch}",
                actual=f"batch_size = {batch_size}",
            )

        if tensor.ndim == 4:
            # 图像模式：(B, C, H, W)
            _, _, h, w = tensor.shape
            min_img = int(self.config.get("MIN_IMAGE_SIZE", self.config.MIN_IMAGE_SIZE))
            max_img = int(self.config.get("MAX_IMAGE_SIZE", self.config.MAX_IMAGE_SIZE))
            if not (min_img <= h <= max_img):
                raise InvalidInputError(
                    expected=f"image height H in [{min_img}, {max_img}]",
                    actual=f"H = {h}",
                )
            if not (min_img <= w <= max_img):
                raise InvalidInputError(
                    expected=f"image width W in [{min_img}, {max_img}]",
                    actual=f"W = {w}",
                )
        elif tensor.ndim in (2, 3):
            # 序列模式：(B, L) 或 (B, L, D)
            seq_len = tensor.shape[1]
            max_seq = int(self.config.get("MAX_SEQUENCE_LENGTH", self.config.MAX_SEQUENCE_LENGTH))
            if seq_len > max_seq:
                raise InvalidInputError(
                    expected=f"sequence_length L <= {max_seq}",
                    actual=f"L = {seq_len}",
                )

        return True

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
        max_bs = int(self.config.get("MAX_BATCH_SIZE", self.config.MAX_BATCH_SIZE))
        if batch_size < 1 or batch_size > max_bs:
            raise InvalidInputError(
                expected=f"1 <= batch_size <= {max_bs}",
                actual=str(batch_size),
            )
        logger.info("创建 DataLoader，batch_size=%d", batch_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class _TextFileImageDataset(Dataset):
    """
    从文本文件列表加载图像的内部辅助数据集类。

    文本文件格式：每行一个图像路径（绝对路径或相对路径）。
    """

    def __init__(self, txt_path: str, transform: Optional[Callable] = None) -> None:
        """
        初始化文本文件图像数据集。

        Args:
            txt_path: 包含图像路径列表的文本文件路径。
            transform: 可选的图像预处理 transform。
        """
        self.transform = transform
        self.image_paths = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.image_paths.append(line)
        logger.info("从文本文件加载图像列表，共 %d 张", len(self.image_paths))

    def __len__(self) -> int:
        """返回数据集样本总数。"""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        获取指定索引的图像。

        Args:
            idx: 样本索引。

        Returns:
            Tensor: 预处理后的图像张量。
        """
        from PIL import Image  # type: ignore

        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
