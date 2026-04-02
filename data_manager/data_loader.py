"""
数据加载器模块
"""

from typing import Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..core.config import Config


class DataManager:
    """数据管理器：负责数据加载和预处理"""
    
    def __init__(self, config: Config) -> None:
        """
        初始化数据管理器
        
        Args:
            config: 全局配置对象
        """
        pass
    
    def load_image_dataset(self, data_path: str, transform: Optional[Callable] = None) -> DataLoader:
        """
        加载图像数据集
        
        Args:
            data_path: 数据目录路径或图像文件列表
            transform: 预处理变换（可选）
            
        Returns:
            DataLoader: 图像数据加载器
        """
        pass
    
    def load_sequence_dataset(self, data_path: str) -> DataLoader:
        """
        加载序列数据集（文本/时序）
        
        Args:
            data_path: 序列数据文件路径
            
        Returns:
            DataLoader: 序列数据加载器
        """
        pass
    
    def validate_input(self, tensor: Tensor) -> bool:
        """
        校验输入约束
        
        约束条件：
        - 图像尺寸 H,W ∈ [224, 1024]
        - 批次大小 B ≤ 32
        - 序列长度 L ≤ 4096
        
        Args:
            tensor: 输入张量 (B, C, H, W) 或 (B, L, D)
            
        Returns:
            bool: 是否通过校验
            
        Raises:
            InvalidInputError: 输入不符合约束时抛出
        """
        pass
    
    def create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """
        创建DataLoader
        
        Args:
            dataset: 数据集实例
            batch_size: 批次大小
            
        Returns:
            DataLoader: 数据加载器
        """
        pass
