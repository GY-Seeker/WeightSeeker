"""
输入预处理器模块
"""

from typing import Union, Tuple, Callable
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray as NDArray

from ..core.types import ModelInfo, ModelArchitecture


class Preprocessor:
    """输入预处理器：针对不同架构的预处理"""
    
    def __init__(self, model_info: ModelInfo) -> None:
        """
        初始化预处理器
        
        Args:
            model_info: 模型信息
        """
        pass
    
    def preprocess_image(self, image: Union[str, NDArray], target_size: Tuple[int, int]) -> Tensor:
        """
        图像预处理
        
        Args:
            image: 图像路径或numpy数组
            target_size: 目标尺寸 (H, W)
            
        Returns:
            Tensor: 预处理后的图像张量 (1, C, H, W)
        """
        pass
    
    def preprocess_sequence(self, sequence: Tensor) -> Tensor:
        """
        序列预处理
        
        Args:
            sequence: 原始序列张量
            
        Returns:
            Tensor: 预处理后的序列张量
        """
        pass
    
    def get_default_transform(self, architecture: ModelArchitecture) -> Callable:
        """
        获取默认预处理变换
        
        Args:
            architecture: 模型架构类型
            
        Returns:
            Callable: 预处理变换函数
        """
        pass
