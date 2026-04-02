"""融合策略实现"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor


class FusionStrategy(ABC):
    """融合策略抽象基类"""
    
    @abstractmethod
    def fuse(self, attention: Tensor, gradient: Tensor, **kwargs) -> Tensor:
        """执行融合"""
        pass


class ProductFusion(FusionStrategy):
    """乘积融合策略"""
    
    def fuse(self, attention: Tensor, gradient: Tensor, **kwargs) -> Tensor:
        """
        乘积融合：重要性 = 注意力 × 梯度
        
        仅当两者均高时输出高权重
        """
        pass


class WeightedSumFusion(FusionStrategy):
    """加权求和融合策略"""
    
    def __init__(self, alpha: float = 0.5) -> None:
        """
        初始化
        
        Args:
            alpha: 注意力权重，(1-alpha)为梯度权重
        """
        pass
    
    def fuse(self, attention: Tensor, gradient: Tensor, **kwargs) -> Tensor:
        """
        加权求和：重要性 = α×注意力 + (1-α)×梯度
        """
        pass


class AttentionMaskFusion(FusionStrategy):
    """注意力掩码筛选融合策略"""
    
    def __init__(self, threshold: float = 0.3) -> None:
        """
        初始化
        
        Args:
            threshold: 注意力掩码阈值
        """
        pass
    
    def fuse(self, attention: Tensor, gradient: Tensor, **kwargs) -> Tensor:
        """
        注意力掩码筛选：先对注意力做阈值掩码，再与梯度相乘
        """
        pass
