"""融合编排器"""

from typing import Optional
import torch
from torch import Tensor

from ..core.types import FusionStrategyType
from .strategies import FusionStrategy


class FusionComposer:
    """融合编排器"""
    
    def __init__(self, strategy: Optional[FusionStrategy] = None) -> None:
        """
        初始化编排器
        
        Args:
            strategy: 融合策略实例
        """
        pass
    
    def set_strategy(self, strategy: FusionStrategy) -> None:
        """设置融合策略"""
        pass
    
    def compose(self, 
                attention_map: Tensor, 
                gradient_map: Tensor,
                strategy_type: Optional[FusionStrategyType] = None) -> Tensor:
        """
        执行融合编排
        
        Args:
            attention_map: 注意力图 [0,1]范围
            gradient_map: 梯度图 [0,1]范围
            strategy_type: 可选的策略类型，覆盖默认策略
            
        Returns:
            Tensor: 融合后的重要性分数图 [0,1]范围
        """
        pass
    
    def validate_inputs(self, attention: Tensor, gradient: Tensor) -> bool:
        """验证输入合法性"""
        pass
    
    def post_process(self, fused_map: Tensor) -> Tensor:
        """融合后处理（可选的平滑、裁剪等）"""
        pass
