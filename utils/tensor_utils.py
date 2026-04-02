"""
张量操作工具模块

提供张量形状处理、转换、安全计算等通用工具函数。
"""

from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numpy import ndarray as NDArray

from ..core.exceptions import InvalidInputError


class TensorUtils:
    """张量工具类
    
    提供静态工具方法用于张量形状处理、维度操作、类型转换和安全计算。
    所有方法均为 @staticmethod，无需实例化即可调用。
    """
    
    @staticmethod
    def ensure_same_shape(tensors: List[Tensor]) -> List[Tensor]:
        """确保多个张量形状一致
        
        检查所有输入张量的形状，如果不一致，尝试通过广播或插值对齐到最大形状。
        如果无法对齐，则抛出 InvalidInputError 异常。
        
        Args:
            tensors: 输入张量列表
            
        Returns:
            List[Tensor]: 形状对齐后的张量列表
            
        Raises:
            InvalidInputError: 当张量形状不一致且无法对齐时抛出
            
        Example:
            >>> t1 = torch.randn(2, 3, 4)
            >>> t2 = torch.randn(2, 3, 4)
            >>> aligned = TensorUtils.ensure_same_shape([t1, t2])
        """
        if not tensors:
            return []
        
        if len(tensors) == 1:
            return tensors
        
        # 获取所有张量的形状
        shapes = [t.shape for t in tensors]
        
        # 检查是否所有形状已经一致
        if all(s == shapes[0] for s in shapes):
            return tensors
        
        # 计算目标形状（各维度的最大值）
        max_dims = []
        for dim_idx in range(max(len(s) for s in shapes)):
            dim_sizes = []
            for s in shapes:
                if dim_idx < len(s):
                    dim_sizes.append(s[dim_idx])
            max_dims.append(max(dim_sizes) if dim_sizes else 1)
        
        target_shape = tuple(max_dims)
        
        # 尝试对齐所有张量到目标形状
        aligned_tensors = []
        for tensor in tensors:
            try:
                # 如果张量维度不足，先扩展维度
                while len(tensor.shape) < len(target_shape):
                    tensor = tensor.unsqueeze(0)
                
                # 如果形状已经一致，直接添加
                if tensor.shape == target_shape:
                    aligned_tensors.append(tensor)
                    continue
                
                # 尝试使用插值对齐空间维度（最后两个维度假设为 H, W）
                if len(tensor.shape) >= 2 and len(target_shape) >= 2:
                    # 检查除最后两个维度外的其他维度是否一致
                    other_dims_match = all(
                        tensor.shape[i] == target_shape[i] 
                        for i in range(len(target_shape) - 2)
                    )
                    
                    if other_dims_match:
                        # 使用双线性插值对齐空间维度
                        target_h, target_w = target_shape[-2], target_shape[-1]
                        aligned = F.interpolate(
                            tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1]),
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        # 恢复原始维度
                        aligned = aligned.reshape(*tensor.shape[:-2], target_h, target_w)
                        aligned_tensors.append(aligned)
                        continue
                
                # 如果无法对齐，抛出异常
                raise InvalidInputError(
                    expected=f"Tensors with compatible shapes for broadcasting/interpolation",
                    actual=f"Shapes: {shapes}"
                )
                
            except Exception as e:
                if isinstance(e, InvalidInputError):
                    raise
                raise InvalidInputError(
                    expected=f"Tensors compatible with shape {target_shape}",
                    actual=f"Tensor shape {tensor.shape}"
                ) from e
        
        return aligned_tensors
    
    @staticmethod
    def batch_average(tensor: Tensor, batch_dim: int = 0) -> Tensor:
        """对指定的 batch 维度取平均
        
        使用 torch.mean 对指定维度计算平均值，通常用于消除 batch 维度。
        
        Args:
            tensor: 输入张量
            batch_dim: batch 维度的索引，默认为 0
            
        Returns:
            Tensor: 对 batch 维度取平均后的张量
            
        Example:
            >>> tensor = torch.randn(4, 3, 224, 224)  # (B, C, H, W)
            >>> avg = TensorUtils.batch_average(tensor)  # (C, H, W)
        """
        return torch.mean(tensor, dim=batch_dim)
    
    @staticmethod
    def flatten_spatial(tensor: Tensor) -> Tensor:
        """将空间维度展平
        
        假设最后两个维度是空间维度（H, W），将其展平为一维。
        如果张量维度小于 2，直接返回原张量。
        
        Args:
            tensor: 输入张量，形状为 (..., H, W)
            
        Returns:
            Tensor: 展平后的张量，形状为 (..., H*W)
            
        Example:
            >>> tensor = torch.randn(4, 3, 16, 16)  # (B, C, H, W)
            >>> flat = TensorUtils.flatten_spatial(tensor)  # (B, C, 256)
        """
        if tensor.dim() < 2:
            return tensor
        
        # 展平最后两个维度
        return tensor.reshape(*tensor.shape[:-2], -1)
    
    @staticmethod
    def to_numpy(tensor: Tensor) -> NDArray:
        """安全地将张量转换为 numpy 数组
        
        处理 requires_grad 的情况：先 detach，再移到 CPU，最后转为 numpy。
        确保不会破坏计算图，同时能够安全转换。
        
        Args:
            tensor: 输入 PyTorch 张量
            
        Returns:
            NDArray: 转换后的 NumPy 数组
            
        Example:
            >>> tensor = torch.randn(3, 4, requires_grad=True)
            >>> arr = TensorUtils.to_numpy(tensor)  # 安全转换
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        return tensor.numpy()
    
    @staticmethod
    def safe_divide(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        """安全除法（避免除零）
        
        计算 a / (b + eps)，通过添加极小值 epsilon 避免除零错误。
        
        Args:
            a: 被除数张量
            b: 除数张量
            eps: 极小值，用于避免除零，默认为 1e-8
            
        Returns:
            Tensor: 除法结果 a / (b + eps)
            
        Example:
            >>> a = torch.tensor([1.0, 2.0, 3.0])
            >>> b = torch.tensor([0.0, 1.0, 2.0])
            >>> result = TensorUtils.safe_divide(a, b)  # [1e8, 2.0, 1.5]
        """
        return a / (b + eps)
