"""
内存管理工具模块

提供 GPU 显存监控、缓存清理、批大小建议和内存卸载等功能。
"""

import gc
from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class MemoryManager:
    """内存管理器
    
    用于监控和管理 GPU 显存使用情况，提供缓存清理、批大小建议
    和张量卸载等功能，帮助优化内存使用。
    """
    
    def __init__(self, max_memory_gb: float = 8.0) -> None:
        """初始化内存管理器
        
        Args:
            max_memory_gb: 最大内存限制（GB），默认为 8.0 GB
            
        Example:
            >>> manager = MemoryManager(max_memory_gb=16.0)
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    def check_memory_usage(self) -> float:
        """检查当前内存使用量（GB）
        
        优先返回 GPU 显存使用量，如果 CUDA 不可用则返回 CPU 内存使用量。
        使用 try/except 处理 CUDA 不可用的情况。
        
        Returns:
            float: 当前内存使用量（GB）
            
        Example:
            >>> manager = MemoryManager()
            >>> usage = manager.check_memory_usage()  # 例如: 2.5 (GB)
        """
        try:
            # 尝试获取 GPU 显存使用量
            if torch.cuda.is_available():
                # 获取当前设备
                device = torch.cuda.current_device()
                # 获取已分配的显存（字节）
                allocated_bytes = torch.cuda.memory_allocated(device)
                # 转换为 GB
                allocated_gb = allocated_bytes / (1024 * 1024 * 1024)
                return allocated_gb
        except Exception:
            # CUDA 相关错误，回退到 CPU 内存
            pass
        
        # 如果没有 GPU 或 CUDA 出错，尝试使用 psutil 获取 CPU 内存
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            # 转换为 GB
            memory_gb = memory_info.rss / (1024 * 1024 * 1024)
            return memory_gb
        except ImportError:
            # psutil 不可用，返回 0
            return 0.0
        except Exception:
            # 其他错误，返回 0
            return 0.0
    
    def clear_cache(self) -> None:
        """清理 PyTorch 缓存和 Python 垃圾
        
        调用 torch.cuda.empty_cache() 清理 GPU 缓存，
        并调用 gc.collect() 清理 Python 垃圾对象。
        
        Example:
            >>> manager = MemoryManager()
            >>> manager.clear_cache()  # 清理缓存
        """
        # 清理 GPU 缓存
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        # 清理 Python 垃圾
        gc.collect()
    
    def suggest_batch_size(self, model: nn.Module, sample_input: Tensor) -> int:
        """估算合适的 batch_size
        
        基于单个样本的显存占用和可用显存来计算建议的 batch_size。
        返回的 batch_size 范围为 [1, 32]。
        
        Args:
            model: PyTorch 模型实例
            sample_input: 单个样本输入张量
            
        Returns:
            int: 建议的 batch_size（最小为 1，最大为 32）
            
        Example:
            >>> manager = MemoryManager()
            >>> model = nn.Linear(10, 10)
            >>> sample = torch.randn(1, 10)
            >>> batch_size = manager.suggest_batch_size(model, sample)
        """
        # 清理缓存以获得准确的内存测量
        self.clear_cache()
        
        # 记录初始内存使用
        initial_memory = self.check_memory_usage()
        
        try:
            # 将模型和输入移到 GPU（如果可用）
            device = next(model.parameters()).device
            sample_input = sample_input.to(device)
            
            # 前向传播一次以测量内存占用
            with torch.no_grad():
                _ = model(sample_input)
            
            # 计算单个样本的显存占用
            final_memory = self.check_memory_usage()
            single_sample_memory = final_memory - initial_memory
            
            # 如果无法测量（可能是 CPU 模式），使用默认值
            if single_sample_memory <= 0:
                single_sample_memory = 0.5  # 假设 500MB
            
            # 计算可用显存
            available_memory = self.max_memory_gb - final_memory
            
            # 计算建议的 batch_size（留 20% 的安全余量）
            suggested = int((available_memory * 0.8) / single_sample_memory)
            
            # 限制范围 [1, 32]
            suggested = max(1, min(suggested, 32))
            
            return suggested
            
        except Exception:
            # 出错时返回保守的默认值
            return 1
    
    def offload_to_cpu(self, tensors: List[Tensor]) -> List[Tensor]:
        """将张量从 GPU 移到 CPU
        
        将列表中的张量从 GPU 显存移动到 CPU 内存，返回移动后的张量列表。
        已经在 CPU 上的张量保持不变。
        
        Args:
            tensors: 输入张量列表
            
        Returns:
            List[Tensor]: 移动到 CPU 后的张量列表
            
        Example:
            >>> manager = MemoryManager()
            >>> gpu_tensors = [torch.randn(10).cuda(), torch.randn(10).cuda()]
            >>> cpu_tensors = manager.offload_to_cpu(gpu_tensors)
        """
        cpu_tensors = []
        for tensor in tensors:
            if tensor.is_cuda:
                cpu_tensors.append(tensor.cpu())
            else:
                cpu_tensors.append(tensor)
        return cpu_tensors
