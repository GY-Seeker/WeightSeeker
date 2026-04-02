"""
IO操作工具模块

提供文件系统操作、JSON 数据读写和图像读写等通用 IO 功能。
"""

import json
import os
from typing import Dict, Any
import numpy as np
from numpy import ndarray as NDArray
from PIL import Image


class IOUtils:
    """IO工具类
    
    提供静态工具方法用于目录管理、JSON 数据持久化和图像读写。
    所有方法均为 @staticmethod，无需实例化即可调用。
    """
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在
        
        使用 os.makedirs 递归创建目录，如果目录已存在则忽略。
        
        Args:
            path: 目录路径
            
        Example:
            >>> IOUtils.ensure_dir("./output/results")
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], path: str) -> None:
        """保存 JSON 文件
        
        使用 json.dump 将字典数据保存为 JSON 文件。
        自动确保父目录存在，使用 UTF-8 编码和 2 空格缩进。
        
        Args:
            data: 要保存的字典数据
            path: 文件保存路径
            
        Example:
            >>> data = {"name": "test", "value": 42}
            >>> IOUtils.save_json(data, "./output/config.json")
        """
        # 确保父目录存在
        parent_dir = os.path.dirname(path)
        if parent_dir:
            IOUtils.ensure_dir(parent_dir)
        
        # 保存 JSON 文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """加载 JSON 文件
        
        使用 json.load 从文件加载 JSON 数据。
        
        Args:
            path: JSON 文件路径
            
        Returns:
            Dict[str, Any]: 加载的字典数据
            
        Raises:
            FileNotFoundError: 当文件不存在时抛出
            
        Example:
            >>> data = IOUtils.load_json("./config.json")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_image(image: NDArray, path: str) -> None:
        """保存图像
        
        使用 PIL 保存图像文件。支持 uint8 和 float32 格式，
        float32 格式会自动转换为 0-255 范围的 uint8。
        自动确保父目录存在。
        
        Args:
            image: 图像数组，形状为 (H, W) 或 (H, W, C)
            path: 图像保存路径
            
        Example:
            >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            >>> IOUtils.save_image(image, "./output/result.png")
            >>> 
            >>> # float32 图像会自动转换
            >>> float_image = np.random.rand(224, 224, 3).astype(np.float32)
            >>> IOUtils.save_image(float_image, "./output/result.png")
        """
        # 确保父目录存在
        parent_dir = os.path.dirname(path)
        if parent_dir:
            IOUtils.ensure_dir(parent_dir)
        
        # 处理数据类型
        if image.dtype == np.float32 or image.dtype == np.float64:
            # 假设 float 图像范围是 [0, 1]，转换为 [0, 255]
            image = (image * 255).clip(0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # 其他类型转换为 uint8
            image = image.astype(np.uint8)
        
        # 创建 PIL Image 并保存
        pil_image = Image.fromarray(image)
        pil_image.save(path)
    
    @staticmethod
    def load_image(path: str) -> NDArray:
        """加载图像
        
        使用 PIL 加载图像文件并转换为 numpy 数组。
        返回的数组形状为 (H, W) 或 (H, W, C)，数据类型为 uint8。
        
        Args:
            path: 图像文件路径
            
        Returns:
            NDArray: 图像数组，形状为 (H, W) 或 (H, W, C)
            
        Raises:
            FileNotFoundError: 当文件不存在时抛出
            
        Example:
            >>> image = IOUtils.load_image("./input/photo.jpg")
            >>> print(image.shape)  # (H, W, 3)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # 使用 PIL 加载图像
        pil_image = Image.open(path)
        
        # 转换为 numpy 数组
        image_array = np.array(pil_image)
        
        return image_array
