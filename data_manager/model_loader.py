"""
模型加载器模块

根据 design.md §3.7.1，ModelLoader 支持从多种来源加载模型，
并提供 forward 签名检测能力，用于判断是否需要 InputAdapter 包装。
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class ModelLoader:
    """
    模型加载器：支持从多种来源加载模型。

    支持的加载方式：
    - 本地 .pth / .pt 检查点文件
    - HuggingFace 预训练模型（model_name）
    - timm 预训练模型（model_name）

    还提供 inspect_forward_signature()，用于检测 forward 函数参数签名，
    判断模型是否为多输入模型，从而决定是否需要 InputAdapter。
    """

    def __init__(self, device: str = "auto", precision: str = "fp32") -> None:
        """
        初始化模型加载器。

        Args:
            device: 目标设备。可选值：
                    - "auto"：自动选择（有 GPU 则用 GPU，否则 CPU）
                    - "cpu"：强制使用 CPU
                    - "cuda"：使用默认 GPU
                    - "cuda:0"、"cuda:1" 等：指定 GPU 编号
            precision: 计算精度。可选值：
                       - "fp32"：单精度浮点（默认）
                       - "fp16"：半精度浮点（需要 GPU，至少 8 GB 显存）
        """
        raise NotImplementedError("待实现")

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        model_class: Optional[type] = None,
    ) -> nn.Module:
        """
        从 .pth / .pt 检查点文件加载模型。

        支持两种情况：
        1. 检查点包含完整模型对象（torch.save(model, path)）：
           直接 torch.load 后返回。
        2. 检查点仅包含 state_dict（torch.save(model.state_dict(), path)）：
           需要传入 model_class 进行实例化后再加载权重。

        加载后自动调用 _setup_device() 和 _setup_precision() 完成设备及精度配置。

        Args:
            checkpoint_path: 检查点文件的本地路径（.pth 或 .pt）。
            model_class: 可选。当检查点仅包含 state_dict 时，用于实例化模型的类。
                         若为 None 且检查点为纯 state_dict，则抛出 InvalidInputError。

        Returns:
            nn.Module: 已加载权重并配置好设备/精度的模型实例（eval 模式）。

        Raises:
            FileNotFoundError: checkpoint_path 不存在时抛出。
            InvalidInputError: 检查点为纯 state_dict 但未提供 model_class 时抛出。
        """
        raise NotImplementedError("待实现")

    def load_from_huggingface(self, model_name: str) -> nn.Module:
        """
        从 HuggingFace Hub 加载预训练模型。

        内部调用 transformers.AutoModel.from_pretrained()。
        加载后自动调用 _setup_device() 和 _setup_precision()。

        Args:
            model_name: HuggingFace 模型名称，例如 "bert-base-uncased"。

        Returns:
            nn.Module: 已加载的预训练模型实例（eval 模式）。

        Raises:
            ImportError: 未安装 transformers 库时抛出。
        """
        raise NotImplementedError("待实现")

    def load_from_timm(self, model_name: str) -> nn.Module:
        """
        从 timm 库加载预训练图像模型。

        内部调用 timm.create_model(model_name, pretrained=True)。
        加载后自动调用 _setup_device() 和 _setup_precision()。

        Args:
            model_name: timm 模型名称，例如 "vit_base_patch16_224"。

        Returns:
            nn.Module: 已加载的预训练模型实例（eval 模式）。

        Raises:
            ImportError: 未安装 timm 库时抛出。
        """
        raise NotImplementedError("待实现")

    def inspect_forward_signature(self, model: nn.Module) -> Dict[str, Any]:
        """
        检测模型 forward 函数的参数签名。

        通过 inspect.signature(model.forward) 获取参数信息，
        返回结构化签名字典，供 InputAdapter.from_signature() 使用，
        判断模型是否为多输入模型。

        Args:
            model: 待检测的模型实例（可以是原始模型或 DataParallel 包装后的模型）。
                   若为 DataParallel，内部会先调用 _unwrap_model() 获取原始模型再检测。

        Returns:
            Dict[str, Any]: 参数签名字典，包含以下键：
                - "param_names": List[str]，所有参数名（不含 self）
                - "required": List[str]，无默认值的必选参数名列表
                - "has_defaults": Dict[str, Any]，有默认值的参数及其默认值
                - "is_multi_input": bool，是否有多个必选参数（True 表示需要 InputAdapter）

        示例返回（ECG 双输入模型）：
            {
                "param_names": ["ecg_signal", "meta_data"],
                "required": ["ecg_signal", "meta_data"],
                "has_defaults": {},
                "is_multi_input": True,
            }
        """
        raise NotImplementedError("待实现")

    def _setup_device(self, model: nn.Module) -> nn.Module:
        """
        将模型迁移到目标设备（CPU / GPU）。

        若 device="auto"，自动选择：CUDA 可用则用 cuda，否则用 cpu。

        Args:
            model: 待迁移的模型实例。

        Returns:
            nn.Module: 已迁移到目标设备的模型。
        """
        raise NotImplementedError("待实现")

    def _setup_precision(self, model: nn.Module) -> nn.Module:
        """
        设置模型计算精度（FP32 / FP16）。

        若 precision="fp16"，调用 model.half()；
        若 precision="fp32"，调用 model.float()。

        Args:
            model: 待配置精度的模型实例。

        Returns:
            nn.Module: 已配置精度的模型。

        Raises:
            InvalidInputError: precision="fp16" 但无可用 GPU 时抛出。
        """
        raise NotImplementedError("待实现")

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        处理 DataParallel / DistributedDataParallel 等包装器，获取原始模型。

        若 model 是 nn.DataParallel 或 nn.parallel.DistributedDataParallel，
        返回 model.module；否则直接返回 model。

        Args:
            model: 可能被包装的模型实例。

        Returns:
            nn.Module: 原始模型实例（未包装）。
        """
        raise NotImplementedError("待实现")
