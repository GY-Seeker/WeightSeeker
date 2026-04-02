"""
模型加载器模块

根据 design.md §3.7.1，ModelLoader 支持从多种来源加载模型，
并提供 forward 签名检测能力，用于判断是否需要 InputAdapter 包装。
"""

import inspect
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.exceptions import InvalidInputError

logger = logging.getLogger(__name__)


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

        自动检测可用设备（当 device="auto" 时），保存 precision 配置。

        Args:
            device: 目标设备。可选值：
                    - "auto"：自动选择（有 GPU 则用 GPU，否则 CPU）
                    - "cpu"：强制使用 CPU
                    - "cuda"：使用默认 GPU
                    - "cuda:0"、"cuda:1" 等：指定 GPU 编号
            precision: 计算精度。可选值：
                       - "fp32"：单精度浮点（默认）
                       - "fp16"：半精度浮点（需要 GPU，至少 8 GB 显存）
                       - "bf16"：bfloat16 精度
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if precision not in ("fp32", "fp16", "bf16"):
            raise InvalidInputError(
                expected="precision in {'fp32', 'fp16', 'bf16'}",
                actual=precision,
            )
        self.precision = precision
        logger.info("ModelLoader 初始化：device=%s, precision=%s", self.device, self.precision)

    def load_model(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        weight_path: Optional[str] = None,
    ) -> nn.Module:
        """
        实例化模型类并可选加载权重。

        支持 state_dict 和完整 checkpoint 两种格式：
        - checkpoint 支持键 "model_state_dict"、"state_dict" 或直接为 state_dict。
        使用 strict=False 加载，并打印 missing/unexpected keys 的警告。

        Args:
            model_class: 模型类（可调用），通过 model_class(**model_kwargs) 实例化。
            model_kwargs: 传入模型类构造函数的关键字参数。
            weight_path: 可选的权重文件路径（.pth/.pt）。

        Returns:
            nn.Module: 已实例化并配置好设备/精度的模型（eval 模式）。

        Raises:
            FileNotFoundError: weight_path 不存在时抛出。
        """
        logger.info("实例化模型类：%s", model_class.__name__)
        model = model_class(**model_kwargs)

        if weight_path is not None:
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(f"权重文件不存在：{weight_path}")
            logger.info("加载权重文件：%s", weight_path)
            checkpoint = torch.load(weight_path, map_location="cpu")

            # 判断 checkpoint 格式
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # 假设 checkpoint 本身就是 state_dict
                    state_dict = checkpoint
            else:
                raise InvalidInputError(
                    expected="state_dict dict or checkpoint dict",
                    actual=type(checkpoint).__name__,
                )

            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logger.warning("缺少的权重键（%d 个）：%s", len(result.missing_keys), result.missing_keys[:10])
            if result.unexpected_keys:
                logger.warning("多余的权重键（%d 个）：%s", len(result.unexpected_keys), result.unexpected_keys[:10])

        model = self._setup_device(model)
        model = self._setup_precision(model)
        model.eval()
        logger.info("模型加载完成，已设为 eval 模式")
        return model

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
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在：{checkpoint_path}")

        logger.info("从检查点加载模型：%s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 情况1：checkpoint 本身就是完整的 nn.Module
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            logger.info("检测到完整模型对象，直接使用")
        elif isinstance(checkpoint, dict):
            # 判断是否为完整模型的 state_dict
            if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
                # 含 wrapper 的 checkpoint
                if model_class is None:
                    raise InvalidInputError(
                        expected="model_class when checkpoint contains state_dict",
                        actual="model_class=None",
                    )
                state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
                model = model_class()
                result = model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning("缺少的键（%d 个）：%s", len(result.missing_keys), result.missing_keys[:5])
                if result.unexpected_keys:
                    logger.warning("多余的键（%d 个）：%s", len(result.unexpected_keys), result.unexpected_keys[:5])
            else:
                # 纯 state_dict：字典的值都是 Tensor
                all_tensor = all(isinstance(v, torch.Tensor) for v in checkpoint.values())
                if all_tensor:
                    if model_class is None:
                        raise InvalidInputError(
                            expected="model_class when checkpoint is a pure state_dict",
                            actual="model_class=None",
                        )
                    model = model_class()
                    result = model.load_state_dict(checkpoint, strict=False)
                    if result.missing_keys:
                        logger.warning("缺少的键（%d 个）：%s", len(result.missing_keys), result.missing_keys[:5])
                    if result.unexpected_keys:
                        logger.warning("多余的键（%d 个）：%s", len(result.unexpected_keys), result.unexpected_keys[:5])
                else:
                    # 包含非 Tensor 值，可能是混合格式
                    if model_class is None:
                        raise InvalidInputError(
                            expected="model_class for mixed checkpoint format",
                            actual="model_class=None",
                        )
                    model = model_class()
                    result = model.load_state_dict(checkpoint, strict=False)
                    if result.missing_keys:
                        logger.warning("缺少的键（%d 个）：%s", len(result.missing_keys), result.missing_keys[:5])
        else:
            raise InvalidInputError(
                expected="nn.Module or dict checkpoint",
                actual=type(checkpoint).__name__,
            )

        model = self._setup_device(model)
        model = self._setup_precision(model)
        model.eval()
        logger.info("模型加载完成，已设为 eval 模式")
        return model

    def load_from_path(self, model_path: str) -> nn.Module:
        """
        直接加载完整模型对象（非 state_dict），移到设备并设置 eval。

        Args:
            model_path: 保存完整模型对象的文件路径（torch.save(model, path)）。

        Returns:
            nn.Module: 已配置好设备/精度的模型实例（eval 模式）。

        Raises:
            FileNotFoundError: model_path 不存在时抛出。
            InvalidInputError: 文件内容不是 nn.Module 时抛出。
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        logger.info("直接加载完整模型：%s", model_path)
        obj = torch.load(model_path, map_location="cpu")
        if not isinstance(obj, nn.Module):
            raise InvalidInputError(
                expected="nn.Module",
                actual=type(obj).__name__,
            )
        obj = self._setup_device(obj)
        obj = self._setup_precision(obj)
        obj.eval()
        return obj

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
        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "从 HuggingFace 加载模型需要安装 transformers 库：pip install transformers"
            ) from exc

        logger.info("从 HuggingFace 加载模型：%s", model_name)
        model = AutoModel.from_pretrained(model_name)
        model = self._setup_device(model)
        model = self._setup_precision(model)
        model.eval()
        return model

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
        try:
            import timm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "从 timm 加载模型需要安装 timm 库：pip install timm"
            ) from exc

        logger.info("从 timm 加载模型：%s", model_name)
        model = timm.create_model(model_name, pretrained=True)
        model = self._setup_device(model)
        model = self._setup_precision(model)
        model.eval()
        return model

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
                - "parameters": List[str]，所有参数名（与 param_names 相同，兼容字段）
                - "required_params": List[str]，必选参数（与 required 相同，兼容字段）

        示例返回（ECG 双输入模型）：
            {
                "param_names": ["ecg_signal", "meta_data"],
                "required": ["ecg_signal", "meta_data"],
                "has_defaults": {},
                "is_multi_input": True,
            }
        """
        # 先去除 DataParallel 等包装
        raw_model = self._unwrap_model(model)

        sig = inspect.signature(raw_model.forward)
        param_names: List[str] = []
        required: List[str] = []
        has_defaults: Dict[str, Any] = {}

        for name, param in sig.parameters.items():
            # 跳过 *args 和 **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            param_names.append(name)
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                has_defaults[name] = param.default

        is_multi_input = len(required) > 1

        result = {
            "param_names": param_names,
            "parameters": param_names,            # 兼容字段
            "required": required,
            "required_params": required,          # 兼容字段
            "has_defaults": has_defaults,
            "is_multi_input": is_multi_input,
        }
        logger.info(
            "forward 签名检测结果：必选参数=%s, is_multi_input=%s",
            required,
            is_multi_input,
        )
        return result

    def _setup_device(self, model: nn.Module) -> nn.Module:
        """
        将模型迁移到目标设备（CPU / GPU）。

        若 device="auto"，自动选择：CUDA 可用则用 cuda，否则用 cpu。

        Args:
            model: 待迁移的模型实例。

        Returns:
            nn.Module: 已迁移到目标设备的模型。
        """
        device = self.device
        logger.debug("将模型迁移到设备：%s", device)
        return model.to(device)

    def _setup_precision(self, model: nn.Module) -> nn.Module:
        """
        设置模型计算精度（FP32 / FP16 / BF16）。

        Args:
            model: 待配置精度的模型实例。

        Returns:
            nn.Module: 已配置精度的模型。

        Raises:
            InvalidInputError: precision="fp16" 但无可用 GPU 时抛出。
        """
        if self.precision == "fp16":
            if not torch.cuda.is_available():
                raise InvalidInputError(
                    expected="CUDA available when precision='fp16'",
                    actual="No CUDA device found",
                )
            logger.debug("设置模型精度为 fp16（half）")
            return model.half()
        elif self.precision == "bf16":
            logger.debug("设置模型精度为 bf16（bfloat16）")
            return model.to(torch.bfloat16)
        else:
            # fp32 默认
            logger.debug("设置模型精度为 fp32（float）")
            return model.float()

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
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            logger.debug("检测到 DataParallel 包装，提取原始模型 .module")
            return model.module
        return model
