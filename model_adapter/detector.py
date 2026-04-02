"""Architecture detection module.

本模块提供模型架构自动识别功能，支持ViT、Swin Transformer、
标准Transformer和MoE-Transformer四种架构的检测与参数提取。
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.types import ModelInfo, ModelArchitecture
from ..core.exceptions import ArchitectureNotSupportedError


class ArchitectureDetector:
    """架构探测器：自动识别模型架构并提取参数。
    
    通过分析模型的nn.Module子模块结构，自动识别四种支持的架构类型：
    - ViT (Vision Transformer)
    - Swin Transformer
    - 标准 Transformer
    - MoE-Transformer
    
    Attributes:
        _detected_modules: 缓存检测到的关键模块
    """

    def __init__(self) -> None:
        """初始化探测器"""
        self._detected_modules: Dict[str, Any] = {}

    def detect(self, model: nn.Module) -> ModelInfo:
        """
        自动识别模型架构

        依次尝试检测ViT、Swin、标准Transformer和MoE架构，
        返回第一个匹配的架构信息。如果都不匹配，抛出异常。

        Args:
            model: PyTorch模型实例

        Returns:
            ModelInfo: 模型信息对象，包含架构类型、层数、头数等参数

        Raises:
            ArchitectureNotSupportedError: 当架构不支持时
        """
        # 依次尝试各种架构检测
        detectors = [
            (self._detect_vit, ModelArchitecture.VIT),
            (self._detect_swin, ModelArchitecture.SWIN),
            (self._detect_moe, ModelArchitecture.MOE_TRANSFORMER),
            (self._detect_transformer, ModelArchitecture.TRANSFORMER),
        ]
        
        for detect_method, arch_type in detectors:
            try:
                model_info = detect_method(model)
                if model_info is not None:
                    return model_info
            except Exception:
                continue
        
        # 所有检测都失败
        raise ArchitectureNotSupportedError(
            architecture=type(model).__name__,
            message=f"无法识别的模型架构: {type(model).__name__}"
        )

    def _detect_vit(self, model: nn.Module) -> Optional[ModelInfo]:
        """
        识别ViT架构
        
        检测特征：
        - 存在patch_embed或PatchEmbed模块
        - 存在cls_token参数
        - 存在标准的encoder blocks（通常命名为blocks或encoder.layers）
        
        支持来源：timm的VisionTransformer、HuggingFace的ViTModel
        
        Args:
            model: PyTorch模型实例
            
        Returns:
            ModelInfo: 如果识别成功返回模型信息，否则返回None
        """
        has_patch_embed = False
        has_cls_token = False
        num_blocks = 0
        num_heads = 0
        patch_size = 16
        hidden_dim = 0
        
        for name, module in model.named_modules():
            module_name = type(module).__name__
            
            # 检测PatchEmbed
            if "PatchEmbed" in module_name or name == "patch_embed":
                has_patch_embed = True
                # 提取patch_size
                if hasattr(module, "patch_size"):
                    patch_size = module.patch_size
                elif hasattr(module, "proj") and hasattr(module.proj, "kernel_size"):
                    # 从卷积核大小推断
                    kernel = module.proj.kernel_size
                    if isinstance(kernel, tuple):
                        patch_size = kernel[0]
                    else:
                        patch_size = kernel
            
            # 检测cls_token
            if name == "cls_token" or "cls_token" in name:
                has_cls_token = True
            
            # 检测Transformer Block
            if any(keyword in module_name for keyword in ["Block", "Layer", "EncoderBlock"]):
                if hasattr(module, "attn") or hasattr(module, "attention"):
                    num_blocks += 1
                    
                    # 提取头数
                    attn_module = getattr(module, "attn", getattr(module, "attention", None))
                    if attn_module is not None:
                        if hasattr(attn_module, "num_heads"):
                            num_heads = attn_module.num_heads
                        elif hasattr(attn_module, "heads"):
                            num_heads = attn_module.heads
            
            # 提取hidden_dim
            if hasattr(module, "embed_dim"):
                hidden_dim = module.embed_dim
            elif hasattr(module, "dim"):
                hidden_dim = module.dim
        
        # 检查模型级属性
        if not hidden_dim:
            hidden_dim = getattr(model, "embed_dim", getattr(model, "dim", 768))
        if not num_heads:
            num_heads = getattr(model, "num_heads", 12)
        if not num_blocks:
            # 尝试从blocks属性获取
            if hasattr(model, "blocks"):
                num_blocks = len(model.blocks)
            elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                num_blocks = len(model.encoder.layer)
        
        # 判断是否满足ViT特征
        if has_patch_embed or has_cls_token or num_blocks > 0:
            return ModelInfo(
                architecture=ModelArchitecture.VIT,
                num_layers=num_blocks if num_blocks > 0 else 12,
                num_heads=num_heads if num_heads > 0 else 12,
                patch_size=patch_size,
                hidden_dim=hidden_dim if hidden_dim > 0 else 768,
            )
        
        return None

    def _detect_swin(self, model: nn.Module) -> Optional[ModelInfo]:
        """
        识别Swin Transformer架构
        
        检测特征：
        - 存在WindowAttention或SwinTransformerBlock
        - 多stage结构（layers或stages）
        - 窗口相关参数
        
        Args:
            model: PyTorch模型实例
            
        Returns:
            ModelInfo: 如果识别成功返回模型信息，否则返回None
        """
        has_window_attention = False
        num_stages = 0
        num_heads_total = 0
        window_size = 7
        patch_size = 4
        hidden_dim = 0
        
        for name, module in model.named_modules():
            module_name = type(module).__name__
            
            # 检测窗口注意力
            if any(keyword in module_name for keyword in ["WindowAttention", "SwinTransformerBlock"]):
                has_window_attention = True
                
                # 提取窗口大小
                if hasattr(module, "window_size"):
                    window_size = module.window_size
                elif hasattr(module, "attn") and hasattr(module.attn, "window_size"):
                    window_size = module.attn.window_size
            
            # 检测stage
            if any(keyword in name for keyword in ["layers", "stages", "stages."]) and "." not in name.split(".")[-1]:
                if hasattr(module, "__len__"):
                    num_stages = max(num_stages, len(module))
            
            # 提取头数
            if hasattr(module, "num_heads"):
                num_heads_total = max(num_heads_total, module.num_heads)
            
            # 提取hidden_dim
            if hasattr(module, "dim"):
                hidden_dim = max(hidden_dim, module.dim)
            elif hasattr(module, "embed_dim"):
                hidden_dim = max(hidden_dim, module.embed_dim)
        
        # 检查模型级属性
        if not num_stages:
            if hasattr(model, "layers"):
                num_stages = len(model.layers)
            elif hasattr(model, "stages"):
                num_stages = len(model.stages)
        
        if not num_heads_total:
            num_heads_total = getattr(model, "num_heads", [3, 6, 12, 24])[0] if hasattr(model, "num_heads") else 3
        
        if not hidden_dim:
            hidden_dim = getattr(model, "embed_dim", 96)
        
        if hasattr(model, "window_size"):
            window_size = model.window_size
        
        # 计算总层数（每个stage有多个block）
        total_layers = 0
        if hasattr(model, "layers"):
            for layer in model.layers:
                if hasattr(layer, "blocks"):
                    total_layers += len(layer.blocks)
                elif hasattr(layer, "__len__"):
                    total_layers += len(layer)
        
        if not total_layers:
            total_layers = num_stages * 2  # 默认每个stage 2层
        
        # 判断是否满足Swin特征
        if has_window_attention or num_stages > 0:
            return ModelInfo(
                architecture=ModelArchitecture.SWIN,
                num_layers=total_layers if total_layers > 0 else 4,
                num_heads=num_heads_total if num_heads_total > 0 else 3,
                patch_size=patch_size,
                hidden_dim=hidden_dim if hidden_dim > 0 else 96,
                window_size=window_size,
            )
        
        return None

    def _detect_transformer(self, model: nn.Module) -> Optional[ModelInfo]:
        """
        识别标准Transformer架构
        
        检测特征：
        - 存在TransformerEncoder/TransformerDecoder
        - 存在MultiheadAttention模块
        - 标准的encoder/decoder层结构
        
        Args:
            model: PyTorch模型实例
            
        Returns:
            ModelInfo: 如果识别成功返回模型信息，否则返回None
        """
        has_transformer_encoder = False
        has_multihead_attn = False
        num_layers = 0
        num_heads = 0
        hidden_dim = 0
        
        for name, module in model.named_modules():
            module_name = type(module).__name__
            
            # 检测TransformerEncoder
            if "TransformerEncoder" in module_name:
                has_transformer_encoder = True
                if hasattr(module, "layers"):
                    num_layers = len(module.layers)
            
            # 检测TransformerDecoder
            if "TransformerDecoder" in module_name:
                if hasattr(module, "layers"):
                    num_layers = max(num_layers, len(module.layers))
            
            # 检测MultiheadAttention
            if "MultiheadAttention" in module_name:
                has_multihead_attn = True
                if hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                if hasattr(module, "embed_dim"):
                    hidden_dim = module.embed_dim
            
            # 检测EncoderLayer/DecoderLayer
            if any(keyword in module_name for keyword in ["EncoderLayer", "DecoderLayer", "TransformerBlock"]):
                num_layers += 1
        
        # 检查模型级属性
        if not num_layers:
            if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
                num_layers = len(model.encoder.layers)
            elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder"):
                if hasattr(model.transformer.encoder, "layers"):
                    num_layers = len(model.transformer.encoder.layers)
        
        if not num_heads:
            num_heads = getattr(model, "nhead", getattr(model, "num_heads", 8))
        
        if not hidden_dim:
            hidden_dim = getattr(model, "d_model", getattr(model, "hidden_dim", 512))
        
        # 判断是否满足标准Transformer特征
        if has_transformer_encoder or has_multihead_attn or num_layers > 0:
            return ModelInfo(
                architecture=ModelArchitecture.TRANSFORMER,
                num_layers=num_layers if num_layers > 0 else 6,
                num_heads=num_heads if num_heads > 0 else 8,
                patch_size=0,  # 标准Transformer不使用patch
                hidden_dim=hidden_dim if hidden_dim > 0 else 512,
            )
        
        return None

    def _detect_moe(self, model: nn.Module) -> Optional[ModelInfo]:
        """
        识别MoE-Transformer架构
        
        检测特征：
        - 在Transformer基础上存在MoE/Expert/Router/Gate相关模块
        - 专家路由机制
        
        Args:
            model: PyTorch模型实例
            
        Returns:
            ModelInfo: 如果识别成功返回模型信息，否则返回None
        """
        has_moe = False
        has_expert = False
        has_router = False
        num_experts = 0
        top_k = 2
        
        # 首先检测基础Transformer结构
        base_info = self._detect_transformer(model)
        if base_info is None:
            # 也尝试检测ViT基础
            base_info = self._detect_vit(model)
        
        for name, module in model.named_modules():
            module_name = type(module).__name__
            
            # 检测MoE相关模块
            if any(keyword in module_name for keyword in ["MoE", "Switch", "SparseMoE"]):
                has_moe = True
                if hasattr(module, "num_experts"):
                    num_experts = module.num_experts
                if hasattr(module, "top_k"):
                    top_k = module.top_k
            
            # 检测Expert模块
            if "Expert" in module_name or name.endswith(".experts"):
                has_expert = True
                if hasattr(module, "__len__"):
                    num_experts = max(num_experts, len(module))
            
            # 检测Router/Gate模块
            if any(keyword in module_name for keyword in ["Router", "Gate", "Routing"]):
                has_router = True
                if hasattr(module, "top_k"):
                    top_k = module.top_k
        
        # 检查模型级属性
        if not num_experts:
            num_experts = getattr(model, "num_experts", 8)
        if not top_k:
            top_k = getattr(model, "top_k", 2)
        
        # 判断是否满足MoE特征
        if has_moe or (has_expert and has_router):
            if base_info is not None:
                return ModelInfo(
                    architecture=ModelArchitecture.MOE_TRANSFORMER,
                    num_layers=base_info.num_layers,
                    num_heads=base_info.num_heads,
                    patch_size=base_info.patch_size,
                    hidden_dim=base_info.hidden_dim,
                    num_experts=num_experts,
                )
            else:
                return ModelInfo(
                    architecture=ModelArchitecture.MOE_TRANSFORMER,
                    num_layers=12,
                    num_heads=8,
                    patch_size=16,
                    hidden_dim=768,
                    num_experts=num_experts,
                )
        
        return None

    def extract_parameters(self, model: nn.Module, arch: ModelArchitecture) -> Dict[str, Any]:
        """
        根据已知架构类型提取详细参数
        
        Args:
            model: PyTorch模型实例
            arch: 已识别的模型架构类型
            
        Returns:
            Dict[str, Any]: 包含详细参数的字典，如层数、头数、维度等
        """
        params: Dict[str, Any] = {
            "architecture": arch.name,
            "model_class": type(model).__name__,
        }
        
        if arch == ModelArchitecture.VIT:
            params["patch_size"] = getattr(model, "patch_size", 16)
            params["embed_dim"] = getattr(model, "embed_dim", getattr(model, "dim", 768))
            params["num_heads"] = getattr(model, "num_heads", 12)
            params["depth"] = getattr(model, "depth", 12)
            if hasattr(model, "blocks"):
                params["num_layers"] = len(model.blocks)
        
        elif arch == ModelArchitecture.SWIN:
            params["window_size"] = getattr(model, "window_size", 7)
            params["embed_dim"] = getattr(model, "embed_dim", 96)
            params["num_heads"] = getattr(model, "num_heads", [3, 6, 12, 24])
            if hasattr(model, "layers"):
                params["num_stages"] = len(model.layers)
                total_blocks = 0
                for layer in model.layers:
                    if hasattr(layer, "blocks"):
                        total_blocks += len(layer.blocks)
                params["total_blocks"] = total_blocks
        
        elif arch == ModelArchitecture.TRANSFORMER:
            params["d_model"] = getattr(model, "d_model", 512)
            params["nhead"] = getattr(model, "nhead", 8)
            params["num_encoder_layers"] = getattr(model, "num_encoder_layers", 6)
            params["num_decoder_layers"] = getattr(model, "num_decoder_layers", 6)
        
        elif arch == ModelArchitecture.MOE_TRANSFORMER:
            params["num_experts"] = getattr(model, "num_experts", 8)
            params["top_k"] = getattr(model, "top_k", 2)
            params["aux_loss_coef"] = getattr(model, "aux_loss_coef", 0.01)
        
        return params
