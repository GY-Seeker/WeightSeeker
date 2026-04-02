"""Architecture detection module.

本模块提供模型架构自动识别功能，支持ViT、Swin Transformer、
标准Transformer和MoE-Transformer四种架构的检测与参数提取。
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.types import ModelInfo, ModelArchitecture, DetectionResult
from ..core.exceptions import ArchitectureNotSupportedError


class ArchitectureDetector:
    """架构探测器：自动识别模型架构并提取参数。
    
    通过分析模型的nn.Module子模块结构，自动识别四种支持的架构类型：
    - ViT (Vision Transformer)
    - Swin Transformer
    - 标准 Transformer
    - MoE-Transformer
    
    内置置信度机制：当自动探测对多模态/混合架构置信度较低时，
    自动打印警告并建议用户通过 override 参数手动确认。
    
    Attributes:
        CONFIDENCE_WARNING_THRESHOLD: 置信度警告阈值，低于此值时自动输出警告
        _detected_modules: 缓存检测到的关键模块
    """

    CONFIDENCE_WARNING_THRESHOLD: float = 0.6  # 低于此值弹出警告

    def __init__(self) -> None:
        """初始化探测器"""
        self._detected_modules: Dict[str, Any] = {}

    def detect(self, model: nn.Module, override: Optional[Dict[str, Any]] = None) -> ModelInfo:
        """
        自动识别模型架构

        依次尝试检测ViT、Swin、标准Transformer和MoE架构，
        返回第一个匹配的架构信息。如果都不匹配，抛出异常。
        内置置信度计算：当探测置信度 < CONFIDENCE_WARNING_THRESHOLD 时自动输出警告。

        Args:
            model: PyTorch模型实例
            override: 手动覆盖字典（可选）。支持键：
                - "architecture": str 或 ModelArchitecture 枚举值
                - "num_layers": int
                - "num_heads": int
                - "patch_size": int
                - "hidden_dim": int
                - "window_size": int（Swin特有）
                - "num_experts": int（MoE特有）

        Returns:
            ModelInfo: 模型信息对象。如果提供了 override，使用覆盖后的值。

        Raises:
            ArchitectureNotSupportedError: 当架构不支持时
        """
        # 先进行带置信度的探测
        detection_result = self.detect_with_confidence(model)
        
        # 置信度警告
        if detection_result.confidence < self.CONFIDENCE_WARNING_THRESHOLD:
            for warning in detection_result.warnings:
                print(f"[ArchitectureDetector WARNING] {warning}")
        
        model_info = detection_result.model_info
        
        # 应用 override 覆盖
        if override:
            model_info = self._apply_override(model_info, override)
        
        return model_info

    def detect_with_confidence(self, model: nn.Module) -> DetectionResult:
        """
        架构探测并返回包含置信度的结果。
        
        相比 detect() ，该方法返回包含置信度和警告信息的完整探测结果，
        适合需要对探测质量进行评估的场景。
        
        Args:
            model: PyTorch模型实例
        
        Returns:
            DetectionResult: 包含 model_info、confidence、warnings 的探测结果。
        
        Raises:
            ArchitectureNotSupportedError: 当架构不支持时
        """
        warnings: List[str] = []
        detected_arch: Optional[ModelArchitecture] = None
        model_info: Optional[ModelInfo] = None
        
        # 尝试每种架构检测，记录所有匹配的架构
        matched_archs: List[tuple] = []
        detectors = [
            (self._detect_vit, ModelArchitecture.VIT),
            (self._detect_swin, ModelArchitecture.SWIN),
            (self._detect_moe, ModelArchitecture.MOE_TRANSFORMER),
            (self._detect_transformer, ModelArchitecture.TRANSFORMER),
        ]
        
        for detect_method, arch_type in detectors:
            try:
                info = detect_method(model)
                if info is not None:
                    matched_archs.append((arch_type, info))
            except Exception:
                continue
        
        if not matched_archs:
            raise ArchitectureNotSupportedError(
                architecture=type(model).__name__,
                message=f"无法识别的模型架构: {type(model).__name__}"
            )
        
        # 取第一个匹配结果作为主结果
        detected_arch, model_info = matched_archs[0]
        
        # 计算置信度
        confidence = self._compute_confidence(model, detected_arch, matched_archs)
        
        # 为低置信度生成警告
        if confidence < self.CONFIDENCE_WARNING_THRESHOLD:
            warnings.append(
                f"架构探测置信度较低 ({confidence:.2f})，"
                f"探测到的架构为 {detected_arch.name}。"
                f"建议使用 override 参数手动确认关键架构参数。"
            )
            if len(matched_archs) > 1:
                arch_names = [a.name for a, _ in matched_archs]
                warnings.append(
                    f"检测到多个匹配架构: {arch_names}，"
                    f"此现象常出现在多模态/混合架构模型中。"
                )
        
        return DetectionResult(
            model_info=model_info,
            confidence=confidence,
            warnings=warnings,
        )

    def _apply_override(self, model_info: ModelInfo, override: Dict[str, Any]) -> ModelInfo:
        """
        将 override 字典中的值覆盖到 ModelInfo 对象中。
        
        Args:
            model_info: 原始探测结果
            override: 用户提供的覆盖字典
        
        Returns:
            ModelInfo: 应用覆盖后的模型信息
        """
        # 将 dataclass 转换为字典方便修改
        from dataclasses import asdict
        info_dict = asdict(model_info)
        
        # 处理 architecture 键：支持 str 或 ModelArchitecture
        if "architecture" in override:
            arch_val = override["architecture"]
            if isinstance(arch_val, str):
                # 尝试转换为枚举
                try:
                    arch_val = ModelArchitecture[arch_val.upper()]
                except KeyError:
                    # 找不到则保持原始值
                    pass
            info_dict["architecture"] = arch_val
        
        # 处理其他数字键
        for key in ("num_layers", "num_heads", "hidden_dim", "patch_size",
                    "window_size", "num_experts"):
            if key in override:
                info_dict[key] = override[key]
        
        return ModelInfo(**info_dict)

    def _compute_confidence(
        self,
        model: nn.Module,
        detected: ModelArchitecture,
        matched_archs: Optional[List[tuple]] = None,
    ) -> float:
        """
        计算探测结果的置信度。
        
        策略：根据匹配特征的数量和歧义模块重叠程度计算置信度。
        多模态/混合模型通常会导致置信度下降。
        
        Args:
            model: PyTorch模型实例
            detected: 已探测到的架构类型
            matched_archs: 所有匹配的架构列表，用于计算歧义度
        
        Returns:
            float: 置信度值 [0.0, 1.0]
        """
        if matched_archs is None:
            matched_archs = []
        
        base_confidence = 0.9  # 基础置信度
        
        # 多个架构同时匹配时降低置信度
        if len(matched_archs) > 1:
            base_confidence -= 0.2 * (len(matched_archs) - 1)
        
        # 进一步根据匹配特征数量模拟评估
        feature_count = 0
        
        if detected == ModelArchitecture.VIT:
            if any(hasattr(m, 'patch_embed') or 'PatchEmbed' in type(m).__name__
                   for _, m in model.named_modules()):
                feature_count += 1
            if hasattr(model, 'cls_token') or any(
                'cls_token' in n for n, _ in model.named_parameters()):
                feature_count += 1
            if hasattr(model, 'blocks') or hasattr(model, 'encoder'):
                feature_count += 1
        
        elif detected == ModelArchitecture.SWIN:
            if any('WindowAttention' in type(m).__name__ or
                   'SwinTransformerBlock' in type(m).__name__
                   for _, m in model.named_modules()):
                feature_count += 2  # 独特特征，权重较高
            if hasattr(model, 'layers') or hasattr(model, 'stages'):
                feature_count += 1
        
        elif detected == ModelArchitecture.MOE_TRANSFORMER:
            if any('MoE' in type(m).__name__ or 'Expert' in type(m).__name__
                   for _, m in model.named_modules()):
                feature_count += 1
            if any('Router' in type(m).__name__ or 'Gate' in type(m).__name__
                   for _, m in model.named_modules()):
                feature_count += 1
            if hasattr(model, 'num_experts'):
                feature_count += 1
        
        elif detected == ModelArchitecture.TRANSFORMER:
            if any('TransformerEncoder' in type(m).__name__
                   for _, m in model.named_modules()):
                feature_count += 1
            if any(isinstance(m, nn.MultiheadAttention)
                   for _, m in model.named_modules()):
                feature_count += 1
            if hasattr(model, 'd_model') or hasattr(model, 'nhead'):
                feature_count += 1
        
        # 根据特征数量调整置信度（每个特征加 0.05，最多加 0.1）
        feature_bonus = min(feature_count * 0.05, 0.1)
        confidence = base_confidence + feature_bonus
        
        return max(0.0, min(1.0, confidence))

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
