"""融合工具函数 - 注意力与梯度的简化融合（无策略模式，纯工具函数）。"""

import torch
from torch import Tensor
from typing import Dict, Optional


def normalize_for_fusion(
    tensor: Tensor,
    low_percentile: float = 0.01,
    high_percentile: float = 0.99,
) -> Tensor:
    """百分位裁剪 + Min-Max 归一化至 [0, 1]。"""
    if low_percentile >= high_percentile:
        raise ValueError(f"low_percentile({low_percentile}) >= high_percentile({high_percentile})")
    flat = tensor.flatten().float()
    low_val = torch.quantile(flat, low_percentile).item()
    high_val = torch.quantile(flat, high_percentile).item()
    clipped = tensor.float().clamp(min=low_val, max=high_val)
    v_min = clipped.min().item()
    v_max = clipped.max().item()
    if abs(v_max - v_min) < 1e-8:
        return torch.zeros_like(clipped)
    return (clipped - v_min) / (v_max - v_min)


def weighted_sum_fusion(
    attention: Tensor,
    gradient: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """加权求和融合：alpha * attention + (1-alpha) * gradient。"""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 必须在 [0,1]，当前：{alpha}")
    if attention.shape != gradient.shape:
        raise ValueError(f"形状不一致：{attention.shape} vs {gradient.shape}")
    return alpha * attention + (1.0 - alpha) * gradient


def gradcam_fusion(
    attention: Tensor,
    gradient: Tensor,
) -> Tensor:
    """GradCAM 式融合：normalize(attention * gradient)。"""
    if attention.shape != gradient.shape:
        raise ValueError(f"形状不一致：{attention.shape} vs {gradient.shape}")
    product = attention * gradient
    v_min = product.min().item()
    v_max = product.max().item()
    if abs(v_max - v_min) < 1e-8:
        return torch.zeros_like(product)
    return (product - v_min) / (v_max - v_min)


def compute_token_importance(
    attention_maps: Dict[int, Tensor],   # {layer_idx: (B, H, L, L)}
    hidden_gradients: Dict[int, Tensor], # {layer_idx: (B, L)}
    method: str = "gradcam",             # 'gradcam' | 'weighted_sum' | 'multiply'
    layer_weights: Optional[Dict[int, float]] = None,  # 各层权重（默认均匀）
) -> Tensor:
    """
    计算每个 token 的重要性得分（用于 ECG 的每个时间点/NLP 的每个 token）
    
    Args:
        attention_maps: 各层的注意力图 {layer_idx: (B, H, L, L)}
        hidden_gradients: 各层的隐藏状态梯度 {layer_idx: (B, L)}
        method: 融合方法
            - 'gradcam': Grad-CAM 方式（注意力 × 梯度）
            - 'weighted_sum': 加权求和
            - 'multiply': 简单相乘
        layer_weights: 各层权重，None 表示均匀平均
    
    Returns:
        importance_scores: (B, L) - 每个样本、每个时间点的重要性得分
    """
    # 步骤 1: 从注意力矩阵提取对角线（自我关注度）
    # attn_diagonals[layer_idx] = (B, L)
    attn_diagonals = {}
    for layer_idx, attn in attention_maps.items():
        if attn.dim() == 4:  # (B, H, L, L)
            # 取所有头的对角线并平均
            B, H, L, _ = attn.shape
            diagonal = attn[:, :, range(L), range(L)]  # (B, H, L)
            attn_diagonals[layer_idx] = diagonal.mean(dim=1)  # (B, L)
        elif attn.dim() == 3:  # (B, L, L)
            B, L, _ = attn.shape
            attn_diagonals[layer_idx] = attn[:, range(L), range(L)]  # (B, L)
    
    # 步骤 2: 对每层进行注意力和梯度的融合
    fused_per_layer = []
    for layer_idx in attn_diagonals.keys():
        if layer_idx not in hidden_gradients:
            continue
            
        attn = attn_diagonals[layer_idx]  # (B, L)
        grad = hidden_gradients[layer_idx]  # (B, L)
        
        # 【维度对齐】检查并修正可能的维度颠倒
        if grad.dim() == 2 and attn.dim() == 2:
            if grad.shape == attn.shape:
                pass  # 形状一致，无需调整
            elif grad.shape == (attn.shape[1], attn.shape[0]):
                # 维度颠倒了：(L, B) -> (B, L)
                grad = grad.T  # 转置修正
            elif grad.shape[0] == attn.shape[0] and grad.shape[1] == 1 and attn.shape[1] > 1:
                # (B, 1) vs (B, L)：扩展梯度
                grad = grad.expand_as(attn)
            elif grad.shape[1] == attn.shape[1] and grad.shape[0] == 1 and attn.shape[0] > 1:
                # (1, L) vs (B, L)：扩展梯度
                grad = grad.expand_as(attn)
            else:
                # 无法自动对齐，抛出异常让上层处理
                raise ValueError(f"形状不一致且无法自动对齐：{attn.shape} vs {grad.shape}")
        elif grad.shape != attn.shape:
            raise ValueError(f"形状不一致：{attn.shape} vs {grad.shape}")
        
        # 归一化到 [0, 1]
        attn_norm = normalize_for_fusion(attn)
        grad_norm = normalize_for_fusion(grad)
        
        # 融合
        if method == 'gradcam':
            fused = gradcam_fusion(attn_norm, grad_norm)
        elif method == 'weighted_sum':
            fused = weighted_sum_fusion(attn_norm, grad_norm, alpha=0.5)
        else:  # multiply
            fused = (attn_norm * grad_norm)
        
        fused_per_layer.append(fused)
    
    # 步骤 3: 对所有层的结果取平均（或加权平均）
    if len(fused_per_layer) == 0:
        return torch.zeros_like(list(attention_maps.values())[0][:, 0, :])  # (B, L)
    
    stacked = torch.stack(fused_per_layer, dim=0)  # (num_layers, B, L)
    
    if layer_weights is not None:
        # 加权平均
        weights = torch.tensor([layer_weights.get(i, 1.0) 
                               for i in range(len(fused_per_layer))],
                              dtype=torch.float32, device=stacked.device)
        weights = weights / weights.sum()
        importance = (stacked * weights[:, None, None]).sum(dim=0)
    else:
        # 均匀平均
        importance = stacked.mean(dim=0)  # (B, L)
    
    # 步骤 4: 最终归一化到 [0, 1]
    importance = normalize_for_fusion(importance)
    
    return importance
