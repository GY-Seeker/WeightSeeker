"""
自动生成分析报告 - 将分析结果转化为易读的文字报告
"""

import os
from datetime import datetime
from typing import Any, Dict, List


def generate_analysis_report(
    results: Dict[str, Any],
    accumulator_state: Any,
    output_path: str,
) -> None:
    """
    生成完整的分析报告（Markdown 格式）。
    
    Args:
        results: run_single 或 run_batch 的结果字典
        accumulator_state: AccumulatorState 对象
        output_path: 报告保存路径
    """
    report_lines = []
    
    # 标题
    report_lines.append("# Transformer 模型分析报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 执行摘要
    report_lines.append("## 📊 执行摘要")
    report_lines.append("")
    
    sample_count = accumulator_state.sample_count if hasattr(accumulator_state, 'sample_count') else 0
    report_lines.append(f"- **分析样本数**: {sample_count}")
    
    attention_maps = results.get("attention_maps", {})
    gradient_maps = results.get("gradient_maps", {})
    report_lines.append(f"- **Transformer 层数**: {len(attention_maps)}")
    report_lines.append(f"- **成功捕获注意力图**: {len(attention_maps)} 层")
    report_lines.append(f"- **成功捕获梯度图**: {len(gradient_maps.get('hidden', {}))} 层")
    report_lines.append("")
    
    # 层重要性分析
    single_sample = results.get("single_sample", {})
    layer_importance = single_sample.get("layer_importance", {})
    if layer_importance:
        report_lines.append("## 🔍 层重要性分析")
        report_lines.append("")
        
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_layers[:3]
        bottom_3 = sorted_layers[-3:] if len(sorted_layers) >= 3 else sorted_layers
        
        report_lines.append("### 最关键的层（Top 3）")
        report_lines.append("")
        for i, (layer_idx, score) in enumerate(top_3, 1):
            report_lines.append(f"{i}. **Layer {layer_idx}**: {score:.4f}")
        report_lines.append("")
        
        report_lines.append("### 最不活跃的层（Bottom 3）")
        report_lines.append("")
        for i, (layer_idx, score) in enumerate(bottom_3, 1):
            report_lines.append(f"{i}. Layer {layer_idx}: {score:.4f}")
        report_lines.append("")
        
        # 洞察
        if top_3:
            most_important = top_3[0][0]
            report_lines.append("**洞察**: ")
            if most_important == 0:
                report_lines.append("- 第 0 层（输入层）梯度最大，说明特征提取对原始输入非常敏感")
            elif most_important == len(sorted_layers) - 1:
                report_lines.append("- 最后一层梯度最大，接近输出端，对最终决策影响最直接")
            else:
                report_lines.append(f"- Layer {most_important} 是关键转换层，承载了最重要的信息处理")
        report_lines.append("")
    
    # 全局诊断（如果有累积器数据）
    if hasattr(accumulator_state, 'head_activation_freq'):
        report_lines.append("##  注意力头分析")
        report_lines.append("")
        
        import torch
        freq = accumulator_state.head_activation_freq
        if isinstance(freq, torch.Tensor):
            freq = freq.detach().cpu().float().numpy()
        
        if freq.size > 0:
            num_layers, num_heads = freq.shape
            avg_freq = freq.mean(axis=1)
            
            report_lines.append(f"- **总头数**: {num_layers * num_heads}")
            report_lines.append(f"- **平均激活频率**: {float(freq.mean()):.4f}")
            report_lines.append("")
            
            # 最活跃的层
            most_active_layer = int(avg_freq.argmax())
            least_active_layer = int(avg_freq.argmin())
            
            report_lines.append(f"- **最活跃的层**: Layer {most_active_layer} (平均频率 {avg_freq[most_active_layer]:.4f})")
            report_lines.append(f"- **最不活跃的层**: Layer {least_active_layer} (平均频率 {avg_freq[least_active_layer]:.4f})")
            report_lines.append("")
            
            # 识别冗余头
            redundant_threshold = 0.1
            redundant_heads = []
            for li in range(num_layers):
                for hi in range(num_heads):
                    if freq[li, hi] < redundant_threshold:
                        redundant_heads.append((li, hi))
            
            if redundant_heads:
                report_lines.append(f"- **潜在冗余头** (频率<{redundant_threshold}): {len(redundant_heads)} 个")
                for li, hi in redundant_heads[:5]:  # 只显示前 5 个
                    report_lines.append(f"  - Layer {li}, Head {hi}")
                if len(redundant_heads) > 5:
                    report_lines.append(f"  ... 还有 {len(redundant_heads) - 5} 个")
            report_lines.append("")
    
    # 四象限分析（如果有）
    quadrant_stats = single_sample.get("quadrant_stats", {})
    if quadrant_stats:
        report_lines.append("## 🗺️ 四象限分析")
        report_lines.append("")
        
        from ..core.types import Quadrant
        
        total_ratio = sum(quadrant_stats.values())
        if abs(total_ratio - 1.0) > 0.01:
            # 归一化
            quadrant_stats = {k: v / total_ratio for k, v in quadrant_stats.items()}
        
        for quadrant, ratio in sorted(quadrant_stats.items(), key=lambda x: x[1], reverse=True):
            q_name_map = {
                Quadrant.CORE_DISCRIMINATIVE: "核心判别区",
                Quadrant.REDUNDANT_ATTENTION: "冗余关注区",
                Quadrant.POTENTIAL_INFLUENCE: "潜在影响区",
                Quadrant.IRRELEVANT: "无关区域",
            }
            q_name = q_name_map.get(quadrant, quadrant.name)
            report_lines.append(f"- **{q_name}**: {ratio*100:.2f}%")
        report_lines.append("")
        
        # 解读
        core_ratio = quadrant_stats.get(Quadrant.CORE_DISCRIMINATIVE, 0.0)
        redundant_ratio = quadrant_stats.get(Quadrant.REDUNDANT_ATTENTION, 0.0)
        
        report_lines.append("**解读**: ")
        if core_ratio > 0.3:
            report_lines.append(f"- ✓ 核心判别区占比{core_ratio*100:.1f}%，模型注意力聚焦且有效")
        elif core_ratio < 0.1:
            report_lines.append(f"- ⚠ 核心判别区仅占{core_ratio*100:.1f}%，模型注意力可能过于分散")
        
        if redundant_ratio > 0.4:
            report_lines.append(f"- ⚠ 冗余关注区占比{redundant_ratio*100:.1f}%，存在大量无效注意力")
        report_lines.append("")
    
    # 可视化文件清单
    report_lines.append("## 📁 生成的可视化文件")
    report_lines.append("")
    
    vis_files = {
        "all_heads_attention_heatmap.png": "所有注意力头的热力图全景（每行 6 个）",
        "attention_heatmap.png": "单层单头详细热力图",
        "multi_layer_attention.png": "多层注意力对比图",
        "layer_importance.png": "各层重要性折线图（带数值标签）",
        "accumulator_stats.png": "累积器统计摘要（双子图）",
        "head_frequency_importance_scatter.png": "头频率 - 重要性散点图",
    }
    
    output_dir = os.path.dirname(output_path)
    for filename, description in vis_files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size_kb = os.path.getsize(filepath) / 1024
            report_lines.append(f"- **{filename}** ({file_size_kb:.1f} KB): {description}")
    report_lines.append("")
    
    # 建议与下一步
    report_lines.append("## 💡 优化建议")
    report_lines.append("")
    
    suggestions = []
    
    # 基于层重要性的建议
    if layer_importance:
        scores = list(layer_importance.values())
        if max(scores) > 2 * min(scores):
            suggestions.append("🔹 层间梯度差异较大，可考虑对低梯度层进行剪枝实验")
    
    # 基于注意力的建议
    if hasattr(accumulator_state, 'head_activation_freq'):
        import torch
        freq_tensor = accumulator_state.head_activation_freq
        if isinstance(freq_tensor, torch.Tensor):
            freq_tensor = freq_tensor.detach().cpu().float()
            if freq_tensor.numel() > 0:
                freq = freq_tensor.numpy()
                if freq.min() < 0.05:
                    suggestions.append("🔹 存在极低频的注意力头，可尝试移除以压缩模型")
    
    if suggestions:
        for suggestion in suggestions:
            report_lines.append(suggestion)
    else:
        report_lines.append("✓ 当前模型状态良好，无需特殊优化")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*本报告由 Transformer Analyzer 自动生成*")
    
    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
