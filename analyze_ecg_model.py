"""
ECG Transformer 模型分析脚本

使用 AnalysisPipeline 对 ECG 变压器模型进行完整分析。
功能：
1. 加载模型权重（model_full.pth）
2. 加载测试数据（PTB-XL 心电图记录）
3. 执行前向和反向传播分析
4. 输出所有可用的可视化图像
"""

import os
import sys
import torch
import logging

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AnalysisPipeline, PipelineConfig

# 测试模型导入（从 tests/test_model 目录）
TEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'test_model')
sys.path.insert(0, TEST_MODEL_DIR)

from model import MultiModalECGTransformer
from dateset import PTBXLDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主分析流程"""
    
    # ==================== 配置参数 ====================
    MODEL_PATH = os.path.join(TEST_MODEL_DIR, 'models', 'model_full.pth')
    DATA_ROOT = os.path.join(TEST_MODEL_DIR, 'records100')
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_output')
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ECG Transformer 模型分析开始")
    logger.info("=" * 60)
    
    # ==================== 步骤 1: 加载模型 ====================
    logger.info("\n[步骤 1] 加载模型权重...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"模型文件不存在：{MODEL_PATH}")
        return
    
    model = MultiModalECGTransformer(
        num_classes=5,
        d_model=128,
        nhead=4,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.3,
        use_meta=True,
        use_cnn=True
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.error(f"不支持的 checkpoint 格式：{type(checkpoint)}")
        return
    
    logger.info(f"✓ 模型权重加载成功：{MODEL_PATH}")
    
    # ==================== 步骤 2: 加载测试数据 ====================
    logger.info("\n[步骤 2] 加载测试数据...")
    
    try:
        dataset = PTBXLDataset(
            root_dir=DATA_ROOT,
            usage='test',
            sampling_rate=100
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"✓ 测试数据加载成功，共 {len(dataset)} 个样本")
        
    except Exception as e:
        logger.warning(f"加载 PTB-XL 数据集失败：{e}")
        logger.info("使用随机生成的测试数据...")
        
        batch_size = 4
        seq_length = 2500
        meta_dim = 3
        
        ecg_data = torch.randn(batch_size, 12, seq_length)
        meta_data = torch.randn(batch_size, meta_dim)
        
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(ecg_data, meta_data)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        logger.info(f"✓ 使用随机数据：{batch_size} 个样本")
    
    # ==================== 步骤 3: 配置并执行分析 ====================
    logger.info("\n[步骤 3] 配置分析管道...")
    
    config = PipelineConfig(
        skip_spatial=True,              # 1D 序列不需要空间重构
        skip_global_diagnosis=False,    # 启用全局诊断
        skip_fusion=False,              # 启用特征融合
        skip_visualization=False,       # 启用可视化
        output_dir=OUTPUT_DIR,
        save_visualizations=True,
        save_raw_tensors=True,
        accumulator_limit=100,          # 限制累积样本数用于快速测试
        detector_override={
            "architecture": "TRANSFORMER",
            "num_layers": 6,
            "num_heads": 4,
            "hidden_dim": 128,
        },
        # 多模态输入：ECG 信号 + 元数据
        input_adapter_auxiliary={'meta_data': torch.zeros(4, 3)},
    )
    
    logger.info("✓ 分析管道配置完成（skip_spatial=True 启用时序数据可视化）")
    
    # ==================== 步骤 4: 执行分析 ====================
    logger.info("\n[步骤 4] 执行模型分析...")
    
    try:
        pipeline = AnalysisPipeline(
            model=model,
            config=config
        )
        
        # 获取一个批次的数据
        batch = next(iter(data_loader))
        
        if isinstance(batch, (list, tuple)):
            ecg_signal = batch[0]
            meta_data = batch[1] if len(batch) > 1 else torch.zeros(ecg_signal.shape[0], 3)
        else:
            ecg_signal = batch
            meta_data = torch.zeros(ecg_signal.shape[0], 3)
        
        logger.info(f"输入数据形状：ECG={ecg_signal.shape}, Meta={meta_data.shape}")
        
        # 执行单样本分析
        results = pipeline.run_single(ecg_signal)
        
        logger.info("✓ 分析完成！")
        
        # ==================== 步骤 5: 输出结果统计 ====================
        logger.info("\n[步骤 5] 结果统计...")
        
        logger.info(f"注意力图数量：{len(results.get('attention_maps', {}))}")
        logger.info(f"梯度图数量：{len(results.get('gradient_maps', {}).get('hidden', {}))}")
        
        if 'single_sample' in results:
            single_result = results['single_sample']
            logger.info(f"层重要性分析：{len(single_result.get('layer_importance', {}))} 层")
            logger.info(f"头分类结果：{len(single_result.get('head_classification', {}))} 个头")
        
        # 可视化路径
        if 'visualization_paths' in results:
            vis_paths = results['visualization_paths']
            logger.info("\n已生成的可视化文件:")
            for name, path in vis_paths.items():
                logger.info(f"  - {name}: {path}")
            
            # 时序数据可视化增强
            logger.info("\n【时序数据可视化增强】⭐ 主要图表:")
            expected_files = {
                'token_importance_fusion': 'Token 重要性融合图（注意力×梯度）',
                'key_segments_annotation': '关键区段标注图（峰值/谷值/突变点）',
                'multi_layer_attention_seq': '多层注意力对比图（时序版）',
            }
            for key, desc in expected_files.items():
                if key in vis_paths:
                    logger.info(f"  ✅ {key}: {desc}")
                    logger.info(f"     路径：{vis_paths[key]}")
                else:
                    logger.warning(f"  ⚠️ {key} 未生成（可能缺少梯度信息）")
            
            logger.info("\n【保留的通用图表】:")
            retained_files = {
                'all_heads_heatmap': '所有注意力头全景热力图',
                'multi_layer_comparison': '多层注意力对比图（通用版）',
            }
            for key, desc in retained_files.items():
                if key in vis_paths:
                    logger.info(f"  ✓ {key}: {desc}")
        
        # 原始张量路径
        if 'raw_tensors_path' in results:
            logger.info(f"\n原始张量已保存：{results['raw_tensors_path']}")
        
        # ==================== 步骤 6: 批量分析 ====================
        logger.info("\n[步骤 6] 执行批量分析（最多 10 个批次）...")
        
        batch_results = pipeline.run_batch(data_loader, max_samples=40)
        
        logger.info(f"✓ 批量分析完成，共处理 {batch_results['total_samples_processed']} 个样本")
        
        if 'global_diagnosis' in batch_results and batch_results['global_diagnosis']:
            global_diag = batch_results['global_diagnosis']
            logger.info("\n全局诊断结果:")
            if 'activation_frequency_ranking' in global_diag:
                logger.info(f"  - 激活频率排名：{len(global_diag['activation_frequency_ranking'])} 层")
            if 'gradient_importance_ranking' in global_diag:
                logger.info(f"  - 梯度重要性排名：{len(global_diag['gradient_importance_ranking'])} 层")
            if 'anomaly_analysis' in global_diag:
                logger.info(f"  - 异常分析：{global_diag['anomaly_analysis']}")
            if 'head_classification' in global_diag:
                logger.info(f"  - 头分类：{len(global_diag['head_classification'])} 个头")
        
        acc_state = batch_results.get('accumulator_state')
        if acc_state:
            logger.info(f"\n累积器状态:")
            logger.info(f"  - 样本数：{acc_state.sample_count}")
        
        logger.info("\n" + "=" * 60)
        logger.info("分析完成！输出目录：" + OUTPUT_DIR)
        logger.info("=" * 60)
        
        logger.info("\n📊 查看结果建议:")
        logger.info("  1. 主要图表：token_importance_fusion.png")
        logger.info("     - 显示原始 ECG 信号 + 注意力热力图叠加")
        logger.info("     - 红色越深表示模型越关注该时间步")
        logger.info("  2. 关键区段：key_segments_annotation.png")
        logger.info("     - 自动检测并标注峰值、谷值、突变点")
        logger.info("     - 黄色高亮表示重要性 > 0.3 的区域")
        logger.info("  3. 多层对比：multi_layer_attention_seq.png")
        logger.info("     - 对比不同 Transformer 层的关注模式")
        logger.info("     - 浅层关注低层次特征，深层关注高层次模式")
        
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"\n❌ 分析过程出错：{e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
