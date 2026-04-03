"""
Swin Transformer 图像模型全流程分析脚本

使用 pipeline.py 对 tests/test_model2 下的 SwinIR 模型进行完整分析。
功能：
1. 加载 SwinIR 模型权重（tokenizer/checkpoint.pth）
2. 生成测试图像数据
3. 执行前向和反向传播分析
4. 输出所有可用的可视化图像（注意力热力图、多层对比等）
"""

import os
import sys
import torch
import torch.nn as nn
import logging
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
# 注意：顺序很重要，先添加 test_model2，再添加 transformer_analyzer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_MODEL2_DIR = os.path.join(SCRIPT_DIR, 'tests', 'test_model2')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 先添加 test_model2 确保 models 和 utils 正确导入
sys.path.insert(0, TEST_MODEL2_DIR)
# 然后添加项目根目录
sys.path.insert(0, PROJECT_ROOT)

# 从 tests/test_model2/models 导入 SwinIR （必须在导入 transformer_analyzer 之前）
try:
    from models.SwinTransformer import SwinIR
    logger.info("✓ 成功导入 SwinIR 模型")
except ImportError as e:
    logger.error(f"无法导入 SwinIR 模型：{e}")
    logger.info("提示：可能需要安装依赖：pip install vector_quantize_pytorch")
    raise

# 现在导入 transformer_analyzer
from transformer_analyzer.pipeline import AnalysisPipeline, PipelineConfig



def load_real_images(data_dir, img_size=256, max_samples=10):
    """加载真实图像数据
    
    Args:
        data_dir: 图像数据目录路径
        img_size: 图像大小
        max_samples: 最大加载样本数
    
    Returns:
        DataLoader: 图像数据加载器
    """
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import os
    
    class ImageDataset(Dataset):
        def __init__(self, root_dir, img_size=256, max_samples=10):
            self.root_dir = root_dir
            self.img_size = img_size
            self.image_paths = []
            
            # 支持的图像格式
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            # 遍历目录收集图像路径
            count = 0
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if count >= max_samples:
                        break
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_extensions:
                        self.image_paths.append(os.path.join(root, file))
                        count += 1
                if count >= max_samples:
                    break
            
            logger.info(f"找到 {len(self.image_paths)} 张图像")
            
            # 定义转换
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # 标准化（可选，根据模型需求）
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            try:
                # 加载图像
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert('RGB')
                
                # 应用转换
                image_tensor = self.transform(image)
                
                return image_tensor
            except Exception as e:
                logger.warning(f"加载图像失败 {self.image_paths[idx]}: {e}")
                # 返回随机图像作为后备
                return torch.randn(3, self.img_size, self.img_size)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        logger.warning(f"数据目录不存在：{data_dir}，将使用随机数据")
        return None
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(root_dir=data_dir, img_size=img_size, max_samples=max_samples)
    
    if len(dataset) == 0:
        logger.warning("未找到任何图像文件，将使用随机数据")
        return None
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"✓ 成功加载 {len(dataset)} 张图像")
    return data_loader


class SimpleSwinIRWrapper(torch.nn.Module):
    """SwinIR 模型的简单包装器，只返回 output image
    
    SwinIR 原始 forward 返回 (output, loss) tuple，
    此包装器将其转换为只返回 output，适配 pipeline.py
    """
    def __init__(self, swinir_model):
        super().__init__()
        self.swinir_model = swinir_model
    
    def forward(self, x):
        result = self.swinir_model(x)
        # SwinIR 返回 (output_image, loss) tuple
        # 我们只取第一个元素（output_image）
        if isinstance(result, tuple):
            return result[0]
        return result


def load_swinir_model(checkpoint_path, img_size=256):
    """加载 SwinIR 模型"""
    
    logger.info(f"尝试加载 checkpoint: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 文件不存在：{checkpoint_path}")
        return None
    
    try:
        # 根据 tokenizer.yaml 配置创建模型实例
        # 配置来源：configs/tokenizer.yaml + checkpoint 推断
        model = SwinIR(
            upscale=1,                    # 去噪任务
            img_size=256,                 # 图像尺寸
            patch_size=16,                 # patch 大小 (checkpoint: 4096 patches = 64x64)
            in_chans=3,                   # RGB 图像
            latent_dim=32,                # 潜在维度 (checkpoint: [1, 4096, 32])
            codebook_size=8192,           # 码书大小 (tokenizer.yaml)
            window_size=1,                # 窗口大小
            img_range=1.,                 # 灰度范围 [0, 1]
            depths=[6, 6, 6, 6],          # 4个 stage，每个 6 层
            embed_dim=64,                 # 嵌入维度 (tokenizer.yaml: dim=64)
            num_heads=[6, 6, 6, 6],       # 注意力头数
            mlp_ratio=2.,                 # MLP 比率
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upsampler='',                 # 无上采样
            resi_connection='1conv',
        )
        
        logger.info("✓ SwinIR 模型创建成功")
        logger.info(f"  - img_size: 256")
        logger.info(f"  - patch_size: 4 (checkpoint: 4096 patches)")
        logger.info(f"  - embed_dim: 64")
        logger.info(f"  - latent_dim: 32")
        logger.info(f"  - codebook_size: 1024")
        logger.info(f"  - depths: [6, 6, 6, 6]")
        logger.info(f"  - window_size: 8")
        
        # 加载 checkpoint
        # PyTorch 2.6+ 需要设置 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 处理不同的 checkpoint 格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                # 直接就是 state_dict
                state_dict = checkpoint
        else:
            logger.error(f"不支持的 checkpoint 格式：{type(checkpoint)}")
            return None
        
        # 移除可能的前缀 (如 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 加载权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"✓ 模型权重加载完成：{msg}")
        
        # 使用包装器包装模型，使其只返回 output image
        wrapped_model = SimpleSwinIRWrapper(model)
        logger.info("✓ SwinIR 模型已包装（仅返回 output image）")
        
        return wrapped_model
        
    except Exception as e:
        logger.error(f"加载模型失败：{e}", exc_info=True)
        return None


def main():
    """主分析流程"""
    
    # ==================== 配置参数 ====================
    TEST_MODEL2_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tests',
        'test_model2'
    )
    
    # Tokenizer checkpoint 路径
    CHECKPOINT_PATH = os.path.join(
        TEST_MODEL2_DIR,
        'checkpoints',
        'tokenizer',
        'tokenizer_epoch_3.pt'  # 使用 epoch 3 的 checkpoint
    )
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'image_analysis_output'
    )
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Swin Transformer 图像模型分析开始")
    logger.info("="*60)
    logger.info(f"配置文件: configs/tokenizer.yaml")
    logger.info(f"  - dim: 64")
    logger.info(f"  - patch_size: 16")
    logger.info(f"  - num_latent_tokens: 256")
    logger.info(f"  - codebook_size: 1024")
    logger.info(f"  - enc_depth: 6")
    logger.info(f"  - dec_depth: 6")
    
    # ==================== 步骤 1: 加载模型 ====================
    logger.info("\n[步骤 1] 加载 SwinIR 模型...")
    
    # 尝试多个可能的 checkpoint 文件名（按优先级排序）
    
    model = None
    if os.path.exists(CHECKPOINT_PATH):
        model = load_swinir_model(CHECKPOINT_PATH, img_size=256)
        if model:
            logger.info(f"✓ 使用 checkpoint: {CHECKPOINT_PATH}")
    
    if not model:
        logger.warning("未找到 tokenizer checkpoint，尝试 classifier checkpoint...")
        # 尝试 classifier checkpoint
        classifier_ckpt = os.path.join(
            TEST_MODEL2_DIR,
            'checkpoints',
            'classifier',
            'binary_classifier_best.pt'
        )
        if os.path.exists(classifier_ckpt):
            # 注意：这里需要根据实际模型结构调整
            logger.warning(f"Classifier checkpoint 不兼容 SwinIR，将使用随机初始化模型")
            model = SwinIR(img_size=64)
        else:
            logger.error("未找到任何可用的 checkpoint，使用随机初始化模型")
            model = SwinIR(img_size=64)
    
    # 设置模型为评估模式
    # model.eval()
    
    # 确保模型参数需要梯度（用于分析）
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"需要梯度的参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ==================== 步骤 2: 准备测试数据 ====================
    logger.info("\n[步骤 2] 准备测试图像...")
    
    # 尝试加载真实图像数据 - 使用正确的数据目录
    DATA_DIR = os.path.join(
        TEST_MODEL2_DIR, 
        'breast_cancer_data_IHC4BC_justHE',
        'tokenizer_dataset',
        'all_images'
    )  # SwinIR tokenizer 训练数据集目录
    
    data_loader = None
    
    # 方法 1: 使用自定义的图像加载器
    if os.path.exists(DATA_DIR):
        try:
            logger.info(f"尝试从 {DATA_DIR} 加载图像...")
            # 尝试从 data 目录加载随机图像（默认 256x256）
            data_loader = load_real_images(DATA_DIR, img_size=256, max_samples=10)
        except Exception as e:
            logger.warning(f"加载真实图像失败：{e}")
    else:
        logger.warning(f"数据目录不存在：{DATA_DIR}")
    
    if data_loader is None:
        # 如果没有真实数据，使用随机生成的测试图像
        logger.info("使用随机生成的测试图像")
        img_size = 256  # SwinIR 默认输入大小
        test_image = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(test_image)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        logger.info(f"✓ 测试图像形状：{test_image.shape}")
    else:
        # 从 data_loader 获取第一个批次作为测试图像
        test_image = next(iter(data_loader))
        if isinstance(test_image, (list, tuple)):
            test_image = test_image[0]
        # 确保需要梯度
        if not test_image.requires_grad:
            test_image = test_image.clone().detach().requires_grad_(True)
        logger.info(f"✓ 测试图像形状：{test_image.shape}")
    
    # ==================== 步骤 3: 配置分析管道 ====================
    logger.info("\n[步骤 3] 配置分析管道...")
    
    # SwinIR 是图像恢复/超分模型，需要空间重构
    # 但现在我们有了 ImageVisualizer，可以更好地处理图像模型
    config = PipelineConfig(
        skip_spatial=True,              # 这个重构只针对ViT
        skip_global_diagnosis=False,    # 启用全局诊断
        skip_fusion=False,              # 启用特征融合
        skip_visualization=False,       # 启用可视化
        output_dir=OUTPUT_DIR,
        save_visualizations=True,
        save_raw_tensors=True,
        precision='fp32',
        accumulator_limit=3000,         # 限制样本数（图像模型较慢，但注意力矩阵维度可能导致误判）
        # 架构覆盖：SwinIR 使用 Swin Transformer
        detector_override={
            "architecture": "TRANSFORMER",  # Swin 是 Transformer 变体
            "num_layers": 24,              # 4 个 RSTB 层 × 6 个 Swin Block = 24 个注意力层
            "num_heads": 6,                # 每个头 6 个 attention heads
            "hidden_dim": 96,              # 基础隐藏维度（从 checkpoint 推断）
        },
    )
    
    logger.info("✓ 分析管道配置完成（已添加架构覆盖）")
    logger.info("✓ skip_spatial=False 将使用图像模型可视化路径（ImageVisualizer）")
    
    # ==================== 步骤 4: 执行分析 ====================
    logger.info("\n[步骤 4] 执行模型分析...")
    
    try:
        # 创建分析管道
        pipeline = AnalysisPipeline(
            model=model,
            config=config
        )
        
        logger.info("✓ AnalysisPipeline 创建完成")
        
        # 执行单样本分析
        results = pipeline.run_single(test_image)
        
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
            
            # 主要图表
            logger.info("\n【主要图表】⭐:")
            expected_files = {
                'token_importance_fusion': 'Token 重要性融合图（注意力×梯度）',
                'all_heads_heatmap': '所有注意力头全景热力图',
                'multi_layer_comparison': '多层注意力对比图',
            }
            for key, desc in expected_files.items():
                if key in vis_paths:
                    logger.info(f"  ✅ {key}: {desc}")
                    logger.info(f"     路径：{vis_paths[key]}")
                else:
                    logger.warning(f"  ⚠️ {key} 未生成（可能缺少梯度信息或架构不兼容）")
        
        # 原始张量路径
        if 'raw_tensors_path' in results:
            logger.info(f"\n原始张量已保存：{results['raw_tensors_path']}")
        
        # ==================== 步骤 6: 批量分析（可选） ====================
        logger.info("\n[步骤 6] 执行批量分析（最多 10 个批次）...")
        
        # 使用之前加载的 data_loader，不需要重新创建
        batch_results = pipeline.run_batch(data_loader, max_samples=10)
        
        logger.info(f"✓ 批量分析完成，共处理 {batch_results['total_samples_processed']} 个样本")
        
        # 全局诊断结果
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
        
        # 累积器状态
        acc_state = batch_results.get('accumulator_state')
        if acc_state:
            logger.info(f"\n累积器状态:")
            logger.info(f"  - 样本数：{acc_state.sample_count}")
        
        logger.info("\n" + "=" * 60)
        logger.info("分析完成！输出目录：" + OUTPUT_DIR)
        logger.info("=" * 60)
        
        # 输出使用说明
        logger.info("\n📊 查看结果建议:")
        logger.info("  1. 主要图表：all_heads_attention_heatmap.png")
        logger.info("     - 显示所有注意力头的激活模式")
        logger.info("     - 帮助理解不同头的功能分工")
        logger.info("  2. 多层对比：multi_layer_attention.png")
        logger.info("     - 对比不同 Swin Transformer 层的关注模式")
        logger.info("     - 浅层关注低层次特征，深层关注高层次模式")
        logger.info("  3. 重要性融合：token_importance_fusion.png（如果生成）")
        logger.info("     - 显示注意力权重与梯度的融合结果")
        logger.info("     - 红色越深表示模型越关注该区域")
        
        # 清理资源
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"\n❌ 分析过程出错：{e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
