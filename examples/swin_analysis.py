"""
Swin Transformer 分析示例

展示如何使用本系统分析 Swin Transformer 架构的模型
"""

from transformer_analyzer import (
    AnalysisPipeline,
    Config,
    ModelLoader,
    ArchitectureDetector,
    SwinHandler,
)


def main():
    """Swin Transformer 分析示例"""
    
    # 配置
    config = Config()
    
    # 加载 Swin 模型
    model_loader = ModelLoader(device="auto", precision="fp32")
    model = model_loader.load_from_timm("swin_tiny_patch4_window7_224")
    
    # 架构探测
    detector = ArchitectureDetector()
    model_info = detector.detect(model)
    print(f"检测到架构: {model_info.architecture}")
    print(f"层数: {model_info.num_layers}")
    print(f"头数: {model_info.num_heads}")
    print(f"窗口大小: {model_info.window_size}")
    
    # 方式1: 使用 Pipeline 一键分析
    pipeline = AnalysisPipeline(
        model_path="swin_tiny_patch4_window7_224",
        data_path="path/to/test_images/",
        config=config
    )
    results = pipeline.run(output_dir="./swin_results")
    
    print("Swin Transformer 分析完成！")


if __name__ == "__main__":
    main()
