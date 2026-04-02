"""
MoE (Mixture of Experts) Transformer 分析示例

展示如何使用本系统分析 MoE-Transformer 架构的模型
"""

from transformer_analyzer import (
    AnalysisPipeline,
    Config,
    ModelLoader,
    ArchitectureDetector,
    MoEHandler,
)


def main():
    """MoE Transformer 分析示例"""
    
    # 配置
    config = Config()
    
    # 加载 MoE 模型
    model_loader = ModelLoader(device="auto", precision="fp32")
    # 假设从 checkpoint 加载 MoE 模型
    model = model_loader.load_from_checkpoint("path/to/moe_model.pth")
    
    # 架构探测
    detector = ArchitectureDetector()
    model_info = detector.detect(model)
    print(f"检测到架构: {model_info.architecture}")
    print(f"层数: {model_info.num_layers}")
    print(f"头数: {model_info.num_heads}")
    print(f"专家数量: {model_info.num_experts}")
    
    # 方式1: 使用 Pipeline 一键分析
    pipeline = AnalysisPipeline(
        model_path="path/to/moe_model.pth",
        data_path="path/to/test_data/",
        config=config
    )
    results = pipeline.run(output_dir="./moe_results")
    
    # 获取全局诊断报告（包含专家负载分析）
    global_diagnosis = pipeline.get_global_diagnosis()
    if "expert_load_distribution" in global_diagnosis:
        print("专家负载分布:", global_diagnosis["expert_load_distribution"])
    
    print("MoE Transformer 分析完成！")


if __name__ == "__main__":
    main()
