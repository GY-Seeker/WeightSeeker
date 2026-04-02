"""
基础使用示例

参考 design.md 第5节的使用示例代码
"""

# 方式1：极简使用（推荐）
from transformer_analyzer import AnalysisPipeline, Config

pipeline = AnalysisPipeline(
    model_path="path/to/model.pth",
    data_path="path/to/test_images/",
    config=Config()
)
results = pipeline.run(output_dir="./results")


# 方式2：高级使用（手动控制）
from transformer_analyzer import (
    Config, ArchitectureDetector, HookManager,
    ForwardTracker, BackwardTracker, CrossSampleAccumulator,
    SingleSampleAnalyzer, GlobalDiagnosisEngine,
    FusionComposer, Dashboard,
    ModelLoader, DataManager
)

# 1. 加载模型并初始化
config = Config()

# 使用ModelLoader加载模型
model_loader = ModelLoader(device="auto", precision="fp32")
model = model_loader.load_from_checkpoint("path/to/model.pth")

# 使用DataManager加载数据
data_manager = DataManager(config)
dataloader = data_manager.load_image_dataset("path/to/test_images/")

# 2. 架构探测与Hook注册
detector = ArchitectureDetector()
model_info = detector.detect(model)
hook_manager = HookManager(model, model_info)
hook_manager.register_all_hooks()

# 3. 初始化追踪器和累积器
forward_tracker = ForwardTracker(hook_manager)
backward_tracker = BackwardTracker(model, hook_manager)
accumulator = CrossSampleAccumulator(model_info)

# 4. 处理数据批次
for batch_data in dataloader:
    # 验证输入
    if not data_manager.validate_input(batch_data):
        continue
    
    # 前向传播
    attention_maps = forward_tracker.track(model, batch_data)
    # loss = compute_loss(model, batch_data)  # 用户需自行实现
    
    # 反向传播
    # gradients = backward_tracker.track(loss)
    # input_gradients = gradients["input"]
    # hidden_gradients = gradients["hidden"]
    # attention_gradients = gradients.get("attention")
    
    # 更新累积器
    # accumulator.update(
    #     attention_maps=attention_maps,
    #     input_gradients=input_gradients,
    #     hidden_gradients=hidden_gradients,
    #     attention_gradients=attention_gradients,
    # )

# 5. 单样本分析
sample_analyzer = SingleSampleAnalyzer(
    model_info.num_layers, 
    model_info.num_heads,
    threshold_method="median"
)
# single_result = sample_analyzer.analyze(attention_maps, gradient_maps, normalized_data)

# 6. 全局诊断
diagnosis_engine = GlobalDiagnosisEngine(accumulator)
# global_result = diagnosis_engine.diagnose()

# 7. 融合与可视化
fusion_composer = FusionComposer()
# fusion_map = fusion_composer.compose(attention_map, gradient_map)

# dashboard = Dashboard(...)
# dashboard.generate_full_report(single_result, global_result, output_dir="./results")
