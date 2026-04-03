# Transformer权重监控与分析系统 - 详细设计文档

## 1. 项目概述

本项目是一款专为深度解析Transformer类模型内部决策机制而设计的可解释性分析与可视化工具，支持标准Transformer、MoE-Transformer、Swin Transformer以及ViT四种架构。

---

## 2. 项目目录结构

```
transformer_analyzer/
├── core/                           # 核心基础模块
│   ├── __init__.py
│   ├── config.py                   # 全局配置管理
│   ├── types.py                    # 类型定义与枚举
│   └── exceptions.py               # 自定义异常类
│
├── pipeline.py                    # 推理编排器（主入口）
│
├── model_adapter/                  # 模块1: 模型适配与全局Hook层
│   ├── __init__.py
│   ├── detector.py                 # 架构探测（支持手动override）
│   ├── hooks.py                    # Hook注册与管理（自动处理need_weights）
│   ├── swin_handler.py             # Swin特殊处理
│   └── moe_handler.py              # MoE路由处理
│
├── tracker/                        # 模块2: 动态追踪与跨样本累积层
│   ├── __init__.py
│   ├── forward_tracker.py          # 前向传播追踪（支持InputAdapter）
│   ├── backward_tracker.py         # 反向传播追踪
│   ├── accumulator.py              # 跨样本累积器
│   └── metrics.py                  # 指标计算（熵、范数等）
│
├── spatial/                        # [可选] 模块3: 空间重构与数值对齐层（仅图像输入需要）
│   ├── __init__.py
│   ├── reshaper.py                 # 粒度转换
│   ├── normalizer.py               # 尺度对齐
│   └── interpolator.py             # 插值处理
│
├── analyzer/                       # 模块4: 单样本解释与全局权重诊断引擎
│   ├── __init__.py
│   ├── single_sample.py            # 单样本解释（轨道A）
│   ├── global_diagnosis.py         # 全局权重诊断（轨道B）
│   ├── quadrant.py                 # 四象限划分
│   ├── anomaly_detector.py         # 异常识别
│   └── fusion_utils.py             # 融合工具（原fusion/简化后并入）
│
├── visualization/                  # 模块5: 可视化工具层
│   ├── __init__.py
│   ├── heatmap.py                  # 热力图渲染器（核心）
│   └── plot_utils.py               # 通用绘图工具
│
├── data_manager/                  # 模块6: 模型加载与数据管理
│   ├── __init__.py
│   ├── model_loader.py            # 模型加载器（含forward签名检测）
│   ├── data_loader.py             # 数据加载器
│   ├── preprocessor.py            # 输入预处理
│   └── input_adapter.py           # 多输入模型适配器（新增）
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── tensor_utils.py             # 张量操作工具
│   ├── memory_utils.py             # 内存管理工具
│   └── io_utils.py                 # IO操作工具
│
├── tests/                          # 测试模块
│   ├── __init__.py
│   └── test_model/
│
├── examples/                       # 示例代码
│   ├── basic_usage.py
│   ├── swin_analysis.py
│   └── moe_analysis.py
│
├── requirements.txt
└── setup.py
```

---

## 3. 模块详细设计

### 3.1 core/ - 核心基础模块

#### 3.1.1 config.py - 全局配置管理

```python
class Config:
    """全局配置管理类"""
    
    # 类属性
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MAX_BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 4096
    ACCUMULATOR_LIMIT: int = 100000
    DEFAULT_PERCENTILE_LOW: float = 0.01
    DEFAULT_PERCENTILE_HIGH: float = 0.99
    DEFAULT_FUSION_ALPHA: float = 0.5
    DEFAULT_ATTENTION_THRESHOLD: float = 0.3
    
    # 精度管理
    PRECISION: str = "fp32"                    # 默认计算精度 ("fp32" | "fp16")
    MIN_GPU_MEMORY_FP16: float = 8.0           # FP16最低显存要求（GB）
    MAX_IMAGE_SIZE: int = 1024                 # 最大图像尺寸
    MIN_IMAGE_SIZE: int = 224                  # 最小图像尺寸
    
    # 实例方法
    def __init__(self, config_path: Optional[str] = None) -> None:
        """初始化配置，支持从文件加载"""
        pass
    
    def load_from_file(self, path: str) -> None:
        """从YAML/JSON文件加载配置"""
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        pass
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        pass
    
    def validate(self) -> bool:
        """
        验证配置合法性
        
        校验规则：
        - PRECISION 必须是 "fp32" 或 "fp16"
        - 当 PRECISION="fp16" 时，检查GPU显存是否 >= MIN_GPU_MEMORY_FP16
        - MAX_IMAGE_SIZE 必须 >= MIN_IMAGE_SIZE
        - MAX_BATCH_SIZE 必须 >= 1
        - MAX_SEQUENCE_LENGTH 必须 >= 1
        - ACCUMULATOR_LIMIT 必须 >= 1
        - DEFAULT_FUSION_ALPHA 必须在 [0, 1] 范围内
        
        Returns:
            bool: 配置是否合法
            
        Raises:
            InvalidInputError: 配置不合法时抛出
        """
        pass
```

#### 3.1.2 types.py - 类型定义与枚举

```python
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Optional, Callable
import torch
import numpy as np

# 类型别名
Tensor = torch.Tensor
NDArray = np.ndarray
AttentionMap = Tensor  # 注意力矩阵 (B, H, N, N) 或窗口格式
GradientMap = Tensor   # 梯度矩阵
Heatmap = NDArray      # 热力图 (H, W)

class ModelArchitecture(Enum):
    """支持的模型架构枚举"""
    TRANSFORMER = auto()
    MOE_TRANSFORMER = auto()
    SWIN = auto()
    VIT = auto()

class Quadrant(Enum):
    """四象限枚举"""
    CORE_DISCRIMINATIVE = auto()      # 核心判别区
    REDUNDANT_ATTENTION = auto()      # 冗余关注区
    POTENTIAL_INFLUENCE = auto()      # 潜在影响区
    IRRELEVANT = auto()               # 无关区域

class FusionStrategy(Enum):
    """融合策略枚举（精简版，仅保留常用策略）"""
    WEIGHTED_SUM = auto()             # 加权求和：α×注意力 + (1-α)×梯度
    GRADCAM = auto()                  # GradCAM式融合：注意力 × 梯度

class HookType(Enum):
    """Hook类型枚举"""
    ATTENTION = auto()
    MOE_ROUTER = auto()
    HIDDEN_STATE = auto()

@dataclass
class ModelInfo:
    """模型信息数据类"""
    architecture: ModelArchitecture
    num_layers: int
    num_heads: int
    patch_size: int
    hidden_dim: int
    window_size: Optional[int] = None  # Swin特有
    num_experts: Optional[int] = None  # MoE特有

@dataclass
class AccumulatorState:
    """累积器状态数据类"""
    head_activation_freq: Tensor       # (num_layers, num_heads)
    head_attention_concentration: Tensor  # (num_layers, num_heads)
    layer_gradient_norm: Tensor        # (num_layers,)
    attention_gradient_norm: Optional[Tensor] = None  # (num_layers, num_heads) 注意力梯度范数
    expert_selection_count: Optional[Tensor] = None  # (num_experts,)
    sample_count: int = 0

@dataclass
class HeadClassification:
    """头分类结果"""
    layer_idx: int
    head_idx: int
    activation_freq: float
    concentration: float
    importance_score: float
    category: str  # "high_freq_high_focus" | "high_freq_low_focus" | "low_freq"

@dataclass
class DiagnosisReport:
    """全局诊断报告"""
    activation_frequency_ranking: List[Dict[str, Any]]
    gradient_importance_ranking: Dict[str, List[Dict[str, Any]]]
    anomaly_analysis: Dict[str, Any]
    head_classification: Dict[str, List[HeadClassification]]

@dataclass
class AnalysisConfig:
    """分析配置"""
    model_path: str
    data_path: str
    output_dir: str = "./results"
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 16
    max_samples: Optional[int] = None
```

#### 3.1.3 exceptions.py - 自定义异常类

```python
class AnalyzerException(Exception):
    """基础异常类"""
    pass

class ArchitectureNotSupportedError(AnalyzerException):
    """不支持的架构异常"""
    pass

class HookRegistrationError(AnalyzerException):
    """Hook注册失败异常"""
    pass

class AccumulatorOverflowError(AnalyzerException):
    """累积器溢出异常"""
    pass

class InvalidInputError(AnalyzerException):
    """非法输入异常"""
    pass

class FusionError(AnalyzerException):
    """融合计算异常"""
    pass
```

---

### 3.2 model_adapter/ - 模块1: 模型适配与全局Hook层

#### 3.2.1 detector.py - 架构探测

> **设计说明**：启发式架构探测对多模态/混合模型存在误判风险（例如将包含窗口卷积的 ECG 模型误判为 Swin）。因此提供 `override` 参数允许用户手动纠正，并引入 `confidence` 置信度机制：低置信度时自动打印警告，建议用户手动确认或使用 override。

```python
@dataclass
class DetectionResult:
    """架构探测结果"""
    model_info: ModelInfo
    confidence: float          # 置信度 [0.0, 1.0]
    warnings: List[str]        # 低置信度时的警告信息

class ArchitectureDetector:
    """架构探测器：自动识别模型架构并提取参数"""
    
    CONFIDENCE_WARNING_THRESHOLD: float = 0.6  # 低于此值弹出警告
    
    def __init__(self) -> None:
        """初始化探测器"""
        pass
    
    def detect(self, model: nn.Module, 
               override: Optional[Dict[str, Any]] = None) -> ModelInfo:
        """
        自动识别模型架构
        
        Args:
            model: PyTorch模型实例
            override: 手动覆盖字典（可选）。支持键：
                - "architecture": ModelArchitecture 枚举值
                - "num_layers": int
                - "num_heads": int
                - "patch_size": int
                - "hidden_dim": int
                - "window_size": int（Swin特有）
                - "num_experts": int（MoE特有）
                
        Returns:
            ModelInfo: 模型信息对象
            
        Raises:
            ArchitectureNotSupportedError: 当架构不支持时
            
        Note:
            探测结枚会附带 confidence 置信度。当置信度 < CONFIDENCE_WARNING_THRESHOLD 时
            自动打印警告，建议用户通过 override 参数手动确认。
        """
        pass
    
    def detect_with_confidence(self, model: nn.Module) -> DetectionResult:
        """
        架构探测并返回置信度
        
        Returns:
            DetectionResult: 包含 model_info, confidence, warnings 的探测结枚
        """
        pass
    
    def _detect_vit(self, model: nn.Module) -> ModelInfo:
        """识别ViT架构"""
        pass
    
    def _detect_swin(self, model: nn.Module) -> ModelInfo:
        """识别Swin Transformer架构"""
        pass
    
    def _detect_transformer(self, model: nn.Module) -> ModelInfo:
        """识别标准Transformer架构"""
        pass
    
    def _detect_moe(self, model: nn.Module) -> ModelInfo:
        """识别MoE-Transformer架构"""
        pass
    
    def _compute_confidence(self, model: nn.Module, detected: ModelArchitecture) -> float:
        """
        计算探测结枚的置信度
        
        策略：根据匹配特征的数量和歧义模块重叠程度计算置信度。
        多模态/混合模型通常会导致置信度下降。
        """
        pass
    
    def extract_parameters(self, model: nn.Module, arch: ModelArchitecture) -> Dict[str, Any]:
        """提取模型参数（层数、头数等）"""
        pass
```

#### 3.2.2 hooks.py - Hook注册与管理

> **设计说明**：PyTorch `nn.TransformerEncoderLayer` 默认 `need_weights=False`，导致标准 forward Hook 捕获的注意力权重为 `None`。`register_all_hooks()` 应**自动检测**此类层，并对其 `self_attn` 注册补丁 Hook，通过 `F.multi_head_attention_forward` 底层 API 重新计算并捕获注意力权重。此行为是 HookManager 的**内置自动行为**，用户无需手动处理。

```python
class HookManager:
    """Hook管理器：统一注册和管理所有Hook"""
    
    def __init__(self, model: nn.Module, model_info: ModelInfo, 
                 use_data_parallel: bool = False) -> None:
        """
        初始化Hook管理器
        
        Args:
            model: 目标模型（支持DataParallel包装的模型）
            model_info: 模型信息
            use_data_parallel: 是否使用DataParallel模式
        """
        pass
    
    def register_all_hooks(self) -> Dict[str, Callable]:
        """
        注册全套数据捕获探针
        
        内置行为：
        1. 遍历模型所有子模块
        2. 若检测到 nn.TransformerEncoderLayer，则对其 self_attn 注册
           补丁 Hook：在 Hook 内部调用 F.multi_head_attention_forward
           并强制 need_weights=True，捕获真实的注意力权重矩阵。
        3. 对其他支持标准输出格式的注意力层，注册常规 forward Hook。
        
        Returns:
            Dict: 存储句柄的字典，用于后续数据提取
        """
        pass
    
    def _register_transformer_encoder_patch(self, layer: nn.Module, layer_idx: int) -> Callable:
        """
        针对 nn.TransformerEncoderLayer 的补丁 Hook
        
        原理：绕过 self_attn 的 forward 默认行为，直接调用
        F.multi_head_attention_forward 并设置 need_weights=True，
        从而获取注意力权重矩阵 (B, num_heads, seq_len, seq_len)。
        
        Args:
            layer: TransformerEncoderLayer 实例
            layer_idx: 层索引
            
        Returns:
            Callable: Hook句柄
        """
        pass
    
    def register_attention_hook(self, layer_idx: int, head_idx: int) -> Callable:
        """
        注册标准注意力Hook
        
        Args:
            layer_idx: 层索引
            head_idx: 头索引
            
        Returns:
            Callable: Hook句柄
        """
        pass
    
    def register_hidden_state_hook(self, layer_idx: int) -> Callable:
        """注册隐藏状态Hook"""
        pass
    
    def remove_all_hooks(self) -> None:
        """移除所有注册的Hook"""
        pass
    
    def get_attention_output(self, layer_idx: int) -> Tensor:
        """获取指定层的注意力输出"""
        pass
    
    def get_hidden_state(self, layer_idx: int) -> Tensor:
        """获取指定层的隐藏状态"""
        pass

class AttentionHook:
    """注意力Hook实现类"""
    
    def __init__(self, storage: Dict, key: str) -> None:
        """初始化Hook"""
        pass
    
    def __call__(self, module: nn.Module, input: Tuple, output: Tensor) -> None:
        """Hook回调函数"""
        pass
    
    def normalize_attention(self, attention: Tensor, arch: ModelArchitecture) -> Tensor:
        """将不同架构的注意力输出统一为标准张量"""
        pass
```

#### 3.2.3 swin_handler.py - Swin特殊处理

```python
class SwinHandler:
    """Swin Transformer特殊处理器"""
    
    def __init__(self, window_size: int, num_stages: int) -> None:
        """
        初始化Swin处理器
        
        Args:
            window_size: 窗口大小
            num_stages: stage数量
        """
        pass
    
    def extract_window_attention(self, attention: Tensor, shift_size: int) -> Tensor:
        """
        提取窗口内注意力矩阵
        
        Args:
            attention: 原始注意力输出
            shift_size: 窗口位移大小
            
        Returns:
            Tensor: 格式为 (B, num_heads, num_windows, window_size^2, window_size^2)
        """
        pass
    
    def mark_window_shift(self, layer_idx: int) -> bool:
        """
        标记当前层是否使用window_shift
        
        Args:
            layer_idx: 层索引
            
        Returns:
            bool: 是否位移
        """
        pass
    
    def adapt_stage_resolution(self, stage_idx: int, input_h: int, input_w: int) -> Tuple[int, int]:
        """
        适配各stage的特征图分辨率
        
        Args:
            stage_idx: stage索引
            input_h: 输入高度
            input_w: 输入宽度
            
        Returns:
            Tuple[int, int]: 当前stage的H, W
        """
        pass
    
    def merge_window_attention(self, window_attn: Tensor, num_windows_h: int, num_windows_w: int) -> Tensor:
        """合并窗口注意力为全局格式"""
        pass
```

#### 3.2.4 moe_handler.py - MoE路由处理

```python
class MoEHandler:
    """MoE-Transformer路由处理器"""
    
    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        """
        初始化MoE处理器
        
        Args:
            num_experts: 专家数量
            top_k: 每个token选择的专家数
        """
        pass
    
    def register_router_hook(self, model: nn.Module) -> Callable:
        """
        在路由门控层注册Hook
        
        Returns:
            Callable: Hook句柄
        """
        pass
    
    def capture_expert_assignment(self, router_output: Tensor) -> Tensor:
        """
        捕获每个Token/Patch被分配的专家索引
        
        Args:
            router_output: 路由门控输出
            
        Returns:
            Tensor: 专家索引 (B, L, top_k)
        """
        pass
    
    def get_expert_load_distribution(self) -> Tensor:
        """获取各专家的负载分布"""
        pass
    
    def compute_load_balance_loss(self, expert_counts: Tensor) -> Tensor:
        """计算负载均衡损失（用于评估）"""
        pass
```

---

### 3.3 tracker/ - 模块2: 动态追踪与跨样本累积层

#### 3.3.1 forward_tracker.py - 前向传播追踪

```python
class ForwardTracker:
    """前向传播追踪器"""
    
    def __init__(self, hook_manager: HookManager) -> None:
        """初始化前向追踪器"""
        pass
    
    def track(self, model: nn.Module, input_data: Tensor) -> Dict[str, Tensor]:
        """
        执行前向传播并提取注意力矩阵
        
        Args:
            model: 模型实例
            input_data: 输入数据 (B, C, H, W) 或 (B, L, D)
            
        Returns:
            Dict: 包含各层注意力矩阵的字典
        """
        pass
    
    def extract_attention_matrices(self) -> Dict[int, Tensor]:
        """提取所有层的注意力矩阵"""
        pass
    
    def extract_hidden_states(self) -> Dict[int, Tensor]:
        """提取所有层的隐藏状态"""
        pass
```

#### 3.3.2 backward_tracker.py - 反向传播追踪

```python
class BackwardTracker:
    """反向传播追踪器"""
    
    def __init__(self, model: nn.Module, hook_manager: HookManager) -> None:
        """
        初始化反向追踪器
        
        Args:
            model: 模型实例
            hook_manager: Hook管理器，用于获取前向阶段的中间层信息
        """
        pass
    
    def track(self, loss: Tensor) -> Dict[str, Tensor]:
        """
        执行反向传播并计算梯度
        
        Args:
            loss: 损失张量
            
        Returns:
            Dict: 包含各梯度的字典
        """
        pass
    
    def compute_input_gradient(self, input_data: Tensor) -> Tensor:
        """计算输入梯度"""
        pass
    
    def compute_hidden_gradient(self, layer_idx: int) -> Tensor:
        """
        计算隐藏状态梯度（用于层重要性）
        
        Returns:
            Tensor: 梯度张量
        """
        pass
    
    def compute_attention_gradient(self, layer_idx: int, head_idx: int) -> Tensor:
        """
        计算注意力权重梯度（用于头重要性）
        
        Returns:
            Tensor: 梯度张量
        """
        pass
    
    def aggregate_to_patch_level(self, gradients: Tensor, patch_size: int) -> Tensor:
        """将梯度聚合为Patch级向量"""
        pass
```

#### 3.3.3 accumulator.py - 跨样本累积器

```python
class CrossSampleAccumulator:
    """跨样本持久化累积器"""
    
    def __init__(self, model_info: ModelInfo, limit: int = 100000) -> None:
        """
        初始化累积器
        
        Args:
            model_info: 模型信息
            limit: 样本上限
        """
        pass
    
    def update(self, 
               attention_maps: Dict[int, Tensor],
               input_gradients: Tensor,
               hidden_gradients: Dict[int, Tensor],
               attention_gradients: Optional[Dict[Tuple[int, int], Tensor]] = None,
               expert_assignments: Optional[Tensor] = None) -> None:
        """
        更新累积器状态
        
        Args:
            attention_maps: 注意力矩阵字典 {layer_idx: tensor}
            input_gradients: 输入梯度张量 (B, C, H, W) 或 (B, L, D)
            hidden_gradients: 隐藏状态梯度字典 {layer_idx: tensor}
            attention_gradients: 注意力梯度字典 {(layer_idx, head_idx): tensor}
            expert_assignments: 专家分配索引 (MoE)
        """
        pass
    
    def _update_head_activation_freq(self, attention_maps: Dict[int, Tensor]) -> None:
        """更新头的激活频率"""
        pass
    
    def _update_head_concentration(self, attention_maps: Dict[int, Tensor]) -> None:
        """更新头的注意力集中度"""
        pass
    
    def _update_layer_gradient_norm(self, hidden_gradients: Dict[int, Tensor]) -> None:
        """更新层的梯度L2范数"""
        pass
    
    def _update_expert_count(self, expert_assignments: Tensor) -> None:
        """更新专家选中计数"""
        pass
    
    def reset(self) -> None:
        """清空累积器"""
        pass
    
    def save(self, path: str) -> None:
        """持久化统计状态到磁盘"""
        pass
    
    def load(self, path: str) -> None:
        """从磁盘加载统计状态"""
        pass
    
    def get_statistics(self) -> AccumulatorState:
        """获取当前统计状态"""
        pass
    
    def is_full(self) -> bool:
        """检查是否达到样本上限"""
        pass
```

#### 3.3.4 metrics.py - 指标计算

```python
class MetricsCalculator:
    """指标计算工具类"""
    
    @staticmethod
    def compute_attention_entropy(attention: Tensor) -> Tensor:
        """
        计算注意力熵
        
        Args:
            attention: 注意力矩阵 (..., seq_len)
            
        Returns:
            Tensor: 熵值 (...)
        """
        pass
    
    @staticmethod
    def compute_attention_concentration(attention: Tensor) -> Tensor:
        """
        计算注意力集中度 (1 - 归一化熵)
        
        Args:
            attention: 注意力矩阵
            
        Returns:
            Tensor: 集中度值，范围[0, 1]
        """
        pass
    
    @staticmethod
    def compute_l2_norm(tensor: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
        """
        计算L2范数
        
        Args:
            tensor: 输入张量
            dim: 求范数的维度
            
        Returns:
            Tensor: L2范数
        """
        pass
    
    @staticmethod
    def compute_activation_frequency(attention_history: List[Tensor], threshold: float = 1e-6) -> Tensor:
        """
        计算激活频率
        
        Args:
            attention_history: 历史注意力列表
            threshold: 非零阈值
            
        Returns:
            Tensor: 激活频率
        """
        pass
```

---

### 3.4 spatial/ - 模块3: 空间重构与数值对齐层（可选模块 — 仅图像输入场景需要）

> **适用范围**：此模块的核心假设是注意力矩阵对应 2D 图像 Patch 网格，需要从 `(num_patches_h, num_patches_w)` 重构回 `(H, W)` 像素空间。
> - **启用条件**：输入为图像且模型使用 Patch Embedding（ViT、Swin等）。
> - **跳过条件**：输入为 1D 序列（ECG、NLP等）时，注意力矩阵本身就是最终结枚，管道应自动跳过此模块。可通过 `pipeline.py` 的 `skip_spatial=True` 强制跳过。

#### 3.4.1 reshaper.py - 粒度转换

```python
class SpatialReshaper:
    """空间重构器"""
    
    def __init__(self, patch_size: int, image_size: Tuple[int, int],
                 architecture: ModelArchitecture = ModelArchitecture.VIT,
                 num_stages: Optional[int] = None) -> None:
        """
        初始化空间重构器
        
        Args:
            patch_size: Patch大小
            image_size: 原始图像尺寸 (H, W)
            architecture: 模型架构类型，用于处理Swin等特殊架构
            num_stages: stage数量（Swin架构必填）
        """
        pass
    
    def patch_to_grid(self, patch_vector: Tensor, num_patches_h: int, num_patches_w: int) -> Tensor:
        """
        将Patch级一维向量重塑为二维网格
        
        Args:
            patch_vector: (B, num_patches) 或 (num_patches,)
            num_patches_h: Patch网格高度
            num_patches_w: Patch网格宽度
            
        Returns:
            Tensor: (B, num_patches_h, num_patches_w) 或 (num_patches_h, num_patches_w)
        """
        pass
    
    def upsample_to_image(self, grid: Tensor, method: str = "bilinear") -> Tensor:
        """
        上采样至原图像素级尺寸
        
        Args:
            grid: 二维网格 (H, W)
            method: 插值方法 ("bilinear" | "gaussian")
            
        Returns:
            Tensor: 上采样后的图像 (image_h, image_w)
        """
        pass
    
    def swin_window_reorganize(self, 
                              window_attention: Tensor,
                              stage_idx: int,
                              feature_h: int,
                              feature_w: int,
                              window_size: int,
                              shift_size: int = 0) -> Tensor:
        """
        将Swin窗口注意力重组为全局格式
        
        Args:
            window_attention: 窗口内注意力 (num_windows, window_size^2, window_size^2)
            stage_idx: stage索引
            feature_h: 当前stage的特征图高度
            feature_w: 当前stage的特征图宽度
            window_size: 窗口大小
            shift_size: 窗口位移大小（0表示无位移）
            
        Returns:
            Tensor: (feature_h, feature_w) 格式的全局注意力图
        """
        pass
```

#### 3.4.2 normalizer.py - 尺度对齐

```python
class Normalizer:
    """尺度对齐器"""
    
    def __init__(self, low_percentile: float = 0.01, high_percentile: float = 0.99) -> None:
        """
        初始化对齐器
        
        Args:
            low_percentile: 下百分位裁剪点
            high_percentile: 上百分位裁剪点
        """
        pass
    
    def percentile_clip(self, tensor: Tensor) -> Tensor:
        """
        百分位裁剪
        
        Args:
            tensor: 输入张量
            
        Returns:
            Tensor: 裁剪后的张量
        """
        pass
    
    def min_max_normalize(self, tensor: Tensor, target_range: Tuple[float, float] = (0.0, 1.0)) -> Tensor:
        """
        Min-Max归一化到目标范围
        
        Args:
            tensor: 输入张量
            target_range: 目标值域
            
        Returns:
            Tensor: 归一化后的张量
        """
        pass
    
    def normalize_for_visualization(self, tensor: Tensor) -> Tensor:
        """
        完整的可视化前归一化流程（裁剪 + Min-Max）
        
        Args:
            tensor: 原始注意力或梯度张量
            
        Returns:
            Tensor: [0, 1]范围内的归一化张量
        """
        pass
    
    def z_score_normalize(self, tensor: Tensor) -> Tensor:
        """Z-Score标准化"""
        pass
```

#### 3.4.3 interpolator.py - 插值处理

```python
class Interpolator:
    """插值处理器"""
    
    def __init__(self) -> None:
        """初始化插值器"""
        pass
    
    def bilinear_interpolate(self, 
                            input_tensor: Tensor, 
                            target_size: Tuple[int, int]) -> Tensor:
        """
        双线性插值
        
        Args:
            input_tensor: 输入张量 (..., H, W)
            target_size: 目标尺寸 (target_h, target_w)
            
        Returns:
            Tensor: 插值后的张量
        """
        pass
    
    def gaussian_smooth(self, tensor: Tensor, sigma: float = 1.0, kernel_size: int = 5) -> Tensor:
        """
        高斯模糊平滑
        
        Args:
            tensor: 输入张量
            sigma: 高斯核标准差
            kernel_size: 核大小
            
        Returns:
            Tensor: 平滑后的张量
        """
        pass
```

---

### 3.5 analyzer/ - 模块4: 单样本解释与全局权重诊断引擎

> **设计说明**：融合功能从原 `fusion/` 独立模块并入本模块，简化为 `fusion_utils.py` 工具方法。去除策略模式和工厂模式，只保留最基础的加权融合和 GradCAM 式融合。对于 ECG 等非图像模型，注意力矩阵本身就是最终结枚，融合不是必需步骤。

#### 3.5.1 single_sample.py - 单样本解释（轨道A）

```python
class SingleSampleAnalyzer:
    """单样本解释器"""
    
    def __init__(self, num_layers: int, num_heads: int,
                 threshold_method: str = "median",
                 attention_threshold: Optional[float] = None,
                 gradient_threshold: Optional[float] = None) -> None:
        """
        初始化单样本分析器
        
        Args:
            num_layers: 层数
            num_heads: 头数
            threshold_method: 阈值计算方法 ("median" | "mean" | "otsu")
            attention_threshold: 自定义注意力阈值（None则自动计算）
            gradient_threshold: 自定义梯度阈值（None则自动计算）
        """
        pass
    
    def analyze(self, 
                attention_maps: Dict[int, Tensor],
                gradient_maps: Dict[int, Tensor],
                normalized_data: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        执行单样本全面分析
        
        Args:
            attention_maps: 注意力图字典 {layer_idx: tensor}
            gradient_maps: 梯度图字典 {layer_idx: tensor}
            normalized_data: 归一化后的数据
            
        Returns:
            Dict: 分析结果
        """
        pass
    
    def split_by_depth(self, data: Dict[int, Tensor]) -> Dict[str, Dict[int, Tensor]]:
        """
        按深度切分为浅/中/深层
        
        Returns:
            Dict: {"shallow": {...}, "middle": {...}, "deep": {...}}
        """
        pass
    
    def cluster_attention_heads(self, attention_maps: Dict[int, Tensor]) -> Dict[int, int]:
        """
        按注意力头进行聚类
        
        Returns:
            Dict: {(layer_idx, head_idx): cluster_id}
        """
        pass
    
    def compute_layer_importance(self, gradient_maps: Dict[int, Tensor]) -> Dict[int, float]:
        """计算各层重要性得分"""
        pass
```

#### 3.5.2 quadrant.py - 四象限划分

```python
class QuadrantAnalyzer:
    """四象限分析器"""
    
    def __init__(self, threshold_method: str = "median") -> None:
        """
        初始化四象限分析器
        
        Args:
            threshold_method: 阈值计算方法 ("median" | "mean" | "otsu")
        """
        pass
    
    def compute_threshold(self, attention: Tensor, gradient: Tensor) -> Tuple[float, float]:
        """
        计算注意力值和梯度值的阈值
        
        Args:
            attention: 注意力张量
            gradient: 梯度张量
            
        Returns:
            Tuple[float, float]: (attn_threshold, grad_threshold)
        """
        pass
    
    def classify_quadrant(self, 
                         attention_value: float, 
                         gradient_value: float,
                         attn_threshold: float,
                         grad_threshold: float) -> Quadrant:
        """
        根据注意力值和梯度值划分象限
        
        Args:
            attention_value: 注意力值
            gradient_value: 梯度值
            attn_threshold: 注意力阈值
            grad_threshold: 梯度阈值
            
        Returns:
            Quadrant: 象限枚举值
        """
        pass
    
    def generate_quadrant_map(self, 
                             attention_map: Tensor, 
                             gradient_map: Tensor) -> Tensor:
        """
        生成四象限分类图
        
        Args:
            attention_map: 注意力热力图 (H, W)
            gradient_map: 梯度热力图 (H, W)
            
        Returns:
            Tensor: 象限分类图 (H, W)，每个像素值为Quadrant枚举
        """
        pass
    
    def compute_quadrant_statistics(self, quadrant_map: Tensor) -> Dict[Quadrant, float]:
        """计算各象限占比统计"""
        pass
```

#### 3.5.3 global_diagnosis.py - 全局权重诊断（轨道B）

```python
class GlobalDiagnosisEngine:
    """全局权重诊断引擎"""
    
    def __init__(self, accumulator: CrossSampleAccumulator) -> None:
        """
        初始化诊断引擎
        
        Args:
            accumulator: 跨样本累积器
        """
        pass
    
    def diagnose(self) -> DiagnosisReport:
        """
        执行全局权重诊断
        
        Returns:
            DiagnosisReport: 诊断报告，包含：
                - activation_frequency_ranking: 基于激活频率和集中度的头排名
                - gradient_importance_ranking: 基于梯度范数的层/头重要性排名
                - anomaly_analysis: 异常模式识别结果（高频低效头、低频高效头、MoE负载偏斜）
                - head_classification: 头分类结果（高频高聚焦/高频低聚焦/低频）
        """
        pass
    
    def rank_by_activation_frequency(self) -> List[Dict[str, Any]]:
        """
        基于激活频率和集中度对头进行排名
        
        Returns:
            List: 排名列表，每项包含layer_idx, head_idx, freq, concentration
        """
        pass
    
    def rank_by_gradient_importance(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        基于梯度范数进行重要性排名
        
        Returns:
            Dict: {"layer_ranking": [...], "head_ranking": [...]}
        """
        pass
    
    def categorize_heads_by_frequency(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        按使用频率和集中度对头分类
        
        Returns:
            Dict: {
                "high_freq_high_focus": [...],  # 高频高聚焦头
                "high_freq_low_focus": [...],   # 高频低聚焦头（均匀头）
                "low_freq": [...]                # 低频头
            }
        """
        pass
```

#### 3.5.4 anomaly_detector.py - 异常识别

```python
class AnomalyDetector:
    """异常识别器"""
    
    def __init__(self, freq_threshold: float = 0.5, importance_threshold: float = 0.5) -> None:
        """
        初始化异常检测器
        
        Args:
            freq_threshold: 频率阈值
            importance_threshold: 重要性阈值
        """
        pass
    
    def detect_redundant_heads(self, 
                              frequency_ranking: List[Dict],
                              importance_ranking: List[Dict]) -> List[Tuple[int, int]]:
        """
        识别"高频低效"头
        
        Args:
            frequency_ranking: 频率排名
            importance_ranking: 重要性排名
            
        Returns:
            List[Tuple[int, int]]: 冗余头列表 [(layer_idx, head_idx), ...]
        """
        pass
    
    def detect_sparse_critical_heads(self,
                                     frequency_ranking: List[Dict],
                                     importance_ranking: List[Dict]) -> List[Tuple[int, int]]:
        """
        识别"低频高效"头
        
        Returns:
            List[Tuple[int, int]]: 稀疏关键头列表
        """
        pass
    
    def detect_moe_load_imbalance(self, expert_counts: Tensor, imbalance_threshold: float = 0.3) -> Dict[str, Any]:
        """
        检测MoE专家负载偏斜
        
        Args:
            expert_counts: 各专家被选中计数
            imbalance_threshold: 偏斜判定阈值
            
        Returns:
            Dict: {
                "is_imbalanced": bool,
                "skewness": float,  # 偏斜度指标
                "overloaded_experts": List[int],
                "underloaded_experts": List[int]
            }
        """
        pass
    
    def compute_skewness(self, distribution: Tensor) -> float:
        """计算分布偏斜度"""
        pass
```

#### 3.5.5 fusion_utils.py - 融合工具（简化版）

> **设计说明**：原 `fusion/` 模块并入此处，去掉策略模式、抽象基类和工厂模式。只提供两种基础融合方法作为工具函数。
> - 对于 ECG/NLP 等非图像模型，融合不是必需步骤，可直接使用注意力矩阵作为最终结枚。
> - 对于图像模型，融合可用于将注意力与梯度信息结合产生更全面的重要性图。

```python
def weighted_sum_fusion(attention: Tensor, gradient: Tensor, alpha: float = 0.5) -> Tensor:
    """
    加权求和融合
    
    Args:
        attention: 归一化到 [0,1] 的注意力张量
        gradient: 归一化到 [0,1] 的梯度张量
        alpha: 注意力权重，(1-alpha) 为梯度权重，默认 0.5
        
    Returns:
        Tensor: 融合后的重要性分数图 [0,1]
    """
    pass

def gradcam_fusion(attention: Tensor, gradient: Tensor) -> Tensor:
    """
    GradCAM式融合：注意力 × 梯度
    
    仅当两者均高时输出高权重。适合需要“联合确认”场景。
    
    Args:
        attention: 归一化到 [0,1] 的注意力张量
        gradient: 归一化到 [0,1] 的梯度张量
        
    Returns:
        Tensor: 融合后的重要性分数图 [0,1]
    """
    pass

def normalize_for_fusion(tensor: Tensor,
                         low_percentile: float = 0.01,
                         high_percentile: float = 0.99) -> Tensor:
    """
    融合前归一化：百分位裁剪 + Min-Max映射到 [0,1]
    
    Args:
        tensor: 原始张量
        low_percentile: 下百分位展刻点
        high_percentile: 上百分位展刻点
        
    Returns:
        Tensor: [0,1] 范围内的归一化张量
    """
    pass
```

---

### 3.6 visualization/ - 模块5: 可视化工具层

> **设计说明**：经测试验证，50 行 matplotlib 代码就可完成可视化需求。`DashboardGenerator`（交互式面板）和 `ReportExporter`（HTML 汇总报告）属于过早优化，标注为未来扩展。当前简化为只包含 `HeatmapRenderer`（热力图渲染器）和 `plot_utils.py`（通用绘图工具）。

#### 3.6.1 heatmap.py - 热力图渲染器

```python
class HeatmapRenderer:
    """热力图渲染器（核心可视化组件）"""
    
    def __init__(self, colormap: str = "jet") -> None:
        """
        初始化渲染器
        
        Args:
            colormap: 颜色映射方案，默认 "jet"
        """
        pass
    
    def render_attention(self,
                         attention_map: Tensor,
                         title: str = "",
                         save_path: Optional[str] = None) -> NDArray:
        """
        渲染注意力热力图
        
        Args:
            attention_map: 注意力张量（支持 1D 序列或 2D 图像格式）
            title: 图表标题
            save_path: 保存路径（None 则不保存）
            
        Returns:
            NDArray: 热力图 RGB 数组
        """
        pass
    
    def render_gradient(self,
                        gradient_map: Tensor,
                        title: str = "",
                        save_path: Optional[str] = None) -> NDArray:
        """渲染梯度热力图"""
        pass
    
    def render_multi_layer(self,
                           attention_dict: Dict[int, Tensor],
                           num_cols: int = 4,
                           save_path: Optional[str] = None) -> NDArray:
        """
        渲染多层注意力热力图面板
        
        Args:
            attention_dict: {layer_idx: attention_tensor} 字典
            num_cols: 面板列数
            save_path: 保存路径
            
        Returns:
            NDArray: 拼接后的面板图像
        """
        pass
    
    def overlay_on_signal(self,
                          heatmap: NDArray,
                          signal: NDArray,
                          alpha: float = 0.4,
                          save_path: Optional[str] = None) -> NDArray:
        """
        将热力图叠加到原始信号/图像上
        
        Args:
            heatmap: 热力图（支持 1D 时序信号或 2D 图像）
            signal: 原始信号
            alpha: 热力图透明度
            save_path: 保存路径
            
        Returns:
            NDArray: 叠加后的图像
        """
        pass
```

#### 3.6.2 plot_utils.py - 通用绘图工具

```python
def plot_layer_importance(layer_importance: Dict[int, float],
                          title: str = "Layer Importance",
                          save_path: Optional[str] = None) -> Figure:
    """
    绘制层重要性柱状图
    
    Args:
        layer_importance: {layer_idx: importance_score}
        title: 图表标题
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib 图表对象
    """
    pass

def plot_head_scatter(frequency_data: List[Dict],
                      importance_data: List[Dict],
                      save_path: Optional[str] = None) -> Figure:
    """
    绘制注意力头“频率-重要性”二维散点图
    
    Args:
        frequency_data: 频率数据列表
        importance_data: 重要性数据列表
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib 图表对象
    """
    pass

def plot_accumulator_stats(accumulator_state: AccumulatorState,
                           save_path: Optional[str] = None) -> Figure:
    """
    绘制累积器统计摘要图（激活频率热力矩阵 + 梯度范数柱状图）
    
    Args:
        accumulator_state: 累积器状态对象
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib 图表对象
    """
    pass

def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """保存 matplotlib 图表到磁盘"""
    pass
```

> **未来扩展**（当前不实现）：
> - `DashboardGenerator`：基于 Plotly 的交互式汇总面板
> - `ReportExporter`：自动生成 HTML/PDF 综合分析报告

---

### 3.7 data_manager/ - 模块6: 模型加载与数据管理

#### 3.7.1 model_loader.py - 模型加载器

```python
class ModelLoader:
    """模型加载器：支持从多种来源加载模型"""
    
    def __init__(self, device: str = "auto", precision: str = "fp32") -> None:
        """
        初始化模型加载器
        
        Args:
            device: 设备类型 ("auto" | "cpu" | "cuda" | "cuda:0"等)
            precision: 计算精度 ("fp32" | "fp16")
        """
        pass
    
    def load_from_checkpoint(self, checkpoint_path: str, model_class: Optional[type] = None) -> nn.Module:
        """
        从.pth/.pt文件加载模型
        
        Args:
            checkpoint_path: 检查点文件路径
            model_class: 模型类（可选）
            
        Returns:
            nn.Module: 加载的模型实例
        """
        pass
    
    def load_from_huggingface(self, model_name: str) -> nn.Module:
        """从 HuggingFace 加载预训练模型"""
        pass
    
    def load_from_timm(self, model_name: str) -> nn.Module:
        """从 timm 库加载预训练模型"""
        pass
    
    def inspect_forward_signature(self, model: nn.Module) -> Dict[str, Any]:
        """
        检测模型 forward 函数的参数签名
        
        返回参数名、默认値信息，用于判断模型是否需要多输入适配器。
        
        Returns:
            Dict: {"param_names": [...], "required": [...], "has_defaults": [...]}
        """
        pass
    
    def _setup_device(self, model: nn.Module) -> nn.Module:
        """设置模型运行设备（CPU/GPU）"""
        pass
    
    def _setup_precision(self, model: nn.Module) -> nn.Module:
        """设置模型计算精度（FP32/FP16）"""
        pass
    
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """处理 DataParallel 等包装器，获取原始模型"""
        pass
```

#### 3.7.2 data_loader.py - 数据管理器

```python
class DataManager:
    """数据管理器：负责数据加载和预处理"""
    
    def __init__(self, config: Config) -> None:
        pass
    
    def load_image_dataset(self, data_path: str, transform: Optional[Callable] = None) -> DataLoader:
        """加载图像数据集"""
        pass
    
    def load_sequence_dataset(self, data_path: str) -> DataLoader:
        """加载序列数据集（文本/时序）"""
        pass
    
    def validate_input(self, tensor: Tensor) -> bool:
        """
        校验输入约束
        
        约束条件：
        - 图像尺寸 H,W ∈ [224, 1024]
        - 批次大小 B ≤ 32
        - 序列长度 L ≤ 4096
        
        Raises:
            InvalidInputError: 输入不符合约束时抛出
        """
        pass
    
    def create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """创建 DataLoader"""
        pass
```

#### 3.7.3 preprocessor.py - 输入预处理器

```python
class Preprocessor:
    """输入预处理器：针对不同架构的预处理"""
    
    def __init__(self, model_info: ModelInfo) -> None:
        pass
    
    def preprocess_image(self, image: Union[str, NDArray], target_size: Tuple[int, int]) -> Tensor:
        """图像预处理 → (1, C, H, W)"""
        pass
    
    def preprocess_sequence(self, sequence: Tensor) -> Tensor:
        """序列预处理"""
        pass
    
    def get_default_transform(self, architecture: ModelArchitecture) -> Callable:
        """获取默认预处理变换"""
        pass
```

#### 3.7.4 input_adapter.py - 多输入模型适配器

> **背景**：在测试 MultiModalECGTransformer 时发现，该模型 forward 函数接受两个输入
> `(ecg_signal, meta_data)`，而分析流水线默认假设单输入接口 `model(x)`。
> InputAdapter 负责将多输入模型包装为单输入接口，使分析流水线无需感知模型签名。

```python
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

class AdaptStrategy(Enum):
    """多输入适配策略枚举"""
    PASSTHROUGH = auto()      # 单输入直通：模型本身只有一个输入，不需要适配
    BIND_AUXILIARY = auto()   # 辅助输入绑定：固定辅助输入为预设张量，仅主输入变化
    DICT_EXPAND = auto()      # 字典展开：输入为 dict，将各 key 展开为位置参数


class InputAdapter(nn.Module):
    """
    多输入模型适配器：将多输入模型包装为标准单输入接口 model(x)。
    
    适配策略：
    - PASSTHROUGH：透传，不做任何修改
    - BIND_AUXILIARY：绑定固定的辅助输入（如 meta_data），主输入动态传入
    - DICT_EXPAND：输入为 dict，自动展开为 **kwargs 调用
    
    使用场景：
    - ECG 多模态模型（ecg_signal + meta_data 双输入）
    - 带 attention_mask 的 NLP 模型
    - 任何 forward 签名包含多个必选参数的模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: AdaptStrategy = AdaptStrategy.PASSTHROUGH,
        auxiliary_inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        初始化适配器
        
        Args:
            model: 被包装的原始模型
            strategy: 适配策略
            auxiliary_inputs: BIND_AUXILIARY 策略下的辅助输入字典
                             例如：{"meta_data": torch.zeros(1, 16)}
        """
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.auxiliary_inputs = auxiliary_inputs or {}
    
    def forward(self, x: Any) -> Any:
        """
        统一单输入调用接口
        
        Args:
            x: 主输入张量（PASSTHROUGH/BIND_AUXILIARY），
               或输入字典（DICT_EXPAND）
        
        Returns:
            Any: 模型输出
        """
        if self.strategy == AdaptStrategy.PASSTHROUGH:
            return self.model(x)
        elif self.strategy == AdaptStrategy.BIND_AUXILIARY:
            return self._forward_bind_auxiliary(x)
        elif self.strategy == AdaptStrategy.DICT_EXPAND:
            return self._forward_dict_expand(x)
    
    def _forward_bind_auxiliary(self, x: Any) -> Any:
        """
        辅助输入绑定模式：自动将 auxiliary_inputs 中的张量移到
        与 x 相同的设备，然后与 x 一起传入模型。
        
        示例：
            model(ecg_signal=x, meta_data=self.auxiliary_inputs["meta_data"])
        """
        pass
    
    def _forward_dict_expand(self, x: Dict[str, Any]) -> Any:
        """
        字典展开模式：将输入字典展开为 **kwargs 传入模型。
        
        示例：
            model(**x)  # x = {"ecg_signal": ..., "meta_data": ...}
        """
        pass
    
    @classmethod
    def from_signature(
        cls,
        model: nn.Module,
        auxiliary_inputs: Optional[Dict[str, Any]] = None,
    ) -> "InputAdapter":
        """
        自动从模型 forward 签名推断适配策略（工厂方法）。
        
        推断规则：
        1. 若 forward 仅有一个必选参数（除 self 外）→ PASSTHROUGH
        2. 若 forward 有多个必选参数，且传入了 auxiliary_inputs → BIND_AUXILIARY
        3. 否则记录警告并默认为 PASSTHROUGH
        
        Args:
            model: 被包装的模型
            auxiliary_inputs: 辅助输入字典（可选）
        
        Returns:
            InputAdapter: 适配器实例
        """
        pass
    
    def get_wrapped_model(self) -> nn.Module:
        """获取被包装的原始模型（用于 Hook 注册等需要原始模型的场景）"""
        return self.model
    
    def to_device(self, device: Union[str, torch.device]) -> "InputAdapter":
        """将适配器（含辅助输入）移动到指定设备"""
        pass
```

**使用示例（ECG 双输入模型适配）：**

```python
# 示例：MultiModalECGTransformer 有双输入 forward(ecg_signal, meta_data)
from data_manager.input_adapter import InputAdapter, AdaptStrategy

model = MultiModalECGTransformer(...)

# 方法1：自动推断策略（推荐）
# 提供一个固定的 meta_data（如全零的辅助输入用于测试）
meta_placeholder = torch.zeros(1, 16)  # 根据实际维度调整
adapter = InputAdapter.from_signature(
    model,
    auxiliary_inputs={"meta_data": meta_placeholder}
)

# 方法2：手动指定策略
adapter = InputAdapter(
    model,
    strategy=AdaptStrategy.BIND_AUXILIARY,
    auxiliary_inputs={"meta_data": meta_placeholder}
)

# 之后 adapter 表现为单输入模型，可直接接入分析流水线
pipeline = AnalysisPipeline(model=adapter, data_path="path/to/ecg_data/")
```

---

### 3.8 utils/ - 工具模块

#### 3.8.1 tensor_utils.py - 张量操作工具

```python
class TensorUtils:
    """张量工具类"""
    
    @staticmethod
    def ensure_same_shape(tensors: List[Tensor]) -> List[Tensor]:
        """确保多个张量形状一致"""
        pass
    
    @staticmethod
    def batch_average(tensor: Tensor, batch_dim: int = 0) -> Tensor:
        """对batch维度取平均"""
        pass
    
    @staticmethod
    def flatten_spatial(tensor: Tensor) -> Tensor:
        """展平空间维度"""
        pass
    
    @staticmethod
    def to_numpy(tensor: Tensor) -> NDArray:
        """安全转换为numpy数组"""
        pass
    
    @staticmethod
    def safe_divide(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        """安全除法（避免除零）"""
        pass
```

#### 3.8.2 memory_utils.py - 内存管理工具

```python
class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_gb: float = 8.0) -> None:
        """
        初始化内存管理器
        
        Args:
            max_memory_gb: 最大内存限制（GB）
        """
        pass
    
    def check_memory_usage(self) -> float:
        """检查当前内存使用量（GB）"""
        pass
    
    def clear_cache(self) -> None:
        """清理PyTorch缓存"""
        pass
    
    def suggest_batch_size(self, model: nn.Module, sample_input: Tensor) -> int:
        """建议合适的batch size"""
        pass
    
    def offload_to_cpu(self, tensors: List[Tensor]) -> List[Tensor]:
        """将张量卸载到CPU"""
        pass
```

#### 3.8.3 io_utils.py - IO操作工具

```python
class IOUtils:
    """IO工具类"""
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在"""
        pass
    
    @staticmethod
    def save_json(data: Dict, path: str) -> None:
        """保存JSON文件"""
        pass
    
    @staticmethod
    def load_json(path: str) -> Dict:
        """加载JSON文件"""
        pass
    
    @staticmethod
    def save_image(image: NDArray, path: str) -> None:
        """保存图像"""
        pass
    
    @staticmethod
    def load_image(path: str) -> NDArray:
        """加载图像"""
        pass
```

---

### 3.9 pipeline.py - 推理编排器（主入口）

```python
@dataclass
class PipelineConfig:
    """
    流水线运行时配置（各 Stage 跳过开关）。
    
    设计动机（来自 ECG 模型测试反思）：
    - 1D 序列模型（ECG / NLP）无需空间重构，应跳过 spatial/ 阶段
    - 多输入模型可通过 input_adapter 在加载阶段完成适配，无需修改后续所有阶段
    - 全局诊断开销较大，单次快速分析时可跳过
    """
    # Stage 跳过开关
    skip_spatial: bool = False
    """跳过空间重构阶段（spatial/）。1D 序列模型（ECG/NLP）应设为 True"""
    
    skip_global_diagnosis: bool = False
    """跳过全局诊断阶段（cross-sample 累积统计）。快速单样本分析时可设为 True"""
    
    skip_fusion: bool = False
    """跳过融合阶段（analyzer/fusion_utils）。仅需原始注意力/梯度图时可设为 True"""
    
    # 架构探测覆盖
    detector_override: Optional[Dict[str, Any]] = None
    """
    传入 ArchitectureDetector.detect(override=...) 的覆盖参数。
    用于多模态/混合架构的误判修正。
    示例：{"architecture": ModelArchitecture.BERT, "num_heads": 8}
    """
    
    # 输入适配器配置
    input_adapter_auxiliary: Optional[Dict[str, Any]] = None
    """
    传入 InputAdapter 的辅助输入字典（BIND_AUXILIARY 策略）。
    示例：{"meta_data": torch.zeros(1, 16)}
    若为 None，则由 InputAdapter.from_signature 自动推断策略。
    """


class AnalysisPipeline:
    """分析流程编排器：提供一键式分析接口"""
    
    def __init__(
        self,
        model: Union[str, nn.Module],
        data_path: str,
        config: Optional[Config] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        初始化分析流程编排器
        
        Args:
            model: 模型路径（.pth/.pt 文件或 HuggingFace 模型名）
                   或直接传入 nn.Module 实例（包括 InputAdapter 包装后的模型）
            data_path: 数据路径（图像目录或序列数据文件）
            config: 全局配置对象（可选）
            pipeline_config: 流水线运行时配置，控制 Stage 跳过开关和架构覆盖
        """
        pass
    
    def run(self, output_dir: str = "./results") -> Dict[str, Any]:
        """
        执行分析流程（支持按照 PipelineConfig 条件跳过指定 Stage）。
        
        Stage 执行顺序：
        1. 加载模型和数据（含 InputAdapter 自动适配）
        2. 架构探测与 Hook 注册
        3. 前向+反向传播追踪
        4. 跨样本累积统计（可跳过：skip_global_diagnosis=True）
        5. [Optional] 空间重构（可跳过：skip_spatial=True）
        6. 单样本分析与全局诊断
        7. [Optional] 融合计算（可跳过：skip_fusion=True）
        8. 可视化输出
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict: 包含所有分析结果的字典
        """
        pass
    
    def _init_components(self) -> None:
        """
        初始化所有组件，并根据 PipelineConfig 自动适配模型输入。
        
        操作顺序：
        1. 加载模型（若 model 为路径字符串）
        2. 检查 forward 签名，必要时自动创建 InputAdapter
        3. 架构探测器 (ArchitectureDetector)，应用 detector_override
        4. Hook管理器 (HookManager)
        5. 前向/反向追踪器 (ForwardTracker/BackwardTracker)
        6. 累积器 (CrossSampleAccumulator)
        7. 分析器 (SingleSampleAnalyzer/GlobalDiagnosisEngine)
        8. 可视化工具 (HeatmapRenderer)
        """
        pass
    
    def analyze_batch(self, batch_data: Tensor) -> Dict[str, Any]:
        """
        单批次分析：前向+反向+捕获
        
        Args:
            batch_data: 批次数据张量
            
        Returns:
            Dict: 批次分析结果
        """
        pass
    
    def _compute_loss(self, output: Tensor, input_data: Tensor) -> Tensor:
        """
        计算损失（支持分类/无监督场景）
        
        对于分类任务：使用交叉熵损失
        对于无监督任务：使用重构损失或其他自监督损失
        
        Args:
            output: 模型输出
            input_data: 输入数据
            
        Returns:
            Tensor: 损失张量
        """
        pass
    
    def get_single_sample_result(self, sample_idx: int) -> Dict[str, Any]:
        """
        获取单样本分析结果
        
        Args:
            sample_idx: 样本索引
            
        Returns:
            Dict: 单样本分析结果，包含：
                - attention_maps: 注意力图
                - gradient_maps: 梯度图
                - fusion_map: 融合重要性图
                - quadrant_map: 四象限分类图
        """
        pass
    
    def get_global_diagnosis(self) -> Dict[str, Any]:
        """
        获取全局诊断报告
        
        Returns:
            Dict: 全局诊断报告，包含：
                - activation_frequency_ranking: 激活频率排名
                - gradient_importance_ranking: 梯度重要性排名
                - anomaly_analysis: 异常分析结果
                - head_classification: 头分类结果
        """
        pass
    
    def generate_report(self, output_dir: str) -> str:
        """
        生成完整可视化报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        pass
```

---

## 4. 数据流图

```
输入配置（model + data_path + PipelineConfig）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 推理编排器: pipeline.py                                       │
│  - AnalysisPipeline.__init__()                              │
│  - ModelLoader.load_from_checkpoint()                       │
│  - InputAdapter.from_signature()  [多输入模型自动适配]       │
│  - DataManager.load_*_dataset()                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
输入图像 / 序列（单接口 model(x)，由 InputAdapter 统一处理）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 模块1: model_adapter/                                        │
│  - ArchitectureDetector.detect(override=...)                 │
│    → DetectionResult(confidence, warnings)                  │
│  - HookManager.register_all_hooks()                          │
│    → 内置 TransformerEncoderLayer need_weights 补丁 Hook    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 模块2: tracker/                                              │
│  - ForwardTracker.track() → 注意力矩阵                       │
│  - BackwardTracker.track() → 梯度矩阵                        │
│  - CrossSampleAccumulator.update() → 累积统计                │
└─────────────────────────────────────────────────────────────┘
    │
    ├──────────────────[skip_spatial=True 时跳过]─────────────┐
    │                                                         │
    ▼                                                         │
┌─────────────────────────────────┐                          │
│ [可选] 模块3: spatial/          │                          │
│  仅图像输入（ViT/Swin 等）需要  │                          │
│  - SpatialReshaper              │                          │
│  - Normalizer                   │                          │
│  ECG/NLP 等 1D 序列跳过此模块  │                          │
└─────────────────────────────────┘                          │
    │                                                         │
    └──────────────────────────────────────────────────── ───┘
    │
    ├────────────────────────────────────────────────────────┐
    ▼                                                        ▼
┌─────────────────────────────────┐              ┌─────────────────────────────────┐
│ 模块4: analyzer/ (单样本轨道)   │              │ 模块4: analyzer/ (全局诊断轨道) │
│  - SingleSampleAnalyzer         │              │  - GlobalDiagnosisEngine        │
│  - QuadrantAnalyzer             │              │  - AnomalyDetector              │
│  - fusion_utils（加权求和/      │              │  （skip_global_diagnosis=True   │
│    GradCAM式融合）              │              │    时跳过）                     │
└─────────────────────────────────┘              └─────────────────────────────────┘
    │                                                        │
    │                [skip_fusion=True 时直接跳到下一步]      │
    ▼                                                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 模块5: visualization/                                        │
│  - HeatmapRenderer → 注意力/梯度/融合热力图                  │
│  - plot_utils → 累积频率统计图、头分类散点图                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
输出结果（热力图、统计图表、JSON 摘要）
```

---

## 5. 使用示例

### 5.1 基础使用流程（图像模型）

```python
from transformer_analyzer import AnalysisPipeline, Config, PipelineConfig

# 方式1：极简使用（推荐）
pipeline = AnalysisPipeline(
    model="path/to/model.pth",
    data_path="path/to/test_images/",
    config=Config()
)
results = pipeline.run(output_dir="./results")
```

### 5.2 ECG 多模态模型使用（InputAdapter + skip_spatial）

> **场景**：MultiModalECGTransformer 有双输入 forward(ecg_signal, meta_data)，
> 且为 1D 序列模型，不需要空间重构。

```python
import torch
from transformer_analyzer import AnalysisPipeline, Config, PipelineConfig
from data_manager.input_adapter import InputAdapter, AdaptStrategy

# 1. 加载原始模型
from your_module import MultiModalECGTransformer
model = MultiModalECGTransformer(...)
checkpoint = torch.load("path/to/ecg_model.pth")
model.load_state_dict(checkpoint["state_dict"])

# 2. 创建 InputAdapter（将双输入模型包装为单输入接口）
meta_placeholder = torch.zeros(1, 16)  # 固定 meta_data，维度根据实际调整
adapter = InputAdapter.from_signature(
    model,
    auxiliary_inputs={"meta_data": meta_placeholder}
)

# 3. 配置流水线：1D 序列跳过 spatial/ 阶段
pipeline_config = PipelineConfig(
    skip_spatial=True,           # ECG 为 1D 序列，无需空间重构
    skip_global_diagnosis=False, # 默认开启跨样本累积统计
    skip_fusion=False,           # 默认开启融合
)

# 4. 直接将 adapter 作为模型传入 pipeline
pipeline = AnalysisPipeline(
    model=adapter,               # 待分析模型（已包装为单输入）
    data_path="path/to/ecg_data/",
    config=Config(),
    pipeline_config=pipeline_config,
)
results = pipeline.run(output_dir="./ecg_results")
```

### 5.3 架构探测覆盖（防止多模态模型误判）

```python
from transformer_analyzer import AnalysisPipeline, PipelineConfig
from core.types import ModelArchitecture

# 当自动探测置信度较低时，手动覆盖关键架构参数
pipeline_config = PipelineConfig(
    skip_spatial=True,
    detector_override={
        "architecture": ModelArchitecture.BERT,  # 强制指定架构类型
        "num_heads": 8,                           # 强制指定头数
        "num_layers": 6,                          # 强制指定层数
    }
)

pipeline = AnalysisPipeline(
    model=adapter,
    data_path="path/to/ecg_data/",
    pipeline_config=pipeline_config,
)
```

### 5.4 高级手动控制（逐步操作）

```python
from transformer_analyzer import (
    Config, ArchitectureDetector, HookManager,
    ForwardTracker, BackwardTracker, CrossSampleAccumulator,
    SingleSampleAnalyzer, GlobalDiagnosisEngine,
    ModelLoader, DataManager
)
from data_manager.input_adapter import InputAdapter
from analyzer.fusion_utils import gradcam_fusion, normalize_for_fusion
from visualization.heatmap import HeatmapRenderer

# 1. 加载模型并适配输入
model_loader = ModelLoader(device="auto", precision="fp32")
model = model_loader.load_from_checkpoint("path/to/ecg_model.pth")

# 检查 forward 签名并自动创建适配器
meta_placeholder = torch.zeros(1, 16)
adapter = InputAdapter.from_signature(model, auxiliary_inputs={"meta_data": meta_placeholder})

# 2. 架构探测
detector = ArchitectureDetector()
detection_result = detector.detect_with_confidence(adapter.get_wrapped_model())
if detection_result.confidence < 0.6:
    print(f"[Warning] Low confidence detection: {detection_result.warnings}")
model_info = detection_result.model_info

# 3. Hook 注册
hook_manager = HookManager(adapter.get_wrapped_model(), model_info)
hook_manager.register_all_hooks()  # 内置 TransformerEncoderLayer need_weights 补丁

# 4. 追踪器初始化
forward_tracker = ForwardTracker(hook_manager)
backward_tracker = BackwardTracker(adapter.get_wrapped_model(), hook_manager)
accumulator = CrossSampleAccumulator(model_info)

# 5. 加载数据
data_manager = DataManager(Config())
dataloader = data_manager.load_sequence_dataset("path/to/ecg_data/")

# 6. 处理数据批次
for batch_data in dataloader:
    if not data_manager.validate_input(batch_data):
        continue
    
    # 前向传播（通过 adapter 统一接口）
    attention_maps = forward_tracker.track(adapter, batch_data)
    
    # 反向传播
    loss = compute_loss(adapter, batch_data)
    gradients = backward_tracker.track(loss)
    
    accumulator.update(
        attention_maps=attention_maps,
        input_gradients=gradients["input"],
        hidden_gradients=gradients["hidden"],
        attention_gradients=gradients.get("attention"),
    )

# 7. 单样本分析
sample_analyzer = SingleSampleAnalyzer(
    model_info.num_layers,
    model_info.num_heads,
    threshold_method="median"
)
single_result = sample_analyzer.analyze(attention_maps, gradients["input"])

# 8. 融合计算（使用 fusion_utils 工具函数）
attn_norm = normalize_for_fusion(attention_maps["layer_0"])
grad_norm = normalize_for_fusion(gradients["hidden"]["layer_0"])
fusion_map = gradcam_fusion(attn_norm, grad_norm)

# 9. 可视化
renderer = HeatmapRenderer()
renderer.render_1d_sequence(fusion_map, title="ECG 融合重要性图")
```

---

## 6. 设计迭代说明

> 本章记录设计文档在实际模型测试过程中发现的问题及其导致的设计变更。

### 6.1 反思来源：MultiModalECGTransformer 测试

**测试模型**：一个双输入 ECG Transformer 模型，具体为
`MultiModalECGTransformer.forward(ecg_signal, meta_data)`。

在测试中遇到以下问题：

| 问题 | 现象 | 导致的设计变更 |
|------|------|------------------|
| nn.TransformerEncoderLayer need_weights | 默认 need_weights=False，注意力权重全为 None | hooks.py 内置补丁，直接调用 F.multi_head_attention_forward | 
| ECG 输入无法转为 2D Patch 映射 | SpatialReshaper 假设 2D 空间结构 | spatial/ 降级为可选模块，1D 模型设 skip_spatial=True |
| 双输入模型接口不兼容 | pipeline 默认假设 model(x) 单输入 | 新增 InputAdapter，将双输入包装为单输入接口 |
| 多模态架构探测误判 | 启发式论断对混合模型置信度屏 | detector.py 新增 override 和 confidence 机制 |
| fusion/ 策略模式过重 | ABC + 工厂设计对2个融合方法来说过于臃重 | fusion/ 并入 analyzer/fusion_utils.py 工具函数 |
| DashboardGenerator 过重 | 单次分析出图无需全功能交互面板 | visualization/ 简化为 HeatmapRenderer + plot_utils |

### 6.2 核心设计原则变更

**局部修改不影响全局：**
- 所有跳过逻辑均通过 PipelineConfig 配置开关控制，不会影响其它模块
- InputAdapter 包装层对下游透明：HookManager 仍在原始模型上注册，不受 Adapter 层干扰
- spatial/ 保留完整设计，仅标注为可选，图像模型不受影响

**简化而非删除：**
- fusion/ 的功能未删除，而是迁移到 analyzer/fusion_utils.py，以工具函数形式提供
- visualization/ 的存档/汇总功能标注为“未来扩展”，而非删除
- ArchitectureDetector 的 confidence 机制可将自动探测结果降级为“参考建议”的起点

---

## 7. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-01 | 初始版本，基于PRD设计文档创建 |
| v1.1 | 2026-04-02 | 新增模型加载与数据管理模块；新增推理编排器；修复接口设计（BackwardTracker/SpatialReshaper/SingleSampleAnalyzer/GlobalDiagnosisEngine/Accumulator）；补充FP16精度管理和DataParallel支持；补充类型定义；完善Swin和Dashboard接口 |
| v1.2 | 2026-04-02 | 基于 MultiModalECGTransformer 测试反思进行大规模设计迭代：spatial/ 降级为可选模块；fusion/ 并入 analyzer/fusion_utils.py；visualization/ 简化为 HeatmapRenderer + plot_utils；新增 InputAdapter 外层适配器；hooks.py 内置 TransformerEncoderLayer need_weights 补丁；detector.py 新增 override + confidence 机制；pipeline.py 新增 PipelineConfig Stage 跳过开关 |
