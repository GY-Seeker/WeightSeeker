"""Hook registration and management module."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from ..core.types import Tensor, ModelInfo, ModelArchitecture
from ..core.exceptions import HookRegistrationError


class HookManager:
    """Hook管理器：统一注册和管理所有Hook。

    本管理器根据 :class:`ModelInfo` 中的架构信息，在模型的注意力层和
    Transformer Block 输出处注册前向Hook，用于捕获：
    
    - 注意力权重矩阵
    - 隐藏状态（Block输出）
    
    所有捕获到的数据会缓存在 ``_hook_storage`` 中，调用者可以通过
    :meth:`get_attention_output` 和 :meth:`get_hidden_state` 进行访问。
    """

    def __init__(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        use_data_parallel: bool = False,
    ) -> None:
        """初始化Hook管理器。

        Args:
            model: 目标模型（支持DataParallel包装的模型）。
            model_info: 模型信息，用于指导Hook注册策略。
            use_data_parallel: 是否使用DataParallel模式；若为True且
                ``model`` 带有 ``.module`` 属性，则自动解包。
        """
        if use_data_parallel and hasattr(model, "module"):
            # 兼容DataParallel封装
            model = getattr(model, "module")  # type: ignore[assignment]

        self.model: nn.Module = model
        self.model_info: ModelInfo = model_info
        self.use_data_parallel: bool = use_data_parallel

        # 存储结构：{"attention": {layer_idx: Tensor}, "hidden_state": {layer_idx: Tensor}}
        self._hook_storage: Dict[str, Dict[int, Tensor]] = {
            "attention": {},
            "hidden_state": {},
        }
        # 所有已注册Hook的句柄，用于统一移除
        self._handles: List[RemovableHandle] = []

    def register_all_hooks(self) -> Dict[str, List[RemovableHandle]]:
        """注册全套数据捕获探针。

        根据已识别的模型架构自动遍历子模块，在注意力层和Transformer
        Block输出处注册前向Hook。

        Returns:
            Dict[str, List[RemovableHandle]]: 以Hook类型为键、Hook句柄列表
            为值的字典，例如 ``{"attention": [...], "hidden_state": [...]}``。

        Raises:
            HookRegistrationError: 若未能在模型中找到任何注意力层或Block
                模块，导致无法注册对应类型的Hook时抛出。
        """
        attention_handles: List[RemovableHandle] = []
        hidden_handles: List[RemovableHandle] = []

        # 注册注意力Hook
        layer_idx = 0
        for name, module in self.model.named_modules():
            if self._is_attention_module(name, module):
                handle = self.register_attention_hook(layer_idx, module)
                attention_handles.append(handle)
                layer_idx += 1

        # 注册隐藏状态Hook（以Block/Layer为单位）
        hidden_layer_idx = 0
        for name, module in self.model.named_modules():
            if self._is_block_module(name, module):
                handle = self.register_hidden_state_hook(hidden_layer_idx, module)
                hidden_handles.append(handle)
                hidden_layer_idx += 1

        if not attention_handles:
            raise HookRegistrationError(hook_type="attention")
        if not hidden_handles:
            # 对部分模型，可能没有显式Block；此时不强制抛错，但给出警告。
            # 为保持简单实现，这里仅在无任何Block时发出异常。
            raise HookRegistrationError(hook_type="hidden_state")

        return {
            "attention": attention_handles,
            "hidden_state": hidden_handles,
        }

    def register_attention_hook(self, layer_idx: int, module: nn.Module) -> RemovableHandle:
        """注册标准注意力Hook。

        在给定的注意力模块上注册前向Hook，捕获其输出中的注意力权重，
        并将其归一化为统一格式后存入 ``_hook_storage["attention"][layer_idx]``。

        Args:
            layer_idx: 逻辑层索引，用于在存储字典中区分不同层。
            module: 需要注册Hook的注意力模块。

        Returns:
            RemovableHandle: Hook句柄，可用于后续移除。
        """
        storage = self._hook_storage.setdefault("attention", {})
        hook_fn = AttentionHook(storage=storage, key=layer_idx, arch=self.model_info.architecture)
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)
        return handle

    def register_hidden_state_hook(self, layer_idx: int, module: nn.Module) -> RemovableHandle:
        """注册隐藏状态Hook。

        Hook将捕获指定Block的前向输出（通常为隐藏状态），并存入
        ``_hook_storage["hidden_state"][layer_idx]``。

        Args:
            layer_idx: 逻辑层索引。
            module: 需要注册Hook的模块，一般为Transformer Block或EncoderLayer。

        Returns:
            RemovableHandle: Hook句柄。
        """

        def _hook_fn(mod: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # output 可能是Tensor或Tuple[Tensor, ...]
            tensor: Optional[Tensor]
            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, (tuple, list)) and output:
                first = output[0]
                tensor = first if isinstance(first, torch.Tensor) else None
            else:
                tensor = None

            if tensor is not None:
                self._hook_storage.setdefault("hidden_state", {})[layer_idx] = tensor.detach().clone()

        handle = module.register_forward_hook(_hook_fn)
        self._handles.append(handle)
        return handle

    def remove_all_hooks(self) -> None:
        """移除所有注册的Hook并清空缓存。"""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                # 移除失败不应影响整体流程，忽略个别异常
                continue
        self._handles.clear()
        self._hook_storage["attention"].clear()
        self._hook_storage["hidden_state"].clear()

    def get_attention_output(self, layer_idx: int) -> Tensor:
        """获取指定层的注意力输出。

        Args:
            layer_idx: 层索引。

        Returns:
            Tensor: 形状通常为 ``(B, num_heads, seq_len, seq_len)`` 的注意力矩阵。

        Raises:
            KeyError: 若指定层尚未捕获到注意力输出。
        """
        return self._hook_storage["attention"][layer_idx]

    def get_hidden_state(self, layer_idx: int) -> Tensor:
        """获取指定层的隐藏状态。

        Args:
            layer_idx: 层索引。

        Returns:
            Tensor: 对应层的隐藏状态张量。

        Raises:
            KeyError: 若指定层尚未捕获到隐藏状态。
        """
        return self._hook_storage["hidden_state"][layer_idx]

    def clear_storage(self) -> None:
        """清空当前批次的缓存数据但不移除Hook。

        在跨批次分析场景下，可在每个批次处理结束后调用该方法，以
        释放张量引用、减少显存占用。
        """
        self._hook_storage["attention"].clear()
        self._hook_storage["hidden_state"].clear()

    @staticmethod
    def _is_attention_module(name: str, module: nn.Module) -> bool:
        """判断给定模块是否为注意力模块。

        通过模块类名和层级命名进行启发式匹配，兼容多种实现：
        - nn.MultiheadAttention
        - 类名中包含"Attention"/"Attn"/"SelfAttention"等
        - 名称以"attn"结尾的子模块
        """
        class_name = type(module).__name__.lower()
        name_lower = name.lower()

        if isinstance(module, nn.MultiheadAttention):
            return True
        if "attention" in class_name or "attn" in class_name:
            return True
        if name_lower.endswith(".attn") or name_lower.endswith("attn"):
            return True
        return False

    @staticmethod
    def _is_block_module(name: str, module: nn.Module) -> bool:
        """判断给定模块是否为Transformer Block/Layer。

        通过类名中的关键字进行启发式匹配，例如：
        - "Block"
        - "EncoderLayer" / "DecoderLayer"
        - "TransformerLayer" / "TransformerBlock" 等。
        """
        class_name = type(module).__name__.lower()

        block_keywords = [
            "block",
            "encoderlayer",
            "decoderlayer",
            "transformerlayer",
            "transformerblock",
        ]
        return any(kw in class_name for kw in block_keywords)


class AttentionHook:
    """注意力Hook实现类。

    该Hook用于在前向传播时捕获注意力权重，并将其统一整理为
    ``(B, num_heads, seq_len, seq_len)`` 或窗口格式（Swin）后存储。
    """

    def __init__(
        self,
        storage: Dict[int, Tensor],
        key: int,
        arch: Optional[ModelArchitecture] = None,
    ) -> None:
        """初始化Hook。

        Args:
            storage: 外部传入的存储字典引用，Hook会将结果写入其中。
            key: 存储键，一般为层索引 ``layer_idx``。
            arch: 模型架构类型，用于指导注意力张量的标准化逻辑。
        """
        self._storage: Dict[int, Tensor] = storage
        self._key: int = key
        self._arch: Optional[ModelArchitecture] = arch

    def __call__(self, module: nn.Module, input: Tuple[Any, ...], output: Any) -> None:
        """Hook回调函数。

        从模块的前向输出中提取注意力权重，并进行标准化后存储。

        注意：
            - 对于 ``nn.MultiheadAttention``，尝试从 ``output[1]`` 中获取
              ``attn_weights``。
            - 若输出本身即为注意力矩阵，则直接使用。
            - 对于其他自定义实现，目前采用启发式判断，仅在输出为张量
              时按注意力矩阵处理。
        """
        attn: Optional[Tensor] = None

        if isinstance(output, torch.Tensor):
            attn = output
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            second = output[1]
            if isinstance(second, torch.Tensor):
                attn = second
        # 其他复杂情况暂不处理

        if attn is None:
            return

        arch = self._arch or ModelArchitecture.TRANSFORMER
        normalized = self.normalize_attention(attn, arch)
        # detach + clone 避免持有计算图并防止后续修改
        self._storage[self._key] = normalized.detach().clone()

    def normalize_attention(self, attention: Tensor, arch: ModelArchitecture) -> Tensor:
        """将不同架构的注意力输出统一为标准张量。

        标准格式定义为： ``(B, num_heads, seq_len, seq_len)``。

        对于Swin架构，窗口级注意力通常为 ``(B*num_windows, num_heads,
        window_size^2, window_size^2)``，此处不强行重排，由
        :class:`SwinHandler` 进行进一步适配，因此直接返回原始张量。

        Args:
            attention: 原始注意力张量。
            arch: 模型架构类型。

        Returns:
            Tensor: 尽可能转换为 ``(B, num_heads, seq_len, seq_len)`` 的张量；
            若无法可靠判断，则返回原始张量。
        """
        attn = attention

        # Swin交由SwinHandler处理，不在此处重排
        if arch == ModelArchitecture.SWIN:
            return attn

        # 已是标准4维格式 (B, H, N, N) 或 (H, B, N, N)
        if attn.dim() == 4:
            b0, b1, n1, n2 = attn.shape
            # 若第二维较小，通常表示head数
            if b1 <= 64:
                # 视为 (B, H, N, N)
                return attn
            # 否则可能是 (H, B, N, N)
            return attn.permute(1, 0, 2, 3)

        # 三维情况：可能是 (B, N, N) 或 (H, N, N) 或 (N, B, N)
        if attn.dim() == 3:
            s0, s1, s2 = attn.shape
            # 方阵情况，尝试视为 (H, N, N)
            if s1 == s2:
                # 视作多头、单batch
                return attn.unsqueeze(0)  # (1, H, N, N) 或 (1, 1, N, N)
            # 非方阵，视作 (B, L, S)
            return attn.unsqueeze(1)  # (B, 1, L, S)

        # 二维情况：单头单样本 (N, N)
        if attn.dim() == 2 and attn.shape[0] == attn.shape[1]:
            n = attn.shape[0]
            return attn.view(1, 1, n, n)

        # 其他情况暂不处理，直接返回
        return attn
