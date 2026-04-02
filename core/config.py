"""Global configuration management for transformer analyzer.

本模块提供全局配置管理功能，
包括默认参数、文件加载、自定义覆盖与合法性校验等能力。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

try:  # 可选依赖：yaml
    import yaml  # type: ignore
except Exception:  # pragma: no cover - 运行时兜底
    yaml = None  # type: ignore

from .exceptions import InvalidInputError


@dataclass
class Config:
    """全局配置管理类。

    该类既提供一组合理的类级默认值，又允许通过实例级配置进行覆盖，
    并支持从 JSON / YAML 文件加载配置。
    """

    # ------------------------------------------------------------------
    # 类属性（作为默认值，同时也暴露为配置键）
    # ------------------------------------------------------------------
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MAX_IMAGE_SIZE: int = 1024
    MIN_IMAGE_SIZE: int = 224
    MAX_BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 4096
    ACCUMULATOR_LIMIT: int = 100000
    DEFAULT_PERCENTILE_LOW: float = 0.01
    DEFAULT_PERCENTILE_HIGH: float = 0.99
    DEFAULT_FUSION_ALPHA: float = 0.5
    DEFAULT_ATTENTION_THRESHOLD: float = 0.3
    PRECISION: str = "fp32"
    MIN_GPU_MEMORY_FP16: float = 8.0

    # ------------------------------------------------------------------
    # 实例级自定义配置存储
    # ------------------------------------------------------------------
    _config: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(self, config_path: Optional[str] = None) -> None:  # type: ignore[override]
        """初始化配置，支持从文件加载。

        Args:
            config_path: 可选的配置文件路径（JSON 或 YAML）。
        """
        # dataclass 默认 __init__ 被覆盖，这里手动初始化 _config
        object.__setattr__(self, "_config", {})

        if config_path is not None:
            self.load_from_file(config_path)

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------
    def load_from_file(self, path: str) -> None:
        """从 YAML/JSON 文件加载配置。

        根据文件扩展名自动识别格式：
        - ``.json``: 使用 :mod:`json` 加载
        - ``.yml`` / ``.yaml``: 使用 :mod:`yaml` 加载（若未安装则抛出 :class:`InvalidInputError`）
        """
        if not isinstance(path, str) or not path:
            raise InvalidInputError(expected="non-empty file path", actual=str(path))

        lower = path.lower()
        try:
            with open(path, "r", encoding="utf-8") as f:
                if lower.endswith(".json"):
                    data = json.load(f)
                elif lower.endswith(".yml") or lower.endswith(".yaml"):
                    if yaml is None:
                        raise InvalidInputError(
                            expected="PyYAML to be installed for YAML config",
                            actual="yaml library not available",
                        )
                    data = yaml.safe_load(f)  # type: ignore[call-arg]
                else:
                    raise InvalidInputError(
                        expected="file with extension .json/.yml/.yaml",
                        actual=path,
                    )
        except FileNotFoundError as exc:
            raise InvalidInputError(expected="existing config file", actual=path) from exc

        if not isinstance(data, dict):
            raise InvalidInputError(expected="mapping at top level", actual=type(data).__name__)

        for key, value in data.items():
            self.set(key, value)

    # ------------------------------------------------------------------
    # 访问与修改
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项。

        查找顺序：
        1. 实例级配置字典 ``_config``
        2. 类属性（默认值）
        3. 调用方提供的 ``default``
        """
        if key in self._config:
            return self._config[key]
        if hasattr(self.__class__, key):
            return getattr(self.__class__, key)
        return default

    def set(self, key: str, value: Any) -> None:
        """设置配置项。

        Args:
            key: 配置键名。
            value: 配置值。
        """
        self._config[key] = value

    # ------------------------------------------------------------------
    # 校验与导出
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        """验证配置合法性。

        校验规则：
        - ``PRECISION`` 必须是 "fp32" 或 "fp16"
        - ``MIN_IMAGE_SIZE`` <= ``MAX_IMAGE_SIZE``
        - ``MAX_BATCH_SIZE`` > 0 且 <= 32
        - ``MAX_SEQUENCE_LENGTH`` > 0 且 <= 4096
        - ``ACCUMULATOR_LIMIT`` > 0
        - ``DEFAULT_FUSION_ALPHA`` 在 [0, 1] 范围内
        - ``DEFAULT_PERCENTILE_LOW`` < ``DEFAULT_PERCENTILE_HIGH``

        Returns:
            bool: 当配置合法时返回 True。

        Raises:
            InvalidInputError: 配置不合法时抛出。
        """
        precision = str(self.get("PRECISION", self.PRECISION)).lower()
        if precision not in {"fp32", "fp16"}:
            raise InvalidInputError(
                expected="PRECISION in {'fp32', 'fp16'}",
                actual=precision,
            )

        min_image = int(self.get("MIN_IMAGE_SIZE", self.MIN_IMAGE_SIZE))
        max_image = int(self.get("MAX_IMAGE_SIZE", self.MAX_IMAGE_SIZE))
        if min_image > max_image:
            raise InvalidInputError(
                expected="MIN_IMAGE_SIZE <= MAX_IMAGE_SIZE",
                actual=f"MIN_IMAGE_SIZE={min_image}, MAX_IMAGE_SIZE={max_image}",
            )

        max_batch = int(self.get("MAX_BATCH_SIZE", self.MAX_BATCH_SIZE))
        if not (1 <= max_batch <= 32):
            raise InvalidInputError(
                expected="1 <= MAX_BATCH_SIZE <= 32",
                actual=str(max_batch),
            )

        max_seq = int(self.get("MAX_SEQUENCE_LENGTH", self.MAX_SEQUENCE_LENGTH))
        if not (1 <= max_seq <= 4096):
            raise InvalidInputError(
                expected="1 <= MAX_SEQUENCE_LENGTH <= 4096",
                actual=str(max_seq),
            )

        acc_limit = int(self.get("ACCUMULATOR_LIMIT", self.ACCUMULATOR_LIMIT))
        if acc_limit <= 0:
            raise InvalidInputError(
                expected="ACCUMULATOR_LIMIT > 0",
                actual=str(acc_limit),
            )

        alpha = float(self.get("DEFAULT_FUSION_ALPHA", self.DEFAULT_FUSION_ALPHA))
        if not (0.0 <= alpha <= 1.0):
            raise InvalidInputError(
                expected="DEFAULT_FUSION_ALPHA in [0, 1]",
                actual=str(alpha),
            )

        p_low = float(self.get("DEFAULT_PERCENTILE_LOW", self.DEFAULT_PERCENTILE_LOW))
        p_high = float(self.get("DEFAULT_PERCENTILE_HIGH", self.DEFAULT_PERCENTILE_HIGH))
        if not (p_low < p_high):
            raise InvalidInputError(
                expected="DEFAULT_PERCENTILE_LOW < DEFAULT_PERCENTILE_HIGH",
                actual=f"low={p_low}, high={p_high}",
            )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """导出所有配置为字典。

        返回值为**当前有效配置**：实例级配置覆盖类属性默认值。
        """
        # 以类属性为基础
        result: Dict[str, Any] = {
            "DEFAULT_IMAGE_SIZE": self.DEFAULT_IMAGE_SIZE,
            "MAX_IMAGE_SIZE": self.MAX_IMAGE_SIZE,
            "MIN_IMAGE_SIZE": self.MIN_IMAGE_SIZE,
            "MAX_BATCH_SIZE": self.MAX_BATCH_SIZE,
            "MAX_SEQUENCE_LENGTH": self.MAX_SEQUENCE_LENGTH,
            "ACCUMULATOR_LIMIT": self.ACCUMULATOR_LIMIT,
            "DEFAULT_PERCENTILE_LOW": self.DEFAULT_PERCENTILE_LOW,
            "DEFAULT_PERCENTILE_HIGH": self.DEFAULT_PERCENTILE_HIGH,
            "DEFAULT_FUSION_ALPHA": self.DEFAULT_FUSION_ALPHA,
            "DEFAULT_ATTENTION_THRESHOLD": self.DEFAULT_ATTENTION_THRESHOLD,
            "PRECISION": self.PRECISION,
            "MIN_GPU_MEMORY_FP16": self.MIN_GPU_MEMORY_FP16,
        }
        # 覆盖实例级配置
        result.update(self._config)
        return result

    def __repr__(self) -> str:  # pragma: no cover - 纯展示逻辑
        """返回可读的字符串表示，便于调试。

        仅展示当前有效配置字典。
        """
        cfg = self.to_dict()
        return f"Config({cfg})"
