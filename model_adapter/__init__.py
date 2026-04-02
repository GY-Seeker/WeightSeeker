"""Model adapter module for transformer analyzer."""

from .detector import ArchitectureDetector
from .hooks import HookManager, AttentionHook
from .swin_handler import SwinHandler
from .moe_handler import MoEHandler

__all__ = [
    "ArchitectureDetector",
    "HookManager",
    "AttentionHook",
    "SwinHandler",
    "MoEHandler",
]
