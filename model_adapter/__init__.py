"""Model adapter module for transformer analyzer."""

from .detector import ArchitectureDetector
from .hooks import HookManager, AttentionHook

__all__ = [
    "ArchitectureDetector",
    "HookManager",
    "AttentionHook",
]
