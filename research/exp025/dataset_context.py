"""Compatibility import for the shared research dataset context.

CAD and retained PCC models use one dataset/object-order resolver.  Keep the
historical EXP025 import path for builders and diagnostic tools.
"""

from core.gdrn_modeling.datasets.research_context import DatasetContext
from core.gdrn_modeling.datasets.research_context import resolve_dataset_context as _resolve


def resolve_dataset_context(cfg, *, hierarchy_path=None, require_hierarchy=True):
    """Keep the historical CAD default path while delegating all validation."""
    path = (cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
            if hierarchy_path is None else hierarchy_path)
    return _resolve(cfg, hierarchy_path=path, require_hierarchy=require_hierarchy)

__all__ = ("DatasetContext", "resolve_dataset_context")
