"""Helper for patchmap structures used by tutorials and CLI tools.

Re-export get_structure and configuration models for convenience.
"""

from .patchmaps import StructureOptions, get_structure

__all__ = ["StructureOptions", "get_structure"]
