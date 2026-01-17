"""
Compatibility shim for legacy imports.

The core implementation now lives under ``pixel_perfecter.reconstructor``.
This module re-exports everything so that older scripts referencing
``src.pixel_reconstructor`` continue to function during the transition.
"""

from pixel_perfecter.reconstructor import *  # noqa: F401,F403

__all__ = [
    name
    for name in globals().keys()
    if not name.startswith("_")
]
