"""
Compatibility shim for the legacy GUI entry point.

The GUI implementation has moved to ``pixel_perfecter.gui_app``. Import the
new location to ensure older ``python -m src.gui_feedback_app`` invocations
continue to work.
"""

from pixel_perfecter.gui_app import *  # noqa: F401,F403

__all__ = [
    name
    for name in globals().keys()
    if not name.startswith("_")
]
