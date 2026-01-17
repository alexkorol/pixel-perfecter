"""
Convenience launcher for the Qt-based Pixel Perfecter GUI.

Usage:
    python -m pixel_perfecter.gui
"""

from __future__ import annotations

from .gui_app import main as _run_gui


def main() -> None:
    """Launch the interactive GUI."""
    _run_gui()


if __name__ == "__main__":
    main()
