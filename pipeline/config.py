"""Pipeline configuration: tag schema, known palettes, grid presets."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Grid sizes the pixelization model should support
# ---------------------------------------------------------------------------
GRID_SIZES = [8, 16, 24, 32, 48, 64]

# Minimum sprite dimensions (in actual pixels) worth keeping
MIN_SPRITE_PX = 8
MAX_SPRITE_PX = 512


# ---------------------------------------------------------------------------
# Named palettes (hex, no '#' prefix stored internally)
# ---------------------------------------------------------------------------
NAMED_PALETTES: Dict[str, List[str]] = {
    "gameboy": [
        "0f380f", "306230", "8bac0f", "9bbc0f",
    ],
    "gameboy_pocket": [
        "000000", "545454", "a9a9a9", "ffffff",
    ],
    "nes": [
        "000000", "fcfcfc", "f8f8f8", "bcbcbc",
        "7c7c7c", "a4e4fc", "3cbcfc", "0078f8",
        "0000fc", "b8b8f8", "6888fc", "0058f8",
        "0000bc", "d8b8f8", "9878f8", "6844fc",
        "4428bc", "f8b8f8", "f878f8", "d800cc",
        "940084", "f8a4c0", "f85898", "e40058",
        "a80020", "f0d0b0", "f87858", "f83800",
        "a81000", "fce0a8", "fca044", "e45c10",
        "881400", "f8d878", "f8b800", "ac7c00",
        "503000", "d8f878", "b8f818", "00b800",
        "007800", "b8f8b8", "58d854", "00a800",
        "006800", "b8f8d8", "58f898", "00a844",
        "005800", "00fcfc", "00e8d8", "008888",
        "004058", "f8d8f8", "787878",
    ],
    "pico8": [
        "000000", "1d2b53", "7e2553", "008751",
        "ab5236", "5f574f", "c2c3c7", "fff1e8",
        "ff004d", "ffa300", "ffec27", "00e436",
        "29adff", "83769c", "ff77a8", "ffccaa",
    ],
    "endesga32": [
        "be4a2f", "d77643", "ead4aa", "e4a672",
        "b86f50", "733e39", "3e2731", "a22633",
        "e43b44", "f77622", "feae34", "fee761",
        "63c74d", "3e8948", "265c42", "193c3e",
        "124e89", "0099db", "2ce8f5", "ffffff",
        "c0cbdc", "8b9bb4", "5a6988", "3a4466",
        "262b44", "181425", "ff0044", "68386c",
        "b55088", "f6757a", "e8b796", "c28569",
    ],
    "lospec500": [
        "10121c", "2c1e31", "6b2643", "ac2847",
        "ec273f", "94493a", "de5d3a", "e98537",
        "f3a833", "4d3533", "6e4c30", "a26d3f",
        "ce9248", "dab163", "e8d282", "f7f3b7",
        "1e4044", "295447", "44824a", "5ab55a",
        "9de64e", "108b7e", "3bb68f", "6ecf8e",
        "36486b", "4b80ca", "77b8d4", "a8e4f0",
        "223344", "3d5c5c", "6e8287", "9db3b3",
    ],
    "resurrect64": [
        "2e222f", "3e3546", "625565", "966c6c",
        "ab947a", "694f62", "7f708a", "9babb2",
        "c7dcd0", "ffffff", "6e2727", "b33831",
        "ea4f36", "f57d4a", "ae2334", "e83b3b",
        "fb6b1d", "f79617", "f9c22b", "7a3045",
        "9e4539", "cd683d", "e6904e", "fbb954",
        "4c3e24", "676633", "a2a947", "d5e04b",
        "fbff86", "165a4c", "239063", "1ebc73",
        "91db69", "cddf6c", "313638", "374e4a",
        "547e64", "92a984", "b2ba90", "0b5e65",
        "0b8a8f", "0eaf9b", "30e1b9", "8ff8e2",
        "323353", "484a77", "4d65b4", "4d9be6",
        "8fd3ff", "45293f", "6b3e75", "905ea9",
        "a884f3", "eaaded", "753c54", "a24b6f",
        "cf657f", "ed8099", "831c5d", "c32454",
        "f04f78", "f68181", "fca790", "fdcbb0",
    ],
}


class OutlineType(str, Enum):
    NONE = "none"
    BLACK = "black_1px"
    BLACK_2PX = "black_2px"
    COLORED = "colored_1px"
    SELECTIVE = "selective"     # outlines only on some edges
    GLOWING = "glowing"


class ShadingStyle(str, Enum):
    FLAT = "flat"
    DITHERED = "dithered"
    ANTI_ALIASED = "anti_aliased"
    PILLOW = "pillow"          # pillow-shaded / cel-shaded
    HUE_SHIFTED = "hue_shifted"


class SubjectType(str, Enum):
    CHARACTER = "character"
    MONSTER = "monster"
    ITEM = "item"
    TILE = "tile"
    ICON = "icon"
    UI = "ui"
    EFFECT = "effect"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class FacingDirection(str, Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    THREE_QUARTER = "three_quarter"
    TOP_DOWN = "top_down"
    ISOMETRIC = "isometric"
    NA = "na"


class InputStyle(str, Enum):
    PHOTOREALISTIC = "photorealistic"
    DIGITAL_PAINTING = "digital_painting"
    CONCEPT_ART = "concept_art"
    ANIME = "anime"
    VECTOR = "vector"


@dataclass
class AssetTags:
    """Full tag set for a single pixel art asset."""
    grid_size: int                                  # actual pixel grid (8,16,24,32,48,64)
    palette_count: int = 0                          # number of unique colours
    palette_hex: List[str] = field(default_factory=list)   # exact hex values
    palette_name: Optional[str] = None              # closest named palette if any
    outline: OutlineType = OutlineType.NONE
    shading: ShadingStyle = ShadingStyle.FLAT
    subject: SubjectType = SubjectType.UNKNOWN
    facing: FacingDirection = FacingDirection.NA
    source_set: str = ""                            # e.g. "dcss", "lpc", "kenney"
    width_cells: int = 0                            # sprite width in grid cells
    height_cells: int = 0                           # sprite height in grid cells
    transparent_bg: bool = False
    animation_frame: str = ""                       # e.g. "idle_1", "walk_3"

    def to_tag_string(self) -> str:
        """Produce the conditioning tag string for model training."""
        parts = [
            f"grid:{self.grid_size}px",
            f"outline:{self.outline.value}",
            f"shading:{self.shading.value}",
            f"palette_n:{self.palette_count}",
        ]
        if self.palette_name:
            parts.append(f"palette:{self.palette_name}")
        if self.subject != SubjectType.UNKNOWN:
            parts.append(f"subject:{self.subject.value}")
        if self.facing != FacingDirection.NA:
            parts.append(f"facing:{self.facing.value}")
        if self.width_cells and self.height_cells:
            parts.append(f"size:{self.width_cells}x{self.height_cells}")
        if self.source_set:
            parts.append(f"source:{self.source_set}")
        if self.transparent_bg:
            parts.append("bg:transparent")
        if self.animation_frame:
            parts.append(f"frame:{self.animation_frame}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "grid_size": int(self.grid_size),
            "palette_count": int(self.palette_count),
            "palette_hex": list(self.palette_hex),
            "palette_name": self.palette_name,
            "outline": self.outline.value,
            "shading": self.shading.value,
            "subject": self.subject.value,
            "facing": self.facing.value,
            "source_set": str(self.source_set),
            "width_cells": int(self.width_cells),
            "height_cells": int(self.height_cells),
            "transparent_bg": bool(self.transparent_bg),
            "animation_frame": str(self.animation_frame),
            "tag_string": self.to_tag_string(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetTags":
        return cls(
            grid_size=d["grid_size"],
            palette_count=d.get("palette_count", 0),
            palette_hex=d.get("palette_hex", []),
            palette_name=d.get("palette_name"),
            outline=OutlineType(d.get("outline", "none")),
            shading=ShadingStyle(d.get("shading", "flat")),
            subject=SubjectType(d.get("subject", "unknown")),
            facing=FacingDirection(d.get("facing", "na")),
            source_set=d.get("source_set", ""),
            width_cells=d.get("width_cells", 0),
            height_cells=d.get("height_cells", 0),
            transparent_bg=d.get("transparent_bg", False),
            animation_frame=d.get("animation_frame", ""),
        )


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    # Directories
    raw_dir: Path = Path("data/raw")          # scraped / downloaded originals
    sprites_dir: Path = Path("data/sprites")  # extracted individual sprites
    cleaned_dir: Path = Path("data/cleaned")  # pixel-perfect cleaned sprites
    pairs_dir: Path = Path("data/pairs")      # final training pairs
    prompts_dir: Path = Path("data/prompts")  # generated prompts for image gen

    # Manifest
    manifest_path: Path = Path("data/manifest.jsonl")

    # Cleaning
    min_sprite_px: int = MIN_SPRITE_PX
    max_sprite_px: int = MAX_SPRITE_PX
    use_hough: bool = True
    use_ml: bool = False

    # Tagging
    auto_tag: bool = True

    # Pair generation
    input_styles: List[InputStyle] = field(
        default_factory=lambda: [
            InputStyle.PHOTOREALISTIC,
            InputStyle.DIGITAL_PAINTING,
            InputStyle.CONCEPT_ART,
        ]
    )
    pairs_per_asset: int = 3

    def ensure_dirs(self):
        for d in [self.raw_dir, self.sprites_dir, self.cleaned_dir,
                  self.pairs_dir, self.prompts_dir]:
            d.mkdir(parents=True, exist_ok=True)
