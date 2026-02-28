"""Datamine DCSS item/monster descriptions and match them to tiles.

Parses the %%%%-delimited description files from the crawl source tree
(crawl-ref/source/dat/descript/) and fuzzy-matches them to tile filenames
from the crawl/tiles repo.

Workflow:
    1. Clone/download crawl source + tiles repos
    2. Parse description files (items.txt, monsters.txt, unrand.txt)
    3. Match tile filenames → game descriptions
    4. Generate an HTML curation gallery for the user to keep/reject tiles
    5. Export curated tile-description pairs as JSONL

Usage:
    python -m pipeline.crawl_descriptions \\
        --tiles-dir data/dcss-tiles/releases/Nov-2015 \\
        --crawl-dir data/crawl-source \\
        -o data/dcss-curated

    # Or fetch repos automatically:
    python -m pipeline.crawl_descriptions \\
        --fetch \\
        -o data/dcss-curated
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Description file parser
# ---------------------------------------------------------------------------

@dataclass
class GameDescription:
    """A single description entry from the DCSS source."""
    name: str                    # exact name from the file (e.g. "long sword")
    description: str             # full description text
    source_file: str             # e.g. "items.txt", "monsters.txt"
    category: str = ""           # derived: "weapon", "armour", "monster", etc.
    is_unique: bool = False      # unrandart or unique monster
    version_tag: str = ""        # e.g. "TAG_MAJOR_VERSION == 34"


def parse_description_file(filepath: Path) -> List[GameDescription]:
    """Parse a %%%%-delimited DCSS description file.

    Format:
        %%%%
        item name

        Description text that can span
        multiple lines.
        %%%%

    Lines starting with # are version tags or comments.
    Lua blocks {{ ... }} are stripped.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    entries = []

    # split on %%%%
    chunks = text.split("%%%%")
    source_file = filepath.name

    current_version_tag = ""

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = chunk.split("\n")

        # extract version tags and find the name line
        name = ""
        desc_lines = []
        found_name = False
        version_tag = current_version_tag

        for line in lines:
            stripped = line.strip()

            # version tag comment
            if stripped.startswith("# TAG_MAJOR_VERSION"):
                version_tag = stripped.lstrip("# ").strip()
                current_version_tag = version_tag
                continue

            # skip other comments
            if stripped.startswith("#"):
                continue

            # skip empty lines before name
            if not found_name:
                if stripped:
                    name = stripped
                    found_name = True
                continue

            # everything after the name is description
            desc_lines.append(line)

        if not name:
            continue

        description = "\n".join(desc_lines).strip()

        # strip Lua blocks {{ ... }}
        description = re.sub(r"\{\{.*?\}\}", "", description, flags=re.DOTALL).strip()

        # strip [[link]] markers
        description = re.sub(r"\[\[([^\]]+)\]\]", r"\1", description)

        if not description:
            continue

        entries.append(GameDescription(
            name=name,
            description=description,
            source_file=source_file,
            version_tag=version_tag,
        ))

    return entries


def parse_all_descriptions(crawl_dir: Path) -> Dict[str, GameDescription]:
    """Parse all description files and return a name → description mapping.

    Looks in crawl-ref/source/dat/descript/ for:
      - items.txt      → items (weapons, armour, potions, scrolls, etc.)
      - monsters.txt   → monsters
      - unrand.txt     → unique artifacts
      - spells.txt     → spells (for effect tiles)
      - features.txt   → dungeon features (for dngn tiles)
    """
    descript_dir = crawl_dir / "crawl-ref" / "source" / "dat" / "descript"
    if not descript_dir.exists():
        # maybe it's pointed directly at the descript dir
        if (crawl_dir / "items.txt").exists():
            descript_dir = crawl_dir
        else:
            raise FileNotFoundError(
                f"Could not find description files in {crawl_dir}. "
                f"Expected crawl-ref/source/dat/descript/ or direct path."
            )

    files_to_parse = [
        "items.txt", "monsters.txt", "unrand.txt",
        "spells.txt", "features.txt",
    ]

    all_descriptions: Dict[str, GameDescription] = {}

    for filename in files_to_parse:
        filepath = descript_dir / filename
        if not filepath.exists():
            logger.warning("Description file not found: %s", filepath)
            continue

        entries = parse_description_file(filepath)
        logger.info("Parsed %d entries from %s", len(entries), filename)

        # categorize based on source file
        category = filename.replace(".txt", "")

        for entry in entries:
            entry.category = category
            if filename == "unrand.txt":
                entry.is_unique = True

            # normalize the key for lookup
            key = _normalize_name(entry.name)
            all_descriptions[key] = entry

    logger.info("Total descriptions parsed: %d", len(all_descriptions))
    return all_descriptions


# ---------------------------------------------------------------------------
# Name normalization & matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Normalize a name for matching: lowercase, strip variants/articles."""
    name = name.lower().strip()
    # remove leading articles
    for article in ("a ", "an ", "the "):
        if name.startswith(article):
            name = name[len(article):]
    # underscores and hyphens → spaces
    name = name.replace("_", " ").replace("-", " ")
    # strip trailing numbers (tile variants)
    name = re.sub(r"\s*\d+$", "", name)
    # collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _tile_path_to_search_names(tile_path: str) -> List[str]:
    """Generate candidate description names from a tile filepath.

    For example:
        item/weapon/long_sword1.png →
            ["long sword", "long sword weapon"]
        item/potion/i-heal-wounds.png →
            ["heal wounds", "potion of heal wounds", "potion of healing"]
        mon/demons/shadow_demon.png →
            ["shadow demon"]
        item/armour/leather_armour1.png →
            ["leather armour", "leather armor"]
    """
    parts = Path(tile_path).parts
    stem = Path(tile_path).stem

    # strip trailing numbers
    stem_clean = re.sub(r"\d+$", "", stem).rstrip("_")

    # strip known prefixes
    # i- prefix means "identified" in DCSS potion/scroll/wand tiles
    if stem_clean.startswith("i-") or stem_clean.startswith("i_"):
        stem_clean = stem_clean[2:]

    # normalize
    name = stem_clean.replace("_", " ").replace("-", " ").strip()

    candidates = [name]

    # determine category from path
    category = parts[0] if len(parts) > 1 else ""
    subcategory = parts[1] if len(parts) > 2 else ""

    # add variants with category prefix
    if category == "item":
        if subcategory == "potion":
            candidates.append(f"potion of {name}")
        elif subcategory == "scroll":
            candidates.append(f"scroll of {name}")
        elif subcategory == "wand":
            candidates.append(f"wand of {name}")
        elif subcategory == "staff":
            candidates.append(f"staff of {name}")
        elif subcategory == "ring":
            candidates.append(f"ring of {name}")
        elif subcategory == "amulet":
            candidates.append(f"amulet of {name}")
        elif subcategory == "book":
            candidates.append(f"book of {name}")

    # British/American spelling variants
    if "armour" in name:
        candidates.append(name.replace("armour", "armor"))
    if "armor" in name:
        candidates.append(name.replace("armor", "armour"))

    # handle "scales" vs "dragon scales"
    if "dragon" in name and "scale" not in name:
        candidates.append(f"{name} scales")
        candidates.append(f"{name} dragon scales")

    return candidates


@dataclass
class TileMatch:
    """A matched tile-description pair."""
    tile_path: str                  # relative path within tiles dir
    tile_abs_path: str              # absolute path to tile PNG
    game_name: str                  # name from description file
    description: str                # full game description
    category: str                   # item category (weapon, monster, etc.)
    subcategory: str                # path-derived subcategory
    match_type: str                 # "exact", "prefix", "fuzzy"
    match_score: float              # 0.0 - 1.0
    is_variant: bool = False        # variant tile (sword2, sword3)
    variant_num: int = 0            # variant number (0 = base)
    tile_width: int = 0
    tile_height: int = 0
    keep: Optional[bool] = None     # user curation decision


def match_tiles_to_descriptions(
    tiles_dir: Path,
    descriptions: Dict[str, GameDescription],
    categories: Optional[List[str]] = None,
) -> List[TileMatch]:
    """Match tile files to game descriptions.

    Returns a list of TileMatch objects sorted by category and name.
    """
    from PIL import Image

    matches = []
    unmatched = []

    all_pngs = sorted(tiles_dir.rglob("*.png"))
    logger.info("Found %d PNG tiles to match", len(all_pngs))

    for png_path in all_pngs:
        rel_path = str(png_path.relative_to(tiles_dir))
        parts = Path(rel_path).parts

        # filter by category
        top_cat = parts[0] if len(parts) > 1 else ""
        if categories and top_cat not in categories:
            continue

        # skip UNUSED, gui, effect (less useful for training)
        if top_cat in ("UNUSED",):
            continue

        subcategory = parts[1] if len(parts) > 2 else ""

        # detect variant number
        stem = Path(rel_path).stem
        variant_match = re.search(r"(\d+)$", stem)
        variant_num = int(variant_match.group(1)) if variant_match else 0
        is_variant = variant_num > 1  # variant 1 or 0 is "base"

        # get tile size
        try:
            with Image.open(png_path) as img:
                tile_w, tile_h = img.size
        except Exception:
            tile_w, tile_h = 0, 0

        # skip tiny or huge tiles
        if tile_w < 16 or tile_h < 16 or tile_w > 128 or tile_h > 128:
            continue

        # try to match
        search_names = _tile_path_to_search_names(rel_path)
        best_match = None
        best_score = 0.0
        best_type = ""

        for candidate in search_names:
            norm = _normalize_name(candidate)

            # exact match
            if norm in descriptions:
                best_match = descriptions[norm]
                best_score = 1.0
                best_type = "exact"
                break

            # try prefix match (description name starts with our candidate)
            for key, desc in descriptions.items():
                if key.startswith(norm) and len(norm) > 3:
                    score = len(norm) / len(key)
                    if score > best_score and score > 0.6:
                        best_match = desc
                        best_score = score
                        best_type = "prefix"

                # also try if our candidate starts with the description name
                if norm.startswith(key) and len(key) > 3:
                    score = len(key) / len(norm)
                    if score > best_score and score > 0.6:
                        best_match = desc
                        best_score = score
                        best_type = "prefix"

        # fuzzy match as last resort
        if best_score < 0.6:
            primary_name = _normalize_name(search_names[0])
            for key, desc in descriptions.items():
                score = _fuzzy_score(primary_name, key)
                if score > best_score and score > 0.65:
                    best_match = desc
                    best_score = score
                    best_type = "fuzzy"

        if best_match:
            matches.append(TileMatch(
                tile_path=rel_path,
                tile_abs_path=str(png_path),
                game_name=best_match.name,
                description=best_match.description,
                category=top_cat,
                subcategory=subcategory,
                match_type=best_type,
                match_score=best_score,
                is_variant=is_variant,
                variant_num=variant_num,
                tile_width=tile_w,
                tile_height=tile_h,
            ))
        else:
            unmatched.append(rel_path)

    logger.info("Matched: %d tiles, Unmatched: %d tiles", len(matches), len(unmatched))
    if unmatched and len(unmatched) <= 20:
        for u in unmatched:
            logger.debug("  Unmatched: %s", u)
    elif unmatched:
        logger.debug("  First 20 unmatched: %s", unmatched[:20])

    # sort by category, subcategory, name
    matches.sort(key=lambda m: (m.category, m.subcategory, m.game_name, m.variant_num))

    return matches


def _fuzzy_score(a: str, b: str) -> float:
    """Simple token-overlap fuzzy matching score."""
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0

    # Dice coefficient
    return 2.0 * len(overlap) / (len(tokens_a) + len(tokens_b))


# ---------------------------------------------------------------------------
# HTML curation gallery
# ---------------------------------------------------------------------------

def _image_to_data_uri(img_path: str, max_size: int = 256) -> str:
    """Convert an image to a base64 data URI for embedding in HTML."""
    from PIL import Image
    import io

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGBA")
            # upscale small tiles with nearest-neighbor for visibility
            w, h = img.size
            scale = max(1, max_size // max(w, h))
            if scale > 1:
                img = img.resize((w * scale, h * scale), Image.NEAREST)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def generate_curation_gallery(
    matches: List[TileMatch],
    output_path: Path,
    embed_images: bool = True,
) -> Path:
    """Generate an HTML gallery for curating tile-description pairs.

    The gallery shows each tile with its matched description and lets the
    user check/uncheck tiles to keep or reject. The selections are saved
    to a JSON sidecar file.
    """
    # group by category/subcategory
    groups: Dict[str, List[TileMatch]] = {}
    for m in matches:
        key = f"{m.category}/{m.subcategory}" if m.subcategory else m.category
        groups.setdefault(key, []).append(m)

    html_parts = [_GALLERY_HTML_HEAD]

    html_parts.append('<div class="controls">')
    html_parts.append(f'<h1>DCSS Tile Curation Gallery ({len(matches)} tiles)</h1>')
    html_parts.append('<p>Click tiles to toggle keep/reject. Green = keep, Red = reject, Gray = undecided.</p>')
    html_parts.append('<div class="actions">')
    html_parts.append('<button onclick="selectAll()">Select All</button>')
    html_parts.append('<button onclick="deselectAll()">Deselect All</button>')
    html_parts.append('<button onclick="selectCategory(\'item\')">Select All Items</button>')
    html_parts.append('<button onclick="selectCategory(\'mon\')">Select All Monsters</button>')
    html_parts.append('<button onclick="toggleVariants()">Toggle Variants</button>')
    html_parts.append('<button onclick="exportSelections()">Export Selections (JSON)</button>')
    html_parts.append('<button onclick="loadSelections()">Load Selections</button>')
    html_parts.append('<span id="stats"></span>')
    html_parts.append('</div></div>')

    for group_name in sorted(groups.keys()):
        group = groups[group_name]
        html_parts.append(f'<h2 class="group-header" data-category="{group[0].category}">'
                          f'{group_name} ({len(group)} tiles)</h2>')
        html_parts.append('<div class="tile-grid">')

        for m in group:
            tile_id = m.tile_path.replace("/", "__").replace(".", "_")

            if embed_images:
                img_src = _image_to_data_uri(m.tile_abs_path)
            else:
                img_src = m.tile_abs_path

            match_badge = {
                "exact": '<span class="badge exact">exact</span>',
                "prefix": '<span class="badge prefix">prefix</span>',
                "fuzzy": '<span class="badge fuzzy">fuzzy</span>',
            }.get(m.match_type, "")

            variant_badge = f'<span class="badge variant">v{m.variant_num}</span>' if m.is_variant else ""

            # truncate description for display
            short_desc = m.description[:200] + "..." if len(m.description) > 200 else m.description

            html_parts.append(f'''
            <div class="tile-card undecided" id="{tile_id}"
                 data-path="{m.tile_path}"
                 data-category="{m.category}"
                 data-subcategory="{m.subcategory}"
                 data-variant="{str(m.is_variant).lower()}"
                 data-name="{m.game_name}"
                 onclick="toggleTile(this)">
                <img src="{img_src}" alt="{m.game_name}" />
                <div class="tile-info">
                    <strong>{m.game_name}</strong> {match_badge} {variant_badge}
                    <br><small>{m.tile_path}</small>
                    <br><small class="desc">{short_desc}</small>
                </div>
            </div>''')

        html_parts.append('</div>')

    html_parts.append(_GALLERY_HTML_SCRIPT)
    html_parts.append('</body></html>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")

    # also write the raw data as JSON for programmatic use
    json_path = output_path.with_suffix(".json")
    data = []
    for m in matches:
        data.append({
            "tile_path": m.tile_path,
            "tile_abs_path": m.tile_abs_path,
            "game_name": m.game_name,
            "description": m.description,
            "category": m.category,
            "subcategory": m.subcategory,
            "match_type": m.match_type,
            "match_score": m.match_score,
            "is_variant": m.is_variant,
            "variant_num": m.variant_num,
            "tile_width": m.tile_width,
            "tile_height": m.tile_height,
        })
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    logger.info("Gallery written to %s", output_path)
    logger.info("Data written to %s", json_path)

    return output_path


_GALLERY_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DCSS Tile Curation Gallery</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 20px;
}
.controls {
    position: sticky; top: 0; z-index: 100;
    background: #16213e; padding: 15px; border-radius: 8px;
    margin-bottom: 20px; border: 1px solid #333;
}
.controls h1 { font-size: 1.4em; margin-bottom: 8px; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
button {
    background: #0f3460; color: #e0e0e0; border: 1px solid #555;
    padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 0.9em;
}
button:hover { background: #1a5276; }
#stats { margin-left: auto; font-size: 0.9em; color: #aaa; }
h2.group-header {
    margin: 20px 0 10px; padding: 8px 12px;
    background: #16213e; border-radius: 6px; font-size: 1.1em;
    border-left: 4px solid #0f3460;
}
.tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}
.tile-card {
    display: flex; gap: 10px; padding: 10px;
    border-radius: 6px; cursor: pointer;
    transition: all 0.15s ease;
    border: 2px solid transparent;
}
.tile-card.undecided { background: #2a2a3e; border-color: #444; }
.tile-card.keep { background: #1a3a2a; border-color: #2ecc71; }
.tile-card.reject { background: #3a1a1a; border-color: #e74c3c; opacity: 0.5; }
.tile-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.tile-card img {
    width: 64px; height: 64px; object-fit: contain;
    image-rendering: pixelated; background: #111;
    border-radius: 4px; flex-shrink: 0;
}
.tile-info { font-size: 0.85em; overflow: hidden; }
.tile-info strong { color: #fff; }
.tile-info small { color: #888; }
.tile-info .desc { color: #aaa; display: block; margin-top: 4px; max-height: 3.6em; overflow: hidden; }
.badge {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 0.7em; font-weight: bold; vertical-align: middle;
}
.badge.exact { background: #27ae60; color: #fff; }
.badge.prefix { background: #f39c12; color: #000; }
.badge.fuzzy { background: #e74c3c; color: #fff; }
.badge.variant { background: #3498db; color: #fff; }
</style>
</head>
<body>
"""

_GALLERY_HTML_SCRIPT = """
<script>
const selections = {};

function toggleTile(el) {
    const path = el.dataset.path;
    if (el.classList.contains('undecided') || el.classList.contains('reject')) {
        el.classList.remove('undecided', 'reject');
        el.classList.add('keep');
        selections[path] = true;
    } else {
        el.classList.remove('keep');
        el.classList.add('reject');
        selections[path] = false;
    }
    updateStats();
}

function selectAll() {
    document.querySelectorAll('.tile-card').forEach(el => {
        el.classList.remove('undecided', 'reject');
        el.classList.add('keep');
        selections[el.dataset.path] = true;
    });
    updateStats();
}

function deselectAll() {
    document.querySelectorAll('.tile-card').forEach(el => {
        el.classList.remove('undecided', 'keep');
        el.classList.add('reject');
        selections[el.dataset.path] = false;
    });
    updateStats();
}

function selectCategory(cat) {
    document.querySelectorAll(`.tile-card[data-category="${cat}"]`).forEach(el => {
        el.classList.remove('undecided', 'reject');
        el.classList.add('keep');
        selections[el.dataset.path] = true;
    });
    updateStats();
}

function toggleVariants() {
    document.querySelectorAll('.tile-card[data-variant="true"]').forEach(el => {
        if (el.classList.contains('keep')) {
            el.classList.remove('keep');
            el.classList.add('reject');
            selections[el.dataset.path] = false;
        } else {
            el.classList.remove('undecided', 'reject');
            el.classList.add('keep');
            selections[el.dataset.path] = true;
        }
    });
    updateStats();
}

function updateStats() {
    const kept = Object.values(selections).filter(v => v).length;
    const rejected = Object.values(selections).filter(v => !v).length;
    const total = document.querySelectorAll('.tile-card').length;
    const undecided = total - kept - rejected;
    document.getElementById('stats').textContent =
        `${kept} kept / ${rejected} rejected / ${undecided} undecided (${total} total)`;
}

function exportSelections() {
    const data = JSON.stringify(selections, null, 2);
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dcss_curation_selections.json';
    a.click();
    URL.revokeObjectURL(url);
}

function loadSelections() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (ev) => {
            const loaded = JSON.parse(ev.target.result);
            Object.assign(selections, loaded);
            // apply to DOM
            document.querySelectorAll('.tile-card').forEach(el => {
                const path = el.dataset.path;
                if (path in selections) {
                    el.classList.remove('undecided', 'keep', 'reject');
                    el.classList.add(selections[path] ? 'keep' : 'reject');
                }
            });
            updateStats();
        };
        reader.readAsText(file);
    };
    input.click();
}

updateStats();
</script>
"""


# ---------------------------------------------------------------------------
# Export curated pairs
# ---------------------------------------------------------------------------

def apply_curation(
    matches: List[TileMatch],
    selections_path: Path,
) -> List[TileMatch]:
    """Apply user curation selections to the match list.

    The selections file is a JSON dict of {tile_path: bool}.
    """
    selections = json.loads(selections_path.read_text())

    curated = []
    for m in matches:
        if m.tile_path in selections:
            m.keep = selections[m.tile_path]
        if m.keep is True:
            curated.append(m)

    logger.info(
        "Curation applied: %d kept out of %d total",
        len(curated), len(matches),
    )
    return curated


def export_curated_pairs(
    curated: List[TileMatch],
    output_dir: Path,
) -> Path:
    """Export curated tile-description pairs as a JSONL manifest.

    Copies tiles into the output directory and creates a manifest
    suitable for feeding into the existing pipeline (ingest_dcss).
    """
    import shutil
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_out = output_dir / "tiles"
    tiles_out.mkdir(exist_ok=True)
    manifest_path = output_dir / "curated_manifest.jsonl"

    entries = []
    for m in curated:
        # copy tile preserving subdirectory structure
        rel = Path(m.tile_path)
        dest = tiles_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(m.tile_abs_path, dest)

        # get dimensions
        try:
            with Image.open(str(dest)) as img:
                w, h = img.size
        except Exception:
            w, h = m.tile_width, m.tile_height

        entries.append({
            "tile_path": str(rel),
            "game_name": m.game_name,
            "description": m.description,
            "category": m.category,
            "subcategory": m.subcategory,
            "match_type": m.match_type,
            "match_score": m.match_score,
            "is_variant": m.is_variant,
            "variant_num": m.variant_num,
            "tile_width": w,
            "tile_height": h,
        })

    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info("Exported %d curated pairs → %s", len(entries), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Repo fetching helpers
# ---------------------------------------------------------------------------

def fetch_tiles_repo(dest: Path) -> Path:
    """Clone the DCSS tiles repo (shallow, ~50MB)."""
    if (dest / ".git").exists():
        logger.info("Tiles repo already exists at %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning crawl/tiles → %s", dest)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/crawl/tiles.git", str(dest)],
        check=True,
    )
    return dest


def fetch_crawl_descriptions(dest: Path) -> Path:
    """Sparse-checkout only the description files from crawl/crawl."""
    descript_dest = dest / "crawl-ref" / "source" / "dat" / "descript"
    if descript_dest.exists() and any(descript_dest.glob("*.txt")):
        logger.info("Crawl descriptions already exist at %s", descript_dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # use sparse checkout to only get the description files
    logger.info("Fetching crawl descriptions via sparse checkout → %s", dest)

    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none",
         "--sparse", "https://github.com/crawl/crawl.git", str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set",
         "crawl-ref/source/dat/descript"],
        cwd=str(dest),
        check=True,
    )

    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Datamine DCSS tile-description pairs for the pixel art pipeline.",
    )
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for curated pairs")
    parser.add_argument("--tiles-dir", default=None,
                        help="Path to DCSS tiles (e.g. data/dcss-tiles/releases/Nov-2015)")
    parser.add_argument("--crawl-dir", default=None,
                        help="Path to crawl source (for description files)")
    parser.add_argument("--fetch", action="store_true",
                        help="Automatically clone tiles + crawl repos")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only process these tile categories (e.g. item mon)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Don't embed images in HTML (use file paths)")
    parser.add_argument("--selections", default=None,
                        help="Path to curation selections JSON (to apply and export)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # resolve repos
    if args.fetch:
        tiles_base = fetch_tiles_repo(output_dir / "repos" / "tiles")
        crawl_base = fetch_crawl_descriptions(output_dir / "repos" / "crawl")
        tiles_dir = tiles_base / "releases" / "Nov-2015"
    else:
        if not args.tiles_dir or not args.crawl_dir:
            parser.error("Either --fetch or both --tiles-dir and --crawl-dir are required")
        tiles_dir = Path(args.tiles_dir)
        crawl_base = Path(args.crawl_dir)

    # parse descriptions
    descriptions = parse_all_descriptions(crawl_base)

    # match tiles
    matches = match_tiles_to_descriptions(
        tiles_dir, descriptions,
        categories=args.categories,
    )

    # print summary
    from collections import Counter
    cat_counts = Counter(m.category for m in matches)
    type_counts = Counter(m.match_type for m in matches)
    logger.info("Match summary by category: %s", dict(cat_counts))
    logger.info("Match summary by type: %s", dict(type_counts))

    # if selections file provided, apply curation and export
    if args.selections:
        curated = apply_curation(matches, Path(args.selections))
        export_curated_pairs(curated, output_dir)
    else:
        # generate curation gallery
        gallery_path = output_dir / "curation_gallery.html"
        generate_curation_gallery(
            matches, gallery_path,
            embed_images=not args.no_embed,
        )

        # also export all matches as JSON (for later curation)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Open %s in a browser", gallery_path)
        logger.info("  2. Click tiles to keep or reject")
        logger.info("  3. Export selections JSON")
        logger.info("  4. Re-run with --selections <path> to export curated pairs:")
        logger.info("     python -m pipeline.crawl_descriptions \\")
        logger.info("         --tiles-dir %s \\", tiles_dir)
        logger.info("         --crawl-dir %s \\", crawl_base)
        logger.info("         --selections dcss_curation_selections.json \\")
        logger.info("         -o %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
