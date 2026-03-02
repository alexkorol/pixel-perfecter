#!/usr/bin/env python3
"""Visual test for sprite sheet cutting pipeline.

Runs extract_sprites() on multi-sprite input images and generates:
  1. Individual sprite PNGs saved to disk (upscaled for visibility)
  2. Annotated source images showing bounding boxes
  3. A lightweight HTML contact sheet for phone inspection

Usage:
    python tests/test_sprite_cutting_visual.py [--images path1 path2 ...]
    python tests/test_sprite_cutting_visual.py  # runs on multi-sprite input/ images
    python tests/test_sprite_cutting_visual.py --all  # runs on ALL input/ images
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.sheet_splitter import (
    ExtractedSprite,
    _detect_grid_layout,
    _estimate_strategy,
    deduplicate_sprites,
    extract_sprites,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "sprite_cutting_output"

MAX_SPRITES_DISPLAY = 12  # max sprites to save/show per strategy per image


def save_sprite_png(sprite: ExtractedSprite, path: Path, upscale_target: int = 128):
    """Save a sprite as an upscaled PNG for easy viewing on phone."""
    if sprite.image.shape[2] == 4:
        pil = Image.fromarray(sprite.image, "RGBA")
    else:
        pil = Image.fromarray(sprite.image, "RGB")

    w, h = pil.size
    scale = max(1, min(upscale_target // max(w, h), 10))
    if scale > 1:
        pil = pil.resize((w * scale, h * scale), Image.NEAREST)

    # Composite on checkerboard for transparency visibility
    if pil.mode == "RGBA":
        checker = Image.new("RGB", pil.size)
        block = max(4, scale * 2)
        pixels = checker.load()
        for y in range(pil.size[1]):
            for x in range(pil.size[0]):
                if (y // block + x // block) % 2 == 0:
                    pixels[x, y] = (45, 45, 55)
                else:
                    pixels[x, y] = (35, 35, 45)
        checker.paste(pil, mask=pil.split()[3])
        checker.save(path)
    else:
        pil.save(path)


def save_annotated_source(image_path: str, sprites: list, out_path: Path,
                          max_px: int = 600):
    """Save source image with colored bounding boxes drawn for each sprite."""
    pil = Image.open(image_path).convert("RGBA")
    img = np.array(pil)
    h, w = img.shape[:2]

    # Composite on dark background
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    rgb = img[:, :, :3].astype(np.float32)
    composited = (rgb * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    colors = [
        (255, 50, 50), (50, 255, 50), (50, 100, 255),
        (255, 255, 50), (255, 50, 255), (50, 255, 255),
        (255, 150, 0), (150, 0, 255), (0, 200, 128),
        (255, 200, 100), (100, 200, 255), (200, 100, 255),
    ]
    thickness = max(2, min(w, h) // 200)
    font_scale = max(0.4, min(w, h) / 800)
    for i, sprite in enumerate(sprites):
        x, y, sw, sh = sprite.bbox
        color = colors[i % len(colors)]
        cv2.rectangle(composited, (x, y), (x + sw, y + sh), color, thickness)
        cv2.putText(composited, str(i), (x + 3, y + int(20 * font_scale) + 3),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(composited, str(i), (x + 2, y + int(20 * font_scale) + 2),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    pil_out = Image.fromarray(composited, "RGB")
    ow, oh = pil_out.size
    if max(ow, oh) > max_px:
        ratio = max_px / max(ow, oh)
        pil_out = pil_out.resize((int(ow * ratio), int(oh * ratio)), Image.NEAREST)

    pil_out.save(out_path)


def run_extraction(image_path: str, output_subdir: Path) -> dict:
    """Run sprite extraction with all strategies and save results to disk."""
    path = Path(image_path)
    pil = Image.open(path)
    w, h = pil.size

    if pil.mode == "RGBA":
        img_np = np.array(pil)
    elif pil.mode == "P":
        img_np = np.array(pil.convert("RGBA"))
    else:
        img_np = np.array(pil.convert("RGB"))

    strategy = _estimate_strategy(img_np)
    grid = _detect_grid_layout(img_np)

    output_subdir.mkdir(parents=True, exist_ok=True)

    strategies_to_test = ["contours", "components"]

    all_results = {}
    for strat in strategies_to_test:
        t0 = time.time()
        sprites = extract_sprites(str(path), strategy=strat)
        elapsed = time.time() - t0
        unique = deduplicate_sprites(sprites)

        # Save annotated source
        ann_path = output_subdir / f"annotated_{strat}.png"
        save_annotated_source(str(path), unique[:MAX_SPRITES_DISPLAY], ann_path)

        # Save individual sprites
        sprite_paths = []
        for i, sprite in enumerate(unique[:MAX_SPRITES_DISPLAY]):
            sp = output_subdir / f"{strat}_{i:03d}.png"
            save_sprite_png(sprite, sp)
            sh, sw = sprite.image.shape[:2]
            sprite_paths.append({
                "file": sp.name,
                "index": sprite.index,
                "size": (sw, sh),
                "bbox": sprite.bbox,
            })

        all_results[strat] = {
            "count": len(unique),
            "time_ms": int(elapsed * 1000),
            "sprites": sprite_paths,
            "annotated": ann_path.name,
        }

    # Also run auto
    t0 = time.time()
    sprites_auto = extract_sprites(str(path), strategy="auto")
    t_auto = time.time() - t0
    unique_auto = deduplicate_sprites(sprites_auto)

    ann_auto = output_subdir / "annotated_auto.png"
    save_annotated_source(str(path), unique_auto[:MAX_SPRITES_DISPLAY], ann_auto)

    auto_sprite_paths = []
    for i, sprite in enumerate(unique_auto[:MAX_SPRITES_DISPLAY]):
        sp = output_subdir / f"auto_{i:03d}.png"
        save_sprite_png(sprite, sp)
        sh, sw = sprite.image.shape[:2]
        auto_sprite_paths.append({
            "file": sp.name,
            "index": sprite.index,
            "size": (sw, sh),
            "bbox": sprite.bbox,
        })

    all_results["auto"] = {
        "count": len(unique_auto),
        "time_ms": int(t_auto * 1000),
        "sprites": auto_sprite_paths,
        "annotated": ann_auto.name,
    }

    # Save a copy of the source (downscaled for HTML)
    source_thumb = output_subdir / "source.png"
    src_pil = Image.open(path).convert("RGB")
    sw, sh = src_pil.size
    if max(sw, sh) > 400:
        ratio = 400 / max(sw, sh)
        src_pil = src_pil.resize((int(sw * ratio), int(sh * ratio)), Image.NEAREST)
    src_pil.save(source_thumb)

    oversplit = (strategy == "grid" and all_results["auto"]["count"] > 20
                 and all_results["contours"]["count"] > 0
                 and all_results["auto"]["count"] / max(1, all_results["contours"]["count"]) > 10)

    return {
        "filename": path.name,
        "subdir": output_subdir.name,
        "size": (w, h),
        "detected_strategy": strategy,
        "detected_grid": grid,
        "strategies": all_results,
        "oversplit": oversplit,
    }


def generate_html_report(results: list, output_dir: Path):
    """Generate a lightweight HTML report referencing saved sprite files."""
    html_path = output_dir / "report.html"

    total_images = len(results)
    oversplit_count = sum(1 for r in results if r["oversplit"])

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sprite Cutting Test</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,system-ui,sans-serif;background:#111;color:#ddd;padding:10px}}
h1{{text-align:center;color:#0df;margin:8px 0 12px;font-size:1.3em}}
.sum{{background:#1a1a2e;border-radius:8px;padding:10px;margin-bottom:14px;font-size:.9em}}
.sum p{{margin:3px 0}}
.warn{{color:#f44;font-weight:bold}}
.card{{background:#16213e;border-radius:10px;padding:12px;margin-bottom:16px;border:1px solid #0f3460}}
.card h2{{color:#ff6b9d;font-size:1em;margin-bottom:6px;word-break:break-all}}
.meta{{font-size:.8em;color:#88a;margin-bottom:8px;display:flex;flex-wrap:wrap;gap:4px}}
.meta span{{background:#0f3460;padding:2px 6px;border-radius:3px}}
.section{{margin:8px 0}}
.section-title{{font-weight:bold;color:#0df;font-size:.85em;margin:6px 0 4px}}
.ann{{text-align:center;margin:4px 0}}
.ann img{{max-width:100%;border-radius:4px;border:1px solid #333}}
.sprites{{display:flex;flex-wrap:wrap;gap:6px}}
.sp{{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:4px;text-align:center}}
.sp img{{display:block;margin:0 auto 2px;image-rendering:pixelated;max-width:100px;max-height:100px}}
.sp .inf{{font-size:.65em;color:#8b949e}}
.ok{{color:#3fb950}} .bad{{color:#f85149}} .meh{{color:#d29922}}
hr{{border:none;border-top:1px solid #333;margin:8px 0}}
</style>
</head>
<body>
<h1>Sprite Cutting Test</h1>
<div class="sum">
<p><strong>{total_images}</strong> images tested</p>
"""]

    # Summary counts per strategy
    for strat in ["auto", "contours", "components"]:
        total = sum(r["strategies"][strat]["count"] for r in results)
        parts.append(f'<p>{strat}: <strong>{total}</strong> total sprites</p>\n')

    if oversplit_count:
        parts.append(f'<p class="warn">{oversplit_count} images OVER-SPLIT by grid detection '
                     f'(pixel grid confused with sprite grid)</p>\n')
    parts.append('</div>\n')

    for r in results:
        fname = r["filename"]
        short = fname[:45] + "..." if len(fname) > 45 else fname
        w, h = r["size"]
        strategy = r["detected_strategy"]
        grid = r["detected_grid"]
        subdir = r["subdir"]
        oversplit = r["oversplit"]

        auto_n = r["strategies"]["auto"]["count"]
        cont_n = r["strategies"]["contours"]["count"]
        comp_n = r["strategies"]["components"]["count"]

        if oversplit:
            badge = '<span class="bad">OVER-SPLIT</span>'
        elif cont_n >= 2 or comp_n >= 2:
            badge = f'<span class="ok">OK</span>'
        elif auto_n == 1:
            badge = '<span class="meh">single</span>'
        else:
            badge = '<span class="bad">0?</span>'

        parts.append(f'<div class="card">\n<h2>{short} {badge}</h2>\n')
        parts.append(f'<div class="meta">')
        parts.append(f'<span>{w}x{h}</span>')
        parts.append(f'<span>auto: {strategy}</span>')
        if grid:
            parts.append(f'<span>grid: {grid[0]}x{grid[1]}px</span>')
        parts.append(f'<span>auto:{auto_n} cont:{cont_n} comp:{comp_n}</span>')
        parts.append(f'</div>\n')

        # Source thumbnail
        parts.append(f'<div class="ann"><img src="{subdir}/source.png" alt="source"></div>\n')

        # Show strategies (contours first since it usually works best)
        strat_order = ["contours", "components", "auto"]
        for strat in strat_order:
            sr = r["strategies"][strat]
            n = sr["count"]
            ms = sr["time_ms"]
            shown = len(sr["sprites"])
            trunc = f" (showing {shown}/{n})" if shown < n else ""

            label_class = "bad" if (strat == "auto" and oversplit) else ""
            parts.append(f'<div class="section">\n')
            parts.append(f'<div class="section-title">'
                         f'<span class="{label_class}">{strat}</span> '
                         f'- {n} sprites{trunc} ({ms}ms)</div>\n')

            # Annotated image
            parts.append(f'<div class="ann"><img src="{subdir}/{sr["annotated"]}" alt="{strat} boxes"></div>\n')

            # Sprite grid
            parts.append(f'<div class="sprites">\n')
            for sp in sr["sprites"]:
                sw, sh = sp["size"]
                x, y, bw, bh = sp["bbox"]
                parts.append(f'<div class="sp">')
                parts.append(f'<img src="{subdir}/{sp["file"]}">')
                parts.append(f'<div class="inf">#{sp["index"]} {sw}x{sh}</div>')
                parts.append(f'</div>\n')
            parts.append(f'</div>\n</div>\n')

        parts.append(f'</div>\n')

    parts.append("</body></html>\n")

    html_path.write_text("".join(parts))
    print(f"HTML report: {html_path}")
    print(f"  Size: {html_path.stat().st_size / 1024:.0f} KB")
    return html_path


def find_multi_sprite_images(input_dir: Path) -> list:
    """Find images that are likely multi-sprite sheets."""
    sheet_keywords = [
        "Icons", "Assets", "Grid", "Atlas", "Tiles", "Textures",
        "Monsters", "Ingredients", "Fantasy Pixel-Art",
    ]
    candidates = []
    for f in sorted(input_dir.glob("*.png")):
        if any(kw.lower() in f.name.lower() for kw in sheet_keywords):
            candidates.append(str(f))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Visual sprite cutting test")
    parser.add_argument("--images", nargs="*", help="Specific image paths to test")
    parser.add_argument("--all", action="store_true", help="Test ALL images in input/")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    if args.images:
        image_paths = args.images
    elif args.all:
        image_paths = sorted(str(f) for f in INPUT_DIR.glob("*.png"))
    else:
        image_paths = find_multi_sprite_images(INPUT_DIR)

    if not image_paths:
        print("No images found. Use --images or --all.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing sprite cutting on {len(image_paths)} images...")
    print(f"Output dir: {output_dir}")
    results = []
    for i, path in enumerate(image_paths):
        fname = Path(path).name
        short = fname[:55] + "..." if len(fname) > 55 else fname
        # Create a clean subdir name from the filename
        subdir_name = f"img_{i:02d}"
        subdir = output_dir / subdir_name
        print(f"  [{i+1}/{len(image_paths)}] {short}...", end=" ", flush=True)
        try:
            result = run_extraction(path, subdir)
            results.append(result)
            auto_n = result["strategies"]["auto"]["count"]
            cont_n = result["strategies"]["contours"]["count"]
            comp_n = result["strategies"]["components"]["count"]
            flag = " *** OVER-SPLIT" if result["oversplit"] else ""
            print(f"auto={auto_n} cont={cont_n} comp={comp_n} "
                  f"[{result['detected_strategy']}]{flag}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    html_path = generate_html_report(results, output_dir)
    print(f"\nDone! Open {html_path} to inspect results.")


if __name__ == "__main__":
    main()
