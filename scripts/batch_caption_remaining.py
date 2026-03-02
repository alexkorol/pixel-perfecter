#!/usr/bin/env python3
"""Batch-caption all remaining uncaptioned *_up.png sprites via OpenRouter."""

import sys
from pathlib import Path

# ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from pipeline.captioner import caption_sprite
from PIL import Image
import numpy as np
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["CAPTIONER_BACKEND"] = "openrouter"

CLEANED_DIR = PROJECT_ROOT / "data" / "artist-ref" / "cleaned"
CACHE_DIR = PROJECT_ROOT / "data" / "artist-ref" / "captions"
MODEL = "google/gemini-2.0-flash-001"
RATE_LIMIT = 1.5


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_sprites = sorted(CLEANED_DIR.rglob("*_up.png"))
    total_found = len(all_sprites)
    logger.info("Found %d total *_up.png sprites", total_found)

    existing_captions = {
        p.stem.replace("_caption", "")
        for p in CACHE_DIR.glob("*_caption.json")
    }
    to_caption = [f for f in all_sprites if f.stem not in existing_captions]

    logger.info("Already captioned: %d", total_found - len(to_caption))
    logger.info("Remaining to caption: %d", len(to_caption))

    if not to_caption:
        logger.info("All sprites already captioned.")
        return

    success_count = 0
    error_count = 0
    start_time = time.time()

    for i, fpath in enumerate(to_caption):
        if i > 0:
            time.sleep(RATE_LIMIT)

        try:
            pil = Image.open(fpath)
            if pil.mode == "RGBA":
                img = np.array(pil)
            elif pil.mode == "P":
                img = np.array(pil.convert("RGBA"))
            else:
                img = np.array(pil.convert("RGB"))

            caption = caption_sprite(
                img,
                source_path=str(fpath),
                backend="openrouter",
                model=MODEL,
                cache_dir=CACHE_DIR,
            )
            success_count += 1

            if success_count % 10 == 0 or success_count == 1:
                elapsed = time.time() - start_time
                rate = success_count / elapsed if elapsed > 0 else 0
                eta = (len(to_caption) - (i + 1)) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %s -> \"%s\" (%.1f/min, ETA: %.0fm%.0fs)",
                    i + 1, len(to_caption),
                    fpath.name,
                    caption.short_description[:60],
                    rate * 60,
                    eta // 60, eta % 60,
                )

        except KeyboardInterrupt:
            logger.info("Interrupted.")
            break
        except Exception as e:
            error_count += 1
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(to_caption), fpath.name, e)
            continue

    elapsed = time.time() - start_time
    total_cached = len(list(CACHE_DIR.glob("*_caption.json")))
    logger.info("DONE: %d success, %d errors, %d total cached / %d sprites (%.1f min)",
                success_count, error_count, total_cached, total_found, elapsed / 60)


if __name__ == "__main__":
    main()
