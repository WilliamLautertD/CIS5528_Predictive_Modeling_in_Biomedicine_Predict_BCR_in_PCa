"""
Extract thumbnail and level-3 previews from all WSI pyramidal TIFFs.

Outputs:
  previews/thumb/<name>.png   — small thumbnail (~1024px long edge)
  previews/level3/<name>.png  — 64x downsampled overview
"""

from pathlib import Path
import openslide

WSI_DIR = Path("data/pathology/images")
OUT_DIR = Path("previews")
THUMB_SIZE = (1024, 1024)   # fits within this box, aspect ratio preserved

thumb_dir = OUT_DIR / "thumb"
level3_dir = OUT_DIR / "level3"
thumb_dir.mkdir(parents=True, exist_ok=True)
level3_dir.mkdir(parents=True, exist_ok=True)

wsi_files = sorted(
    p for p in WSI_DIR.rglob("*.tif") if "tissue" not in p.name
)
print(f"Found {len(wsi_files)} WSI files\n")

errors = []
for i, wsi_path in enumerate(wsi_files, 1):
    name = wsi_path.stem  # e.g. "1003_1"
    print(f"[{i:3d}/{len(wsi_files)}] {name}", end=" ... ", flush=True)

    try:
        slide = openslide.OpenSlide(str(wsi_path))

        # --- thumbnail ---
        thumb_path = thumb_dir / f"{name}.png"
        if thumb_path.exists():
            print("thumb exists,", end=" ", flush=True)
        else:
            thumb = slide.get_thumbnail(THUMB_SIZE)
            thumb.save(thumb_path)

        # --- level 3 (64x downsample) ---
        level3_path = level3_dir / f"{name}.png"
        if level3_path.exists():
            print("level3 exists", flush=True)
        else:
            # Use level 3 if available, otherwise fall back to the coarsest level
            target_level = min(3, slide.level_count - 1)
            dims = slide.level_dimensions[target_level]
            region = slide.read_region((0, 0), target_level, dims)
            region.convert("RGB").save(level3_path)
            actual_ds = round(slide.level_downsamples[target_level])
            print(f"done (level {target_level}, {actual_ds}x, {dims[0]}x{dims[1]}px)", flush=True)

        slide.close()

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        errors.append((name, str(e)))

print(f"\nDone. Saved to {OUT_DIR}/")
if errors:
    print(f"\n{len(errors)} errors:")
    for name, err in errors:
        print(f"  {name}: {err}")
