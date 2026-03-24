"""
Tissue-filtered patch sampler for ABMIL training.

For each patient, loads pre-extracted UNI features (.pt) and coordinates (.npy),
filters tiles by tissue mask overlap (>=10%), and randomly samples N tiles.
Multi-slide patients are pooled into a single bag.

Optionally fits PCA on all train patients and reduces 1024-dim to a lower dimension.

Usage:
    from patch_sampler import sample_patient, load_train_set

    # Single patient (1024-dim):
    features, coords = sample_patient("1041", n_tiles=3000)
    # features: (3000, 1024) tensor

    # All train patients with PCA to 8D:
    patient_data = load_train_set(n_tiles=3000, pca_dim=8)
    # patient_data: list of (patient_id, bcr, features, coords)
    # features: (3000, 8) tensor
"""

import os
import numpy as np
import torch
import tifffile

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
FEAT_DIR = os.path.join(DATA_ROOT, "pathology/features/features")
COORD_DIR = os.path.join(DATA_ROOT, "pathology/features/coordinates")
IMG_DIR = os.path.join(DATA_ROOT, "pathology/images")

MASK_LEVEL = 4
SCALE = 64  # WSI level 0 -> mask level 4
TISSUE_THRESHOLD = 0.10


def get_tissue_mask(patient_id, slide_id):
    """Return boolean array: True for tiles with >=10% tissue overlap."""
    coord_path = os.path.join(COORD_DIR, f"{slide_id}.npy")
    mask_path = os.path.join(IMG_DIR, patient_id, f"{slide_id}_tissue.tif")

    coords = np.load(coord_path)
    mask = tifffile.imread(mask_path, key=MASK_LEVEL)
    tile_size_lv0 = int(coords["tile_size_lv0"][0])
    mt = max(tile_size_lv0 // SCALE, 1)

    n = len(coords)
    is_tissue = np.zeros(n, dtype=bool)

    for i in range(n):
        x, y = int(coords["x"][i]), int(coords["y"][i])
        mx, my = x // SCALE, y // SCALE
        mx2 = min(mx + mt, mask.shape[1])
        my2 = min(my + mt, mask.shape[0])
        if mx >= mask.shape[1] or my >= mask.shape[0]:
            continue
        if mask[my:my2, mx:mx2].mean() >= TISSUE_THRESHOLD:
            is_tissue[i] = True

    return is_tissue


def get_slide_ids(patient_id):
    """Find all slide IDs for a patient from the features directory."""
    return sorted(
        f.replace(".pt", "")
        for f in os.listdir(FEAT_DIR)
        if f.startswith(f"{patient_id}_") and f.endswith(".pt")
    )


SEED = 430


def sample_patient(patient_id, n_tiles=3000, seed=SEED):
    """
    Sample n_tiles tissue patches for a patient, pooled across all slides.

    Returns:
        features: (n_tiles, 1024) float32 tensor
        coords:   (n_tiles,) structured numpy array with x, y, etc.
    """
    patient_id = str(patient_id)
    slide_ids = get_slide_ids(patient_id)

    all_features = []
    all_coords = []

    for sid in slide_ids:
        feats = torch.load(
            os.path.join(FEAT_DIR, f"{sid}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        coords = np.load(os.path.join(COORD_DIR, f"{sid}.npy"))
        tissue_mask = get_tissue_mask(patient_id, sid)

        all_features.append(feats[tissue_mask])
        all_coords.append(coords[tissue_mask])

    all_features = torch.cat(all_features, dim=0)
    all_coords = np.concatenate(all_coords)

    n_available = len(all_features)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_available, size=min(n_tiles, n_available), replace=False)

    return all_features[idx], all_coords[idx]


def load_train_set(n_tiles=3000, pca_dim=None):
    """
    Load and sample all train patients. Optionally apply PCA.

    PCA is fit on the combined sampled tiles from all train patients,
    then applied per patient.

    Returns:
        list of (patient_id, bcr, features, coords) tuples.
        features shape: (n_tiles, pca_dim or 1024)
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_ROOT, "preliminary_split.csv"))
    train = df[df["split"] == "train"]

    # Sample all patients
    sampled = []
    for _, row in train.iterrows():
        pid = str(row["patient_id"])
        feats, coords = sample_patient(pid, n_tiles=n_tiles)
        sampled.append((pid, int(row["BCR"]), feats, coords))

    if pca_dim is not None:
        from sklearn.decomposition import PCA

        # Fit PCA on all train tiles combined
        all_feats = torch.cat([s[2] for s in sampled], dim=0).numpy()
        pca = PCA(n_components=pca_dim, random_state=SEED)
        pca.fit(all_feats)

        # Transform per patient and track per-patient variance
        transformed = []
        for pid, bcr, feats, coords in sampled:
            feats_np = feats.numpy()
            reduced = pca.transform(feats_np)
            recon = pca.inverse_transform(reduced)
            total_var = np.var(feats_np, axis=0).sum()
            recon_var = np.var(feats_np - recon, axis=0).sum()
            patient_retained = 1.0 - recon_var / total_var
            transformed.append((
                pid, bcr,
                torch.from_numpy(reduced).float(),
                coords,
                patient_retained,
            ))

        pca_info = {
            "global_variance_retained": float(pca.explained_variance_ratio_.sum()),
            "per_component": pca.explained_variance_ratio_.tolist(),
            "per_patient": {t[0]: t[4] for t in transformed},
        }

        sampled = [(pid, bcr, feats, coords) for pid, bcr, feats, coords, _ in transformed]
        return sampled, pca_info

    return sampled, None


def save_output(patient_data, pca_info, output_dir, n_tiles, pca_dim=None):
    """Save per-patient .pt files and optionally a variance summary .txt."""
    os.makedirs(output_dir, exist_ok=True)

    for pid, bcr, feats, coords in patient_data:
        torch.save(feats, os.path.join(output_dir, f"{pid}.pt"))

    if pca_info is None:
        # Raw features — just write a simple summary
        lines = [
            "Patch Sampler Output",
            "====================",
            f"Seed:           {SEED}",
            f"Tiles/patient:  {n_tiles}",
            f"Tissue filter:  >={TISSUE_THRESHOLD:.0%}",
            f"PCA:            none (raw 1024-dim)",
            "",
        ]
        for pid, bcr, feats, coords in patient_data:
            lines.append(f"  {pid} (BCR={bcr}): {tuple(feats.shape)}")
    else:
        lines = []
        lines.append(f"Patch Sampler Output")
        lines.append(f"====================")
        lines.append(f"Seed:           {SEED}")
        lines.append(f"Tiles/patient:  {n_tiles}")
        lines.append(f"Tissue filter:  >={TISSUE_THRESHOLD:.0%}")
        lines.append(f"PCA:            1024 -> {pca_dim}")
        lines.append(f"")
        lines.append(f"Global variance retained: {pca_info['global_variance_retained']:.4f} "
                     f"({pca_info['global_variance_retained']:.1%})")
        lines.append(f"")
        lines.append(f"Per-component variance:")
        for i, v in enumerate(pca_info["per_component"]):
            cumulative = sum(pca_info["per_component"][:i+1])
            lines.append(f"  PC{i+1}: {v:.4f} (cumulative: {cumulative:.4f})")
        lines.append(f"")
        lines.append(f"Per-patient variance retained:")
        for pid, bcr, feats, coords in patient_data:
            var = pca_info["per_patient"][pid]
            lines.append(f"  {pid} (BCR={bcr}): {var:.4f} ({var:.1%})")

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {len(patient_data)} .pt files to {output_dir}/")


if __name__ == "__main__":
    import sys
    import pickle
    import pandas as pd
    from datetime import datetime
    from sklearn.decomposition import PCA

    TILE_COUNTS = [3000, 50000]
    PCA_DIMS = [64, 8]
    FEAT_BASE = os.path.join(DATA_ROOT, "pathology/features")

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def dir_for(dim_label, n_tiles):
        suffix = "_50k" if n_tiles == 50000 else ""
        return os.path.join(FEAT_BASE, f"sampled_{dim_label}{suffix}")

    # Load all patients
    df = pd.read_csv(os.path.join(DATA_ROOT, "preliminary_split.csv"))
    train_ids = set(df[df["split"] == "train"]["patient_id"].astype(str))
    total = len(df)

    # ── Step 1: Sample 1024D for all patients at both tile counts ──
    for n_tiles in TILE_COUNTS:
        out_dir = dir_for("1024", n_tiles)
        os.makedirs(out_dir, exist_ok=True)
        log(f"── 1024D, {n_tiles} tiles → {out_dir}")

        for i, row in df.iterrows():
            pid = str(row["patient_id"])
            out_path = os.path.join(out_dir, f"{pid}.pt")

            if os.path.exists(out_path):
                log(f"  [{i+1}/{total}] {pid} — skipped (exists)")
                continue

            feats, coords = sample_patient(pid, n_tiles=n_tiles)
            torch.save(feats, out_path)
            log(f"  [{i+1}/{total}] {pid} — saved {tuple(feats.shape)}")

    # ── Step 2: Fit PCA on train patients (using 3k tiles) ──
    log("── Fitting PCA on train patients (3k tiles) ──")
    train_feats_list = []
    for pid in sorted(train_ids):
        feats = torch.load(
            os.path.join(dir_for("1024", 3000), f"{pid}.pt"),
            map_location="cpu", weights_only=True,
        )
        train_feats_list.append(feats)
    train_feats_all = torch.cat(train_feats_list, dim=0).numpy()
    log(f"  Train tile matrix: {train_feats_all.shape}")

    pca_objects = {}
    for pca_dim in PCA_DIMS:
        pca = PCA(n_components=pca_dim, random_state=SEED)
        pca.fit(train_feats_all)
        pca_objects[pca_dim] = pca

        pkl_path = os.path.join(FEAT_BASE, f"pca_{pca_dim}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(pca, f)
        log(f"  PCA {pca_dim}D: variance retained = {pca.explained_variance_ratio_.sum():.4f} — saved {pkl_path}")

    del train_feats_all  # free memory

    # ── Step 3: Apply PCA to all patients at both tile counts ──
    for pca_dim in PCA_DIMS:
        pca = pca_objects[pca_dim]
        for n_tiles in TILE_COUNTS:
            src_dir = dir_for("1024", n_tiles)
            out_dir = dir_for(f"pca{pca_dim}", n_tiles)
            os.makedirs(out_dir, exist_ok=True)
            log(f"── PCA {pca_dim}D, {n_tiles} tiles → {out_dir}")

            for i, row in df.iterrows():
                pid = str(row["patient_id"])
                out_path = os.path.join(out_dir, f"{pid}.pt")

                if os.path.exists(out_path):
                    log(f"  [{i+1}/{total}] {pid} — skipped (exists)")
                    continue

                feats = torch.load(
                    os.path.join(src_dir, f"{pid}.pt"),
                    map_location="cpu", weights_only=True,
                )
                reduced = pca.transform(feats.numpy())
                torch.save(torch.from_numpy(reduced).float(), out_path)
                log(f"  [{i+1}/{total}] {pid} — saved {tuple(reduced.shape)}")

    log("\n── Done! ──")
    log("Directories created:")
    for n_tiles in TILE_COUNTS:
        for label in ["1024"] + [f"pca{d}" for d in PCA_DIMS]:
            d = dir_for(label, n_tiles)
            n_files = len([f for f in os.listdir(d) if f.endswith(".pt")])
            log(f"  {d} — {n_files} files")
