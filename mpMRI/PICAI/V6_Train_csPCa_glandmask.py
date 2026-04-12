import os
import glob
import json
import copy
import random
from datetime import datetime

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# =========================
# 1) CONFIG
# =========================
ROOT_DIR = "images"
LABELS_CSV = "marksheet-2.csv"
GLAND_MASK_DIR = "picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23"
OUT_DIR = "outputs_csPCa_mri_cv_PROG3_gpu"
TB_DIR = os.path.join(OUT_DIR, "tensorboard")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SHAPE = (48, 96, 96)
EMBED_DIM = 64
SEED = 42
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 8

# HPC
NUM_WORKERS = 8
PIN_MEMORY = torch.cuda.is_available()

# Reduced pooling
POOL1_KERNEL = (1, 2, 2)
POOL2_KERNEL = (1, 2, 2)
POOL3_KERNEL = (2, 2, 2)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


# =========================
# 2) REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# 3) IMAGE HELPERS
# =========================
def read_image(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr



def zscore_nonzero(x):
    mask = x != 0
    if mask.sum() == 0:
        return x.astype(np.float32)
    vals = x[mask]
    mean = vals.mean()
    std = vals.std() + 1e-8
    y = x.copy()
    y[mask] = (vals - mean) / std
    return y.astype(np.float32)



def get_bbox(mask, margin=8):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        d, h, w = mask.shape
        return (0, d, 0, h, 0, w)

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)

    zmax = min(mask.shape[0], zmax + margin)
    ymax = min(mask.shape[1], ymax + margin)
    xmax = min(mask.shape[2], xmax + margin)

    return (zmin, zmax, ymin, ymax, xmin, xmax)



def crop_to_bbox(vol, bbox):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return vol[zmin:zmax, ymin:ymax, xmin:xmax]



def resize_3d_torch(volume, target_shape, mode="trilinear"):
    x = torch.from_numpy(volume).unsqueeze(0).float()  # (1,C,D,H,W)
    if mode in ["trilinear", "bilinear"]:
        x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=target_shape, mode=mode)
    return x.squeeze(0).cpu().numpy()


# =========================
# 4) LABELS + FILE DISCOVERY
# =========================
def load_case_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"patient_id", "study_id", "case_csPCa"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["patient_id", "study_id", "case_csPCa"]].copy()
    df = df.dropna(subset=["patient_id", "study_id", "case_csPCa"])
    df["patient_id"] = df["patient_id"].astype(int).astype(str)
    df["study_id"] = df["study_id"].astype(int).astype(str)
    df["label"] = df["case_csPCa"].astype(int)

    bad_labels = sorted(set(df[~df["label"].isin([0, 1])]["label"].tolist()))
    if bad_labels:
        raise ValueError(f"Found non-binary case_csPCa labels: {bad_labels}")

    df["case_id"] = df["patient_id"]
    df["case_key"] = df["patient_id"] + "_" + df["study_id"]
    return df



def pick_single_file(case_dir, suffix):
    matches = glob.glob(os.path.join(case_dir, f"*_{suffix}"))
    if len(matches) != 1:
        return None
    return matches[0]



def find_case_files(root_dir, mask_dir, labels_df):
    items = []

    for row in labels_df.itertuples(index=False):
        case_id = row.case_id
        case_key = row.case_key
        label = int(row.label)
        case_dir = os.path.join(root_dir, case_id)

        if not os.path.isdir(case_dir):
            print(f"[WARN] Missing folder for case {case_id}")
            continue

        t2w = pick_single_file(case_dir, "t2w.mha")
        adc = pick_single_file(case_dir, "adc.mha")
        hbv = pick_single_file(case_dir, "hbv.mha")
        mask = os.path.join(mask_dir, f"{case_key}.nii.gz")

        missing_parts = []
        if t2w is None:
            missing_parts.append("t2w")
        if adc is None:
            missing_parts.append("adc")
        if hbv is None:
            missing_parts.append("hbv")
        if not os.path.isfile(mask):
            missing_parts.append("gland_mask")

        if missing_parts:
            print(f"[WARN] Incomplete files for case {case_key}: missing {', '.join(missing_parts)}")
            continue

        items.append({
            "case_id": case_id,
            "study_id": row.study_id,
            "case_key": case_key,
            "label": label,
            "t2w": t2w,
            "adc": adc,
            "hbv": hbv,
            "mask": mask,
        })

    return items


# =========================
# 5) DATASET
# =========================
class CsPCaMriDataset(Dataset):
    def __init__(self, items, target_shape=(48, 96, 96), augment=False):
        self.items = items
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def random_flip(self, x, p=0.5):
        if random.random() < p:
            x = np.flip(x, axis=2).copy()  # H
        if random.random() < p:
            x = np.flip(x, axis=3).copy()  # W
        return x

    def random_intensity(self, x, p=0.3):
        if random.random() < p:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            x = x * scale + shift
        if random.random() < p:
            noise = np.random.normal(0, 0.03, size=x.shape).astype(np.float32)
            x = x + noise
        return x.astype(np.float32)

    def __getitem__(self, idx):
        item = self.items[idx]

        t2w = read_image(item["t2w"])
        adc = read_image(item["adc"])
        hbv = read_image(item["hbv"])
        mask = (read_image(item["mask"]) > 0).astype(np.uint8)

        common_shape = t2w.shape

        if adc.shape != common_shape:
            adc = resize_3d_torch(adc[None], common_shape, mode="trilinear")[0]
        if hbv.shape != common_shape:
            hbv = resize_3d_torch(hbv[None], common_shape, mode="trilinear")[0]
        if mask.shape != common_shape:
            mask = resize_3d_torch(mask[None].astype(np.float32), common_shape, mode="nearest")[0]
            mask = (mask > 0.5).astype(np.uint8)

        bbox = get_bbox(mask, margin=8)
        t2w = crop_to_bbox(t2w, bbox)
        adc = crop_to_bbox(adc, bbox)
        hbv = crop_to_bbox(hbv, bbox)
        mask = crop_to_bbox(mask, bbox)

        if min(t2w.shape) == 0 or min(adc.shape) == 0 or min(hbv.shape) == 0 or min(mask.shape) == 0:
            raise RuntimeError(
                f"Empty crop for case {item['case_key']} | "
                f"t2w={t2w.shape}, adc={adc.shape}, hbv={hbv.shape}, mask={mask.shape}, bbox={bbox}"
            )

        t2w = resize_3d_torch(t2w[None], self.target_shape, mode="trilinear")[0]
        adc = resize_3d_torch(adc[None], self.target_shape, mode="trilinear")[0]
        hbv = resize_3d_torch(hbv[None], self.target_shape, mode="trilinear")[0]

        t2w = zscore_nonzero(t2w)
        adc = zscore_nonzero(adc)
        hbv = zscore_nonzero(hbv)

        image = np.stack([t2w, adc, hbv], axis=0)

        if self.augment:
            image = self.random_flip(image, p=0.5)
            image = self.random_intensity(image, p=0.3)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "case_id": item["case_id"],
            "study_id": item["study_id"],
            "case_key": item["case_key"],
        }


# =========================
# 6) MODEL
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MRIEncoder3DThreeStemSharedTrunk(nn.Module):
    def __init__(self, embed_dim=64, dropout=0.3):
        super().__init__()

        self.stem_t2w = ConvBlock(1, 8)
        self.stem_adc = ConvBlock(1, 8)
        self.stem_hbv = ConvBlock(1, 8)

        self.pool1 = nn.MaxPool3d(kernel_size=POOL1_KERNEL, stride=POOL1_KERNEL)
        self.enc2 = ConvBlock(24, 16)
        self.pool2 = nn.MaxPool3d(kernel_size=POOL2_KERNEL, stride=POOL2_KERNEL)
        self.enc3 = ConvBlock(16, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=POOL3_KERNEL, stride=POOL3_KERNEL)
        self.enc4 = ConvBlock(32, 64)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 2, embed_dim)

    def forward(self, x):
        x_t2w = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]
        x_hbv = x[:, 2:3, ...]

        f_t2w = self.pool1(self.stem_t2w(x_t2w))
        f_adc = self.pool1(self.stem_adc(x_adc))
        f_hbv = self.pool1(self.stem_hbv(x_hbv))

        x = torch.cat([f_t2w, f_adc, f_hbv], dim=1)
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.enc4(x)

        x_avg = self.gap(x).flatten(1)
        x_max = self.gmp(x).flatten(1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.dropout(x)
        z = self.fc(x)
        return z


class CsPCaClassifierThreeStemSharedTrunk(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = MRIEncoder3DThreeStemSharedTrunk(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z


# =========================
# 7) TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds = (prob >= 0.44).long()

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    return total_loss / len(loader.dataset), acc



def eval_one_epoch(model, loader, criterion, return_predictions=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob, case_ids, study_ids, case_keys = [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            ids = batch["case_id"]
            sids = batch["study_id"]
            keys = batch["case_key"]

            logits, _ = model(x)
            loss = criterion(logits, y)

            prob = torch.softmax(logits, dim=1)[:, 1]
            preds = (prob >= 0.44).long()

            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
            case_ids.extend(ids)
            study_ids.extend(sids)
            case_keys.extend(keys)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    out = {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "f1": f1,
        "auc": auc,
    }

    if return_predictions:
        out["predictions_df"] = pd.DataFrame({
            "case_id": case_ids,
            "study_id": study_ids,
            "case_key": case_keys,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })

    return out


# =========================
# 8) EMBEDDING EXTRACTION
# =========================
def extract_embeddings(model, loader, out_dir, fold_name="all"):
    model.eval()
    all_z, all_ids, all_studies, all_keys, all_labels = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            ids = batch["case_id"]
            sids = batch["study_id"]
            keys = batch["case_key"]

            _, z = model(x)
            z = F.normalize(z, p=2, dim=1).cpu().numpy()

            all_z.append(z)
            all_ids.extend(ids)
            all_studies.extend(sids)
            all_keys.extend(keys)
            all_labels.extend(y.tolist())

    all_z = np.concatenate(all_z, axis=0)

    np.save(os.path.join(out_dir, f"{fold_name}_mri_csPCa_embeddings.npy"), all_z)
    pd.DataFrame({
        "case_id": all_ids,
        "study_id": all_studies,
        "case_key": all_keys,
        "label": all_labels,
    }).to_csv(os.path.join(out_dir, f"{fold_name}_mri_csPCa_embedding_ids.csv"), index=False)

    with open(os.path.join(out_dir, f"{fold_name}_mri_csPCa_embedding_meta.json"), "w") as f:
        json.dump({
            "num_cases": int(len(all_ids)),
            "embedding_dim": int(all_z.shape[1]),
            "target_shape": list(TARGET_SHAPE),
            "channels": ["t2w", "adc", "hbv"],
            "mask_source": GLAND_MASK_DIR,
            "pooling": {
                "pool1": list(POOL1_KERNEL),
                "pool2": list(POOL2_KERNEL),
                "pool3": list(POOL3_KERNEL),
            },
            "final_pooling": "global_avg_plus_global_max",
        }, f, indent=2)


# =========================
# 9) SHAPE DEBUGGING
# =========================
def print_feature_map_shapes(model, input_shape=(1, 3, 48, 96, 96)):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(input_shape, device=device, dtype=torch.float32)
    print(f"Input:    {tuple(x.shape)} | device={x.device}")

    with torch.no_grad():
        x_t2w = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]
        x_hbv = x[:, 2:3, ...]

        f_t2w = model.encoder.stem_t2w(x_t2w)
        f_adc = model.encoder.stem_adc(x_adc)
        f_hbv = model.encoder.stem_hbv(x_hbv)

        print(f"stem_t2w: {tuple(f_t2w.shape)}")
        print(f"stem_adc: {tuple(f_adc.shape)}")
        print(f"stem_hbv: {tuple(f_hbv.shape)}")

        f_t2w = model.encoder.pool1(f_t2w)
        f_adc = model.encoder.pool1(f_adc)
        f_hbv = model.encoder.pool1(f_hbv)

        print(f"pool1_t2w: {tuple(f_t2w.shape)}")
        print(f"pool1_adc: {tuple(f_adc.shape)}")
        print(f"pool1_hbv: {tuple(f_hbv.shape)}")

        x = torch.cat([f_t2w, f_adc, f_hbv], dim=1)
        print(f"concat:   {tuple(x.shape)}")

        x = model.encoder.enc2(x)
        print(f"enc2:     {tuple(x.shape)}")
        x = model.encoder.pool2(x)
        print(f"pool2:    {tuple(x.shape)}")

        x = model.encoder.enc3(x)
        print(f"enc3:     {tuple(x.shape)}")
        x = model.encoder.pool3(x)
        print(f"pool3:    {tuple(x.shape)}")

        x = model.encoder.enc4(x)
        print(f"enc4:     {tuple(x.shape)}")

        x_avg = model.encoder.gap(x).flatten(1)
        x_max = model.encoder.gmp(x).flatten(1)
        print(f"gap:      {tuple(x_avg.shape)}")
        print(f"gmp:      {tuple(x_max.shape)}")

        x_cat = torch.cat([x_avg, x_max], dim=1)
        print(f"concat pooled: {tuple(x_cat.shape)}")


# =========================
# 10) MAIN CV LOOP
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count visible: {torch.cuda.device_count()}")

    print(f"Output dir: {OUT_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")
    print(f"Gland mask dir: {GLAND_MASK_DIR}")
    print("Reduced pooling configuration:")
    print(f"  pool1 = {POOL1_KERNEL}")
    print(f"  pool2 = {POOL2_KERNEL}")
    print(f"  pool3 = {POOL3_KERNEL}")
    print("Final pooling = global average pooling + global max pooling")
    print(f"NUM_WORKERS = {NUM_WORKERS}")

    labels_df = load_case_dataframe(LABELS_CSV)
    print(f"Loaded rows from CSV: {len(labels_df)}")
    print("Raw class distribution from CSV:")
    print(labels_df['label'].value_counts().sort_index())

    items = find_case_files(ROOT_DIR, GLAND_MASK_DIR, labels_df)

    print(f"Loaded usable cases: {len(items)}")
    if len(items) < 10:
        raise RuntimeError("Too few valid cases found. Check image paths, mask paths, and CSV rows.")

    labels = np.array([x["label"] for x in items])
    print("Usable class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_fold_predictions = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(items)), labels), start=1):
        print(f"\n========== FOLD {fold}/{N_FOLDS} ==========")

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        train_ds = CsPCaMriDataset(train_items, target_shape=TARGET_SHAPE, augment=True)
        val_ds = CsPCaMriDataset(val_items, target_shape=TARGET_SHAPE, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        )

        model = CsPCaClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train_labels_fold = np.array([x["label"] for x in train_items])
        class_counts = np.bincount(train_labels_fold, minlength=2)
        class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        print(f"Fold {fold} class counts: {class_counts.tolist()}")
        print(f"Fold {fold} class weights: {class_weights.detach().cpu().numpy().tolist()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        if fold == 1:
            print("\nFeature map shapes with reduced pooling + avg/max global pooling:")
            print_feature_map_shapes(model, input_shape=(1, 3, *TARGET_SHAPE))

        run_name = f"fold_{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=os.path.join(TB_DIR, run_name))

        best_state = None
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_metrics = eval_one_epoch(model, val_loader, criterion, return_predictions=False)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
            writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            if not np.isnan(val_metrics["auc"]):
                writer.add_scalar("AUC/val", val_metrics["auc"], epoch)

            print(
                f"Fold {fold:02d} | Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping on fold {fold}")
                    break

        if best_state is None:
            writer.close()
            raise RuntimeError(f"No model state saved for fold {fold}")

        fold_path = os.path.join(OUT_DIR, f"best_csPCa_mri_fold{fold}.pt")
        torch.save(best_state, fold_path)

        model.load_state_dict(best_state)

        best_val_metrics = eval_one_epoch(model, val_loader, criterion, return_predictions=True)
        val_pred_df = best_val_metrics.pop("predictions_df")
        val_pred_df.insert(0, "fold", fold)
        val_pred_df.to_csv(os.path.join(OUT_DIR, f"fold{fold}_val_predictions.csv"), index=False)
        all_fold_predictions.append(val_pred_df)

        extract_embeddings(model, val_loader, OUT_DIR, fold_name=f"fold{fold}")

        fold_results.append({
            "fold": fold,
            "val_loss": best_val_metrics["loss"],
            "val_acc": best_val_metrics["acc"],
            "val_f1": best_val_metrics["f1"],
            "val_auc": best_val_metrics["auc"],
            "n_train": len(train_items),
            "n_val": len(val_items),
            "num_workers": NUM_WORKERS,
            "final_pooling": "avg+max",
        })

        writer.close()

    df = pd.DataFrame(fold_results)
    df.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False)

    if all_fold_predictions:
        all_preds_df = pd.concat(all_fold_predictions, axis=0, ignore_index=True)
        all_preds_df.to_csv(os.path.join(OUT_DIR, "all_folds_val_predictions.csv"), index=False)

    summary = pd.DataFrame([
        {
            "metric": "mean",
            "val_loss": df["val_loss"].mean(),
            "val_acc": df["val_acc"].mean(),
            "val_f1": df["val_f1"].mean(),
            "val_auc": df["val_auc"].mean(),
        },
        {
            "metric": "std",
            "val_loss": df["val_loss"].std(),
            "val_acc": df["val_acc"].std(),
            "val_f1": df["val_f1"].std(),
            "val_auc": df["val_auc"].std(),
        },
    ])
    summary.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"), index=False)

    print("\n========== CV SUMMARY ==========")
    print(df)
    print("\nSummary:")
    print(summary)
    print(f"\nTensorBoard logs: {TB_DIR}")


if __name__ == "__main__":
    main()
