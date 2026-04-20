
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# =========================
# 1) CONFIG
# =========================
ROOT_DIR = "images"

# Choose one:
#   "predefined_train_test" -> original CHIMERA train/test split + CV only on train
#   "hardcoded_folds"       -> use the exact fold lists provided by Khlalil
SPLIT_MODE = "hardcoded_folds"

OUT_DIR = f"outputs_v6_final_{SPLIT_MODE}_128D"
TB_DIR = os.path.join(OUT_DIR, "tensorboard")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SHAPE = (48, 96, 96)
EMBED_DIM = 128
SEED = 42
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 8
PRED_THRESHOLD = 0.44

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
# 2) SPLITS
# =========================
# --- original version: CHIMERA only, fixed train/test split ---
TRAIN_BCR_POS = ['1003', '1010', '1030', '1041', '1064',
                 '1086', '1100', '1106', '1127', '1135',
                 '1137', '1165', '1169', '1185', '1195',
                 '1208', '1214', '1217', '1219', '1240',
                 '1258', '1287']
TRAIN_BCR_NEG = ['1021', '1026', '1028', '1031', '1035',
                 '1048', '1052', '1056', '1060', '1062',
                 '1066', '1071', '1094', '1112', '1114',
                 '1117', '1123', '1129', '1136', '1140',
                 '1141', '1149', '1174', '1179', '1186',
                 '1188', '1192', '1205', '1206', '1207',
                 '1211', '1212', '1216', '1223', '1227',
                 '1260', '1261', '1262', '1264', '1269',
                 '1279', '1282', '1284', '1285', '1290',
                 '1291', '1293', '1294', '1296', '1298',
                 '1299', '1301', '1303', '1304']
TEST_BCR_POS = ['1090', '1122', '1130', '1193', '1295']
TEST_BCR_NEG = ['1011', '1025', '1036', '1037', '1039', '1068', '1107', '1120', '1151', '1184', '1235', '1252', '1267', '1286']

# --- second version: exact hardcoded folds ---
FOLD_VAL_POS = [
    ['1064', '1137', '1165', '1169', '1185', '1214'],
    ['1003', '1030', '1086', '1090', '1122', '1295'],
    ['1010', '1100', '1127', '1130', '1195'],
    ['1041', '1106', '1193', '1240', '1287'],
    ['1135', '1208', '1217', '1219', '1258'],
]
FOLD_VAL_NEG = [
    ['1025', '1035', '1060', '1114', '1117', '1188', '1211', '1216', '1264', '1284', '1294', '1299', '1303'],
    ['1026', '1066', '1123', '1151', '1184', '1212', '1260', '1282', '1285', '1291', '1293', '1296', '1301'],
    ['1011', '1031', '1048', '1052', '1071', '1129', '1174', '1179', '1186', '1206', '1207', '1262', '1279', '1298'],
    ['1021', '1036', '1039', '1062', '1068', '1107', '1136', '1223', '1227', '1252', '1261', '1269', '1286', '1290'],
    ['1028', '1037', '1056', '1094', '1112', '1120', '1140', '1141', '1149', '1192', '1205', '1235', '1267', '1304'],
]
N_FOLDS_HARDCODED = len(FOLD_VAL_POS)

ALL_BCR_POS = sorted({cid for fold in FOLD_VAL_POS for cid in fold}, key=int)
ALL_BCR_NEG = sorted({cid for fold in FOLD_VAL_NEG for cid in fold}, key=int)
ALL_IDS_HARDCODED = ALL_BCR_POS + ALL_BCR_NEG


def get_case_labels(split_mode):
    if split_mode == "predefined_train_test":
        case_labels = {cid: 1 for cid in TRAIN_BCR_POS + TEST_BCR_POS}
        case_labels.update({cid: 0 for cid in TRAIN_BCR_NEG + TEST_BCR_NEG})
        return case_labels
    if split_mode == "hardcoded_folds":
        case_labels = {cid: 1 for cid in ALL_BCR_POS}
        case_labels.update({cid: 0 for cid in ALL_BCR_NEG})
        return case_labels
    raise ValueError(f"Unknown SPLIT_MODE: {split_mode}")


CASE_LABELS = get_case_labels(SPLIT_MODE)


# =========================
# 3) REPRODUCIBILITY
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
# 4) IMAGE HELPERS
# =========================
def read_mha(path):
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
    x = torch.from_numpy(volume).unsqueeze(0).float()  # (1, C, D, H, W)
    if mode in ["trilinear", "bilinear"]:
        x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=target_shape, mode=mode)
    return x.squeeze(0).cpu().numpy()


# =========================
# 5) FIND CASE FILES
# =========================
def find_case_files(root_dir, allowed_case_ids, case_labels):
    items = []

    for case_id in sorted(set(allowed_case_ids), key=int):
        case_dir = os.path.join(root_dir, case_id)
        if not os.path.isdir(case_dir):
            print(f"[WARN] Missing folder for case {case_id}")
            continue

        t2w = glob.glob(os.path.join(case_dir, "*_t2w.mha"))
        adc = glob.glob(os.path.join(case_dir, "*_adc.mha"))
        hbv = glob.glob(os.path.join(case_dir, "*_hbv.mha"))
        mask = glob.glob(os.path.join(case_dir, "*_mask.mha"))

        if len(t2w) != 1 or len(adc) != 1 or len(hbv) != 1 or len(mask) != 1:
            print(f"[WARN] Incomplete files for case {case_id}")
            continue

        if case_id not in case_labels:
            print(f"[WARN] Missing label for case {case_id}")
            continue

        items.append({
            "case_id": case_id,
            "label": int(case_labels[case_id]),
            "t2w": t2w[0],
            "adc": adc[0],
            "hbv": hbv[0],
            "mask": mask[0],
        })

    return items


def build_case_dict(items):
    return {item["case_id"]: item for item in items}


# =========================
# 6) DATASET
# =========================
class BCRMriDataset(Dataset):
    def __init__(self, items, target_shape=(48, 96, 96), augment=False):
        self.items = items
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def random_flip(self, x, p=0.5):
        if random.random() < p:
            x = np.flip(x, axis=2).copy()
        if random.random() < p:
            x = np.flip(x, axis=3).copy()
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

        t2w = read_mha(item["t2w"])
        adc = read_mha(item["adc"])
        hbv = read_mha(item["hbv"])
        mask = (read_mha(item["mask"]) > 0).astype(np.uint8)

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
                f"Empty crop for case {item['case_id']} | "
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
        }


def make_loader(items, augment, shuffle, batch_size=BATCH_SIZE):
    ds = BCRMriDataset(items, target_shape=TARGET_SHAPE, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )


# =========================
# 7) MODEL
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


class BCRClassifierThreeStemSharedTrunk(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = MRIEncoder3DThreeStemSharedTrunk(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z


# =========================
# 8) METRICS / TRAINING
# =========================
def probabilities_to_predictions(y_prob, threshold=PRED_THRESHOLD):
    return (np.asarray(y_prob) >= threshold).astype(int)


def compute_metrics(y_true, y_prob, threshold=PRED_THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = probabilities_to_predictions(y_prob, threshold=threshold)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "y_pred": y_pred,
    }


def evaluate_model(model, loader, criterion, split_name, fold=None, return_predictions=False):
    model.eval()
    total_loss = 0.0
    y_true, y_prob, case_ids, all_embeddings = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            ids = batch["case_id"]

            logits, z = model(x)
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
            case_ids.extend(ids)
            all_embeddings.append(F.normalize(z, p=2, dim=1).cpu().numpy())

    metrics = compute_metrics(y_true, y_prob, threshold=PRED_THRESHOLD)
    out = {
        "split": split_name,
        "fold": fold,
        "loss": total_loss / len(loader.dataset),
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "threshold": metrics["threshold"],
    }

    if return_predictions:
        out["predictions_df"] = pd.DataFrame({
            "fold": fold if fold is not None else "final",
            "split": split_name,
            "case_id": case_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": metrics["y_pred"].tolist(),
        })

    if all_embeddings:
        out["embeddings"] = np.concatenate(all_embeddings, axis=0)
        out["case_ids"] = case_ids
        out["labels"] = np.asarray(y_true).astype(int)

    return out


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    y_true, y_prob = [], []

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
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_prob, threshold=PRED_THRESHOLD)
    return total_loss / len(loader.dataset), metrics


def get_class_weights(items):
    labels = np.array([x["label"] for x in items], dtype=int)
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE), counts


def train_model(train_items, val_items, run_dir, fold_label):
    os.makedirs(run_dir, exist_ok=True)

    train_loader = make_loader(train_items, augment=True, shuffle=True)
    val_loader = make_loader(val_items, augment=False, shuffle=False)

    model = BCRClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    class_weights, class_counts = get_class_weights(train_items)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"[{fold_label}] class counts: {class_counts.tolist()}")
    print(f"[{fold_label}] class weights: {class_weights.detach().cpu().numpy().tolist()}")

    writer = SummaryWriter(log_dir=os.path.join(TB_DIR, f"{fold_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    best_state = None
    best_val_loss = float("inf")
    patience = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate_model(model, val_loader, criterion, split_name="val", fold=fold_label)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["acc"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
        writer.add_scalar("F1/train", train_metrics["f1"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        if not np.isnan(train_metrics["auc"]):
            writer.add_scalar("AUC/train", train_metrics["auc"], epoch)
        if not np.isnan(val_metrics["auc"]):
            writer.add_scalar("AUC/val", val_metrics["auc"], epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history_rows.append(row)

        print(
            f"{fold_label} | Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} train_auc={train_metrics['auc']:.4f} | "
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
                print(f"Early stopping on {fold_label}")
                break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    writer.close()

    if best_state is None:
        raise RuntimeError(f"No best model saved for {fold_label}")

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    final_train_metrics = evaluate_model(model, train_loader, criterion, split_name="train", fold=fold_label, return_predictions=True)
    final_val_metrics = evaluate_model(model, val_loader, criterion, split_name="val", fold=fold_label, return_predictions=True)

    final_train_metrics["predictions_df"].to_csv(os.path.join(run_dir, "train_predictions.csv"), index=False)
    final_val_metrics["predictions_df"].to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)

    return model, criterion, final_train_metrics, final_val_metrics, history_df


# =========================
# 9) EMBEDDING EXTRACTION
# =========================
def save_embedding_array(embeddings, case_ids, labels, out_prefix):
    np.save(f"{out_prefix}_embeddings.npy", embeddings)
    pd.DataFrame({
        "case_id": case_ids,
        "label": labels,
    }).to_csv(f"{out_prefix}_embedding_ids.csv", index=False)

    with open(f"{out_prefix}_embedding_meta.json", "w") as f:
        json.dump({
            "num_cases": int(len(case_ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "target_shape": list(TARGET_SHAPE),
            "channels": ["t2w", "adc", "hbv"],
            "pooling": {
                "pool1": list(POOL1_KERNEL),
                "pool2": list(POOL2_KERNEL),
                "pool3": list(POOL3_KERNEL),
            },
            "final_pooling": "global_avg_plus_global_max",
            "split_mode": SPLIT_MODE,
            "threshold": PRED_THRESHOLD,
        }, f, indent=2)


def extract_embeddings_for_items(model, items, out_prefix):
    loader = make_loader(items, augment=False, shuffle=False)
    dummy_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
    dummy_criterion = nn.CrossEntropyLoss(weight=dummy_weights)
    eval_out = evaluate_model(model, loader, dummy_criterion, split_name="embeddings_only", fold="final")
    save_embedding_array(eval_out["embeddings"], eval_out["case_ids"], eval_out["labels"], out_prefix)
    return eval_out


# =========================
# 10) SHAPE DEBUGGING
# =========================
def print_feature_map_shapes(model, input_shape=(1, 3, 48, 96, 96)):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(input_shape, device=device, dtype=torch.float32)
    print(f"Input: {tuple(x.shape)} | device={x.device}")

    with torch.no_grad():
        x_t2w = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]
        x_hbv = x[:, 2:3, ...]

        f_t2w = model.encoder.pool1(model.encoder.stem_t2w(x_t2w))
        f_adc = model.encoder.pool1(model.encoder.stem_adc(x_adc))
        f_hbv = model.encoder.pool1(model.encoder.stem_hbv(x_hbv))
        print(f"stem_t2w+pool1: {tuple(f_t2w.shape)}")
        print(f"stem_adc+pool1: {tuple(f_adc.shape)}")
        print(f"stem_hbv+pool1: {tuple(f_hbv.shape)}")

        x_cat = torch.cat([f_t2w, f_adc, f_hbv], dim=1)
        print(f"concat: {tuple(x_cat.shape)}")

        x2 = model.encoder.pool2(model.encoder.enc2(x_cat))
        print(f"enc2+pool2: {tuple(x2.shape)}")

        x3 = model.encoder.pool3(model.encoder.enc3(x2))
        print(f"enc3+pool3: {tuple(x3.shape)}")

        x4 = model.encoder.enc4(x3)
        print(f"enc4: {tuple(x4.shape)}")

        x_avg = model.encoder.gap(x4).flatten(1)
        x_max = model.encoder.gmp(x4).flatten(1)
        print(f"gap: {tuple(x_avg.shape)}")
        print(f"gmp: {tuple(x_max.shape)}")
        print(f"concat pooled: {tuple(torch.cat([x_avg, x_max], dim=1).shape)}")


# =========================
# 11) SPLIT BUILDERS
# =========================
def build_splits_predefined(case_dict):
    train_ids = [cid for cid in TRAIN_BCR_POS + TRAIN_BCR_NEG if cid in case_dict]
    test_ids = [cid for cid in TEST_BCR_POS + TEST_BCR_NEG if cid in case_dict]

    train_items = [case_dict[cid] for cid in train_ids]
    test_items = [case_dict[cid] for cid in test_ids]

    train_labels = np.array([x["label"] for x in train_items], dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_items)), train_labels), start=1):
        fold_train_items = [train_items[i] for i in train_idx]
        fold_val_items = [train_items[i] for i in val_idx]
        cv_splits.append((fold, fold_train_items, fold_val_items))

    return train_items, test_items, cv_splits


def build_splits_hardcoded(case_dict):
    case_ids = [cid for cid in ALL_IDS_HARDCODED if cid in case_dict]
    missing_ids = sorted(set(ALL_IDS_HARDCODED) - set(case_ids), key=int)
    if missing_ids:
        print(f"[WARN] Missing cases excluded from hardcoded folds: {missing_ids}")

    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    all_idx = np.arange(len(case_ids))
    splits = []

    for f in range(N_FOLDS_HARDCODED):
        val_ids_f = [cid for cid in (FOLD_VAL_POS[f] + FOLD_VAL_NEG[f]) if cid in case_to_idx]
        val_set = set(val_ids_f)
        val_idx_f = np.array([case_to_idx[cid] for cid in val_ids_f], dtype=int)
        train_idx_f = np.array([i for i in all_idx if case_ids[i] not in val_set], dtype=int)

        train_ids = [case_ids[i] for i in train_idx_f]
        val_ids = [case_ids[i] for i in val_idx_f]

        print(f"\n========== TRAIN/TEST SPLIT (fold {f+1}/{N_FOLDS_HARDCODED}) ==========")
        print(f"  Train: {train_ids}")
        print(f"  Val:   {val_ids}")

        train_items = [case_dict[cid] for cid in train_ids]
        val_items = [case_dict[cid] for cid in val_ids]
        splits.append((f + 1, train_items, val_items))

    all_items = [case_dict[cid] for cid in case_ids]
    return all_items, splits


# =========================
# 12) REPORT HELPERS
# =========================
def metrics_to_row(metrics, split_name, fold):
    return {
        "fold": fold,
        "split": split_name,
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "threshold": metrics["threshold"],
        "n_cases": len(metrics["labels"]) if "labels" in metrics else np.nan,
    }


def save_summary_tables(results_df, out_dir, prefix):
    results_df.to_csv(os.path.join(out_dir, f"{prefix}_metrics.csv"), index=False)

    summary_rows = []
    for split_name in sorted(results_df["split"].unique()):
        sub = results_df[results_df["split"] == split_name]
        mean_row = {"metric": "mean", "split": split_name}
        std_row = {"metric": "std", "split": split_name}
        for col in ["loss", "acc", "f1", "auc", "sensitivity", "specificity"]:
            mean_row[col] = sub[col].mean()
            std_row[col] = sub[col].std()
        summary_rows.extend([mean_row, std_row])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)
    return summary_df


# =========================
# 13) MAIN
# =========================
def run_predefined_train_test(case_dict):
    mode_dir = os.path.join(OUT_DIR, "predefined_train_test")
    os.makedirs(mode_dir, exist_ok=True)

    train_items, test_items, cv_splits = build_splits_predefined(case_dict)

    print(f"Predefined train cases: {len(train_items)}")
    print(f"Predefined test cases:  {len(test_items)}")

    all_cv_rows = []
    all_cv_pred_dfs = []

    debug_model = BCRClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
    print("\nFeature map shapes:")
    print_feature_map_shapes(debug_model, input_shape=(1, 3, *TARGET_SHAPE))

    for fold, fold_train_items, fold_val_items in cv_splits:
        fold_dir = os.path.join(mode_dir, f"cv_fold_{fold}")
        model, criterion, train_metrics, val_metrics, _ = train_model(
            train_items=fold_train_items,
            val_items=fold_val_items,
            run_dir=fold_dir,
            fold_label=f"cv_fold_{fold}",
        )

        all_cv_rows.append(metrics_to_row(train_metrics, "train", fold))
        all_cv_rows.append(metrics_to_row(val_metrics, "val", fold))
        all_cv_pred_dfs.append(train_metrics["predictions_df"])
        all_cv_pred_dfs.append(val_metrics["predictions_df"])

        save_embedding_array(
            val_metrics["embeddings"],
            val_metrics["case_ids"],
            val_metrics["labels"],
            os.path.join(fold_dir, "val")
        )

    cv_results_df = pd.DataFrame(all_cv_rows)
    cv_summary_df = save_summary_tables(cv_results_df, mode_dir, prefix="cv")

    if all_cv_pred_dfs:
        pd.concat(all_cv_pred_dfs, axis=0, ignore_index=True).to_csv(
            os.path.join(mode_dir, "cv_all_predictions.csv"), index=False
        )

    # final model: full TRAIN -> evaluate TRAIN and TEST
    final_dir = os.path.join(mode_dir, "final_train_on_full_train")
    # Use train itself as the monitored split for early stopping fallback-free final fit
    # Here we split internally again for model selection using the first CV fold's val set.
    _, internal_val_items = cv_splits[0][1], cv_splits[0][2]
    internal_train_case_ids = {x["case_id"] for x in internal_val_items}
    internal_train_items = [x for x in train_items if x["case_id"] not in internal_train_case_ids]

    final_model, final_criterion, final_train_metrics_internal, final_val_metrics_internal, _ = train_model(
        train_items=internal_train_items,
        val_items=internal_val_items,
        run_dir=final_dir,
        fold_label="final_model",
    )

    # Re-evaluate final selected model on full TRAIN and held-out TEST
    full_train_loader = make_loader(train_items, augment=False, shuffle=False)
    test_loader = make_loader(test_items, augment=False, shuffle=False)

    full_train_metrics = evaluate_model(
        final_model, full_train_loader, final_criterion, split_name="train_full", fold="final", return_predictions=True
    )
    test_metrics = evaluate_model(
        final_model, test_loader, final_criterion, split_name="test", fold="final", return_predictions=True
    )

    final_results_df = pd.DataFrame([
        metrics_to_row(final_train_metrics_internal, "train_internal", "final"),
        metrics_to_row(final_val_metrics_internal, "val_internal", "final"),
        metrics_to_row(full_train_metrics, "train_full", "final"),
        metrics_to_row(test_metrics, "test", "final"),
    ])
    final_results_df.to_csv(os.path.join(final_dir, "final_metrics.csv"), index=False)

    full_train_metrics["predictions_df"].to_csv(os.path.join(final_dir, "train_full_predictions.csv"), index=False)
    test_metrics["predictions_df"].to_csv(os.path.join(final_dir, "test_predictions.csv"), index=False)

    # embeddings for all patients in predefined split
    all_items = train_items + test_items
    all_emb_out = extract_embeddings_for_items(final_model, all_items, os.path.join(final_dir, "all_patients"))
    train_emb_out = extract_embeddings_for_items(final_model, train_items, os.path.join(final_dir, "train_patients"))
    test_emb_out = extract_embeddings_for_items(final_model, test_items, os.path.join(final_dir, "test_patients"))

    print("\n========== PREDEFINED TRAIN/TEST SUMMARY ==========")
    print(cv_results_df)
    print("\nCV summary:")
    print(cv_summary_df)
    print("\nFinal model metrics:")
    print(final_results_df)

    return {
        "cv_results_df": cv_results_df,
        "cv_summary_df": cv_summary_df,
        "final_results_df": final_results_df,
        "all_embeddings_shape": all_emb_out["embeddings"].shape,
        "train_embeddings_shape": train_emb_out["embeddings"].shape,
        "test_embeddings_shape": test_emb_out["embeddings"].shape,
    }


def run_hardcoded_folds(case_dict):
    mode_dir = os.path.join(OUT_DIR, "hardcoded_folds")
    os.makedirs(mode_dir, exist_ok=True)

    all_items, splits = build_splits_hardcoded(case_dict)

    print(f"Hardcoded-fold usable cases: {len(all_items)}")

    debug_model = BCRClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
    print("\nFeature map shapes:")
    print_feature_map_shapes(debug_model, input_shape=(1, 3, *TARGET_SHAPE))

    all_rows = []
    all_pred_dfs = []
    oof_embedding_chunks = []
    oof_id_chunks = []
    oof_label_chunks = []

    for fold, train_items, val_items in splits:
        fold_dir = os.path.join(mode_dir, f"fold_{fold}")
        model, criterion, train_metrics, val_metrics, _ = train_model(
            train_items=train_items,
            val_items=val_items,
            run_dir=fold_dir,
            fold_label=f"hardcoded_fold_{fold}",
        )

        all_rows.append(metrics_to_row(train_metrics, "train", fold))
        all_rows.append(metrics_to_row(val_metrics, "val", fold))

        all_pred_dfs.append(train_metrics["predictions_df"])
        all_pred_dfs.append(val_metrics["predictions_df"])

        save_embedding_array(
            val_metrics["embeddings"],
            val_metrics["case_ids"],
            val_metrics["labels"],
            os.path.join(fold_dir, "val")
        )

        oof_embedding_chunks.append(val_metrics["embeddings"])
        oof_id_chunks.extend(val_metrics["case_ids"])
        oof_label_chunks.extend(val_metrics["labels"].tolist())

    results_df = pd.DataFrame(all_rows)
    summary_df = save_summary_tables(results_df, mode_dir, prefix="hardcoded_cv")

    if all_pred_dfs:
        pd.concat(all_pred_dfs, axis=0, ignore_index=True).to_csv(
            os.path.join(mode_dir, "hardcoded_all_predictions.csv"), index=False
        )

    if oof_embedding_chunks:
        save_embedding_array(
            np.concatenate(oof_embedding_chunks, axis=0),
            oof_id_chunks,
            oof_label_chunks,
            os.path.join(mode_dir, "all_patients_oof")
        )

    print("\n========== HARDCODED-FOLDS SUMMARY ==========")
    print(results_df)
    print("\nSummary:")
    print(summary_df)

    return {
        "results_df": results_df,
        "summary_df": summary_df,
        "oof_embeddings_shape": np.concatenate(oof_embedding_chunks, axis=0).shape if oof_embedding_chunks else None,
    }


def main():
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count visible: {torch.cuda.device_count()}")

    print(f"Output dir: {OUT_DIR}")
    print(f"Split mode: {SPLIT_MODE}")
    print(f"Prediction threshold: {PRED_THRESHOLD}")
    print("Reduced pooling configuration:")
    print(f"  pool1 = {POOL1_KERNEL}")
    print(f"  pool2 = {POOL2_KERNEL}")
    print(f"  pool3 = {POOL3_KERNEL}")
    print("Final pooling = global average pooling + global max pooling")
    print(f"NUM_WORKERS = {NUM_WORKERS}")

    if SPLIT_MODE == "predefined_train_test":
        allowed_ids = TRAIN_BCR_POS + TRAIN_BCR_NEG + TEST_BCR_POS + TEST_BCR_NEG
    elif SPLIT_MODE == "hardcoded_folds":
        allowed_ids = ALL_IDS_HARDCODED
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    items = find_case_files(ROOT_DIR, allowed_ids, CASE_LABELS)
    print(f"Loaded usable cases: {len(items)}")
    if len(items) < 10:
        raise RuntimeError("Too few valid cases found. Check file paths.")

    case_dict = build_case_dict(items)

    if SPLIT_MODE == "predefined_train_test":
        run_predefined_train_test(case_dict)
    elif SPLIT_MODE == "hardcoded_folds":
        run_hardcoded_folds(case_dict)
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    print(f"\nTensorBoard logs: {TB_DIR}")


if __name__ == "__main__":
    main()

