import os
import glob
import json
import copy
import random
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# =========================
# 1) CONFIG
# =========================
# ----- Phase 1: final csPCa training on PI-CAI -----
PICAI_ROOT_DIR = "images"
PICAI_LABELS_CSV = "marksheet-2.csv"
PICAI_GLAND_MASK_DIR = "picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23"
TRAIN_OUT_DIR = "outputs_csPCa_mri_final_model_and_embeddings_CHIMERA_all_labels"
TRAIN_TB_DIR = os.path.join(TRAIN_OUT_DIR, "tensorboard")

# ----- Phase 2: embedding extraction on downstream cohort -----
# Edit these for your BCR / downstream cohort.
CHIMERA_ROOT_DIR = "/rs01/home/lauterw/project_MRI/CHIMERA/radiology/images"
CHIMERA_OUT_DIR = "outputs_chimera_csPCa_embeddings"

TRAIN_BCR_POS = ['1003', '1010', '1030', '1041', '1064', '1086', '1100', '1106', '1127', '1135', '1137', '1165', '1169', '1185', '1195', '1208', '1214', '1217', '1219', '1240', '1258', '1287']
TRAIN_BCR_NEG = ['1021', '1026', '1028', '1031', '1035', '1048', '1052', '1056', '1060', '1062', '1066', '1071', '1094', '1112', '1114', '1117', '1123', '1129', '1136', '1140', '1141', '1149', '1174', '1179', '1186', '1188', '1192', '1205', '1206', '1207', '1211', '1212', '1216', '1223', '1227', '1260', '1261', '1262', '1264', '1269', '1279', '1282', '1284', '1285', '1290', '1291', '1293', '1294', '1296', '1298', '1299', '1301', '1303', '1304']
TEST_BCR_POS = ['1090', '1122', '1130', '1193', '1295']
TEST_BCR_NEG = ['1011', '1025', '1036', '1037', '1039', '1068', '1107', '1120', '1151', '1184', '1235', '1252', '1267', '1286']

# ----- Run modes -----
RUN_TRAIN_FINAL = False
RUN_EXTRACT_CHIMERA = True

# ----- Training hyperparameters -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
FINAL_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SHAPE = (48, 96, 96)
EMBED_DIM = 64
DROPOUT = 0.3
SEED = 42

# ----- HPC -----
NUM_WORKERS = 5
PIN_MEMORY = torch.cuda.is_available()

# ----- Pooling configuration -----
POOL1_KERNEL = (1, 2, 2)
POOL2_KERNEL = (1, 2, 2)
POOL3_KERNEL = (2, 2, 2)

# ----- Extraction -----
L2_NORMALIZE_EMBEDDINGS = True

os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
os.makedirs(TRAIN_TB_DIR, exist_ok=True)
os.makedirs(CHIMERA_OUT_DIR, exist_ok=True)


# =========================
# 2) REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42):
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
def read_image(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr



def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    mask = x != 0
    if mask.sum() == 0:
        return x.astype(np.float32)
    vals = x[mask]
    mean = vals.mean()
    std = vals.std() + 1e-8
    y = x.copy()
    y[mask] = (vals - mean) / std
    return y.astype(np.float32)



def get_bbox(mask: np.ndarray, margin: int = 8):
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



def crop_to_bbox(vol: np.ndarray, bbox):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return vol[zmin:zmax, ymin:ymax, xmin:xmax]



def resize_3d_torch(volume: np.ndarray, target_shape, mode: str = "trilinear") -> np.ndarray:
    # volume expected shape: (C, D, H, W)
    x = torch.from_numpy(volume).unsqueeze(0).float()  # (1, C, D, H, W)
    if mode in ["trilinear", "bilinear"]:
        x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=target_shape, mode=mode)
    return x.squeeze(0).cpu().numpy()


# =========================
# 4) CASE DISCOVERY
# =========================
def load_picai_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"patient_id", "study_id", "case_csPCa"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df[["patient_id", "study_id", "case_csPCa"]].copy()
    df = df.rename(columns={"case_csPCa": "label"})

    # Keep integer formatting stable for folder/file matching.
    df["patient_id"] = df["patient_id"].astype(int).astype(str)
    df["study_id"] = df["study_id"].astype(int).astype(str)
    df["label"] = df["label"].astype(int)
    return df



def find_picai_training_cases(root_dir: str, labels_df: pd.DataFrame, gland_mask_dir: str) -> List[Dict]:
    items = []

    for _, row in labels_df.iterrows():
        patient_id = str(row["patient_id"])
        study_id = str(row["study_id"])
        label = int(row["label"])

        case_id = f"{patient_id}_{study_id}"
        patient_dir = os.path.join(root_dir, patient_id)

        if not os.path.isdir(patient_dir):
            print(f"[WARN] Missing patient folder for case {case_id}", flush=True)
            continue

        t2w = glob.glob(os.path.join(patient_dir, f"{case_id}_t2w.mha"))
        adc = glob.glob(os.path.join(patient_dir, f"{case_id}_adc.mha"))
        hbv = glob.glob(os.path.join(patient_dir, f"{case_id}_hbv.mha"))
        gland_mask = os.path.join(gland_mask_dir, f"{case_id}.nii.gz")

        missing = []
        if len(t2w) != 1:
            missing.append("t2w")
        if len(adc) != 1:
            missing.append("adc")
        if len(hbv) != 1:
            missing.append("hbv")
        if not os.path.isfile(gland_mask):
            missing.append("gland_mask")

        if missing:
            print(f"[WARN] Incomplete files for case {case_id}: missing {', '.join(missing)}", flush=True)
            continue

        items.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "study_id": study_id,
                "label": label,
                "t2w": t2w[0],
                "adc": adc[0],
                "hbv": hbv[0],
                "mask": gland_mask,
            }
        )

    return items



def _build_item_from_case_id(case_id: str, patient_dir: str, label: Optional[int], mask_path: Optional[str]) -> Optional[Dict]:
    t2w = glob.glob(os.path.join(patient_dir, f"{case_id}_t2w.mha"))
    adc = glob.glob(os.path.join(patient_dir, f"{case_id}_adc.mha"))
    hbv = glob.glob(os.path.join(patient_dir, f"{case_id}_hbv.mha"))

    missing = []
    if len(t2w) != 1:
        missing.append("t2w")
    if len(adc) != 1:
        missing.append("adc")
    if len(hbv) != 1:
        missing.append("hbv")
    if missing:
        print(f"[WARN] Skipping case {case_id}: missing {', '.join(missing)}", flush=True)
        return None

    item = {
        "case_id": case_id,
        "patient_id": case_id.split("_")[0],
        "study_id": case_id.split("_")[1],
        "label": -1 if label is None else int(label),
        "t2w": t2w[0],
        "adc": adc[0],
        "hbv": hbv[0],
        "mask": mask_path,
    }
    return item



def find_chimera_cases(root_dir: str, patient_ids: List[str], labels: Dict[str, int], split_name: str) -> List[Dict]:
    items: List[Dict] = []

    for patient_id in sorted(patient_ids):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"[WARN] Missing CHIMERA folder for patient {patient_id}", flush=True)
            continue

        case_id = f"{patient_id}_0001"
        t2w = glob.glob(os.path.join(patient_dir, f"{case_id}_t2w.mha"))
        adc = glob.glob(os.path.join(patient_dir, f"{case_id}_adc.mha"))
        hbv = glob.glob(os.path.join(patient_dir, f"{case_id}_hbv.mha"))
        mask = glob.glob(os.path.join(patient_dir, f"{case_id}_mask.mha"))

        missing = []
        if len(t2w) != 1:
            missing.append("t2w")
        if len(adc) != 1:
            missing.append("adc")
        if len(hbv) != 1:
            missing.append("hbv")
        if len(mask) != 1:
            missing.append("mask")

        if missing:
            print(f"[WARN] Skipping CHIMERA case {case_id}: missing {', '.join(missing)}", flush=True)
            continue

        items.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "study_id": "0001",
            "label": int(labels[patient_id]),
            "split": split_name,
            "t2w": t2w[0],
            "adc": adc[0],
            "hbv": hbv[0],
            "mask": mask[0],
        })

    return items


# =========================
# 5) DATASET
# =========================
class MultiModalMriDataset(Dataset):
    def __init__(self, items: List[Dict], target_shape=(48, 96, 96), augment: bool = False, require_label: bool = True):
        self.items = items
        self.target_shape = target_shape
        self.augment = augment
        self.require_label = require_label

    def __len__(self):
        return len(self.items)

    def random_flip(self, x: np.ndarray, p: float = 0.5) -> np.ndarray:
        if random.random() < p:
            x = np.flip(x, axis=2).copy()  # H
        if random.random() < p:
            x = np.flip(x, axis=3).copy()  # W
        return x

    def random_intensity(self, x: np.ndarray, p: float = 0.3) -> np.ndarray:
        if random.random() < p:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            x = x * scale + shift
        if random.random() < p:
            noise = np.random.normal(0, 0.03, size=x.shape).astype(np.float32)
            x = x + noise
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        t2w = read_image(item["t2w"])
        adc = read_image(item["adc"])
        hbv = read_image(item["hbv"])
        mask = read_image(item["mask"]) if item.get("mask") else None

        common_shape = t2w.shape

        if adc.shape != common_shape:
            adc = resize_3d_torch(adc[None], common_shape, mode="trilinear")[0]
        if hbv.shape != common_shape:
            hbv = resize_3d_torch(hbv[None], common_shape, mode="trilinear")[0]

        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != common_shape:
                mask = resize_3d_torch(mask[None].astype(np.float32), common_shape, mode="nearest")[0]
                mask = (mask > 0.5).astype(np.uint8)
            bbox = get_bbox(mask, margin=8)
            t2w = crop_to_bbox(t2w, bbox)
            adc = crop_to_bbox(adc, bbox)
            hbv = crop_to_bbox(hbv, bbox)

        if min(t2w.shape) == 0 or min(adc.shape) == 0 or min(hbv.shape) == 0:
            raise RuntimeError(
                f"Empty crop for case {item['case_id']} | t2w={t2w.shape}, adc={adc.shape}, hbv={hbv.shape}"
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

        out = {
            "image": torch.tensor(image, dtype=torch.float32),
            "case_id": item["case_id"],
        }
        if self.require_label:
            out["label"] = torch.tensor(item["label"], dtype=torch.long)
        return out


# =========================
# 6) MODEL
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, embed_dim: int = 64, dropout: float = 0.3):
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
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.encoder = MRIEncoder3DThreeStemSharedTrunk(embed_dim=embed_dim, dropout=DROPOUT)
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
    y_true, y_pred, y_prob = [], [], []

    for batch_idx, batch in enumerate(loader):
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

        if batch_idx == 0:
            print("First training batch loaded", flush=True)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    return total_loss / len(loader.dataset), acc, f1, auc


@torch.no_grad()
def evaluate_on_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    return total_loss / len(loader.dataset), acc, f1, auc


# =========================
# 8) EMBEDDING EXTRACTION
# =========================
@torch.no_grad()
def extract_embeddings(model, loader, out_dir: str, prefix: str):
    model.eval()
    all_z, all_ids, all_labels = [], [], []

    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        ids = batch["case_id"]
        labels = batch.get("label")
        _, z = model(x)
        if L2_NORMALIZE_EMBEDDINGS:
            z = F.normalize(z, p=2, dim=1)
        z = z.cpu().numpy()

        all_z.append(z)
        all_ids.extend(ids)
        if labels is not None:
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(all_z) == 0:
        raise RuntimeError("No embeddings extracted. Check downstream cohort paths.")

    all_z = np.concatenate(all_z, axis=0)
    np.save(os.path.join(out_dir, f"{prefix}_embeddings.npy"), all_z)
    out_df = pd.DataFrame({"case_id": all_ids})
    if len(all_labels) == len(all_ids):
        out_df["label"] = all_labels
    out_df.to_csv(os.path.join(out_dir, f"{prefix}_embedding_ids.csv"), index=False)

    with open(os.path.join(out_dir, f"{prefix}_embedding_meta.json"), "w") as f:
        json.dump(
            {
                "num_cases": int(len(all_ids)),
                "embedding_dim": int(all_z.shape[1]),
                "target_shape": list(TARGET_SHAPE),
                "channels": ["t2w", "adc", "hbv"],
                "pooling": {
                    "pool1": list(POOL1_KERNEL),
                    "pool2": list(POOL2_KERNEL),
                    "pool3": list(POOL3_KERNEL),
                },
                "final_pooling": "global_avg_plus_global_max",
                "l2_normalized": bool(L2_NORMALIZE_EMBEDDINGS),
                "source_model": os.path.join(TRAIN_OUT_DIR, "best_csPCa_mri_final.pt"),
            },
            f,
            indent=2,
        )

    print(f"Saved embeddings: {all_z.shape}", flush=True)


# =========================
# 9) MAIN PHASES
# =========================
def train_final_model():
    print(f"Using device: {DEVICE}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU count visible: {torch.cuda.device_count()}", flush=True)

    print(f"Output dir: {TRAIN_OUT_DIR}", flush=True)
    print(f"Labels CSV: {PICAI_LABELS_CSV}", flush=True)
    print(f"Gland mask dir: {PICAI_GLAND_MASK_DIR}", flush=True)
    print(f"NUM_WORKERS = {NUM_WORKERS}", flush=True)

    labels_df = load_picai_labels(PICAI_LABELS_CSV)
    print(f"Loaded rows from CSV: {len(labels_df)}", flush=True)
    print("Raw class distribution from CSV:", flush=True)
    print(labels_df["label"].value_counts().sort_index(), flush=True)

    items = find_picai_training_cases(PICAI_ROOT_DIR, labels_df, PICAI_GLAND_MASK_DIR)
    print(f"Loaded usable cases: {len(items)}", flush=True)
    if len(items) == 0:
        raise RuntimeError("No usable PI-CAI training cases found.")

    usable_labels = np.array([x["label"] for x in items])
    unique, counts = np.unique(usable_labels, return_counts=True)
    print("Usable class distribution:", dict(zip(unique.tolist(), counts.tolist())), flush=True)

    train_ds = MultiModalMriDataset(items, target_shape=TARGET_SHAPE, augment=True, require_label=True)
    eval_ds = MultiModalMriDataset(items, target_shape=TARGET_SHAPE, augment=False, require_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = CsPCaClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    class_counts = np.bincount(usable_labels, minlength=2)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Final-train class counts: {class_counts.tolist()}", flush=True)
    print(f"Final-train class weights: {class_weights.detach().cpu().numpy().tolist()}", flush=True)

    run_name = f"final_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(TRAIN_TB_DIR, run_name))

    best_state = None
    best_train_loss = float("inf")
    history = []

    for epoch in range(1, FINAL_EPOCHS + 1):
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
        full_loss, full_acc, full_f1, full_auc = evaluate_on_loader(model, eval_loader, criterion)

        writer.add_scalar("Loss/train_augmented", train_loss, epoch)
        writer.add_scalar("Loss/full_eval", full_loss, epoch)
        writer.add_scalar("Accuracy/full_eval", full_acc, epoch)
        writer.add_scalar("F1/full_eval", full_f1, epoch)
        writer.add_scalar("AUC/full_eval", full_auc, epoch)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} train_auc={train_auc:.4f} | "
            f"full_loss={full_loss:.4f} full_acc={full_acc:.4f} full_f1={full_f1:.4f} full_auc={full_auc:.4f}",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_auc": train_auc,
                "full_loss": full_loss,
                "full_acc": full_acc,
                "full_f1": full_f1,
                "full_auc": full_auc,
            }
        )

        if full_loss < best_train_loss:
            best_train_loss = full_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No final model state was saved.")

    best_path = os.path.join(TRAIN_OUT_DIR, "best_csPCa_mri_final.pt")
    last_path = os.path.join(TRAIN_OUT_DIR, "last_csPCa_mri_final.pt")
    torch.save(best_state, best_path)
    torch.save(model.state_dict(), last_path)

    pd.DataFrame(history).to_csv(os.path.join(TRAIN_OUT_DIR, "final_training_history.csv"), index=False)
    with open(os.path.join(TRAIN_OUT_DIR, "final_training_config.json"), "w") as f:
        json.dump(
            {
                "root_dir": PICAI_ROOT_DIR,
                "labels_csv": PICAI_LABELS_CSV,
                "gland_mask_dir": PICAI_GLAND_MASK_DIR,
                "num_cases": int(len(items)),
                "class_counts": class_counts.tolist(),
                "batch_size": BATCH_SIZE,
                "final_epochs": FINAL_EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "target_shape": list(TARGET_SHAPE),
                "embed_dim": EMBED_DIM,
                "pool1": list(POOL1_KERNEL),
                "pool2": list(POOL2_KERNEL),
                "pool3": list(POOL3_KERNEL),
            },
            f,
            indent=2,
        )

    writer.close()
    print(f"Saved best final model to: {best_path}", flush=True)
    print(f"Saved last final model to: {last_path}", flush=True)



def extract_chimera_embeddings():
    model_path = os.path.join(TRAIN_OUT_DIR, "best_csPCa_mri_final.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Final model not found: {model_path}")

    print(f"Loading final model: {model_path}", flush=True)
    model = CsPCaClassifierThreeStemSharedTrunk(embed_dim=EMBED_DIM).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)

    chimera_labels = {pid: 1 for pid in (TRAIN_BCR_POS + TEST_BCR_POS)}
    chimera_labels.update({pid: 0 for pid in (TRAIN_BCR_NEG + TEST_BCR_NEG)})
    chimera_all_ids = sorted(set(TRAIN_BCR_POS + TRAIN_BCR_NEG + TEST_BCR_POS + TEST_BCR_NEG), key=int)

    all_items = find_chimera_cases(CHIMERA_ROOT_DIR, chimera_all_ids, chimera_labels, split_name="all")

    print(f"Loaded CHIMERA usable cases: {len(all_items)}", flush=True)

    if len(all_items) == 0:
        raise RuntimeError("No CHIMERA cases found. Check CHIMERA_ROOT_DIR.")

    ds = MultiModalMriDataset(all_items, target_shape=TARGET_SHAPE, augment=False, require_label=True)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    extract_embeddings(model, loader, CHIMERA_OUT_DIR, prefix="chimera_all")

    labels_df = pd.DataFrame({
        "patient_id": [item["patient_id"] for item in all_items],
        "case_id": [item["case_id"] for item in all_items],
        "label": [item["label"] for item in all_items],
    })
    labels_df.to_csv(os.path.join(CHIMERA_OUT_DIR, "chimera_all_labels.csv"), index=False)

    with open(os.path.join(CHIMERA_OUT_DIR, "chimera_all_cohort_meta.json"), "w") as f:
        json.dump(
            {
                "num_cases": int(len(all_items)),
                "num_pos": int(sum(item["label"] for item in all_items)),
                "num_neg": int(len(all_items) - sum(item["label"] for item in all_items)),
                "source_model": model_path,
            },
            f,
            indent=2,
        )

    print(f"Saved CHIMERA cohort embeddings to: {CHIMERA_OUT_DIR}", flush=True)


if __name__ == "__main__":
    print("Entered main()", flush=True)
    if RUN_TRAIN_FINAL:
        train_final_model()
    if RUN_EXTRACT_CHIMERA:
        extract_chimera_embeddings()
