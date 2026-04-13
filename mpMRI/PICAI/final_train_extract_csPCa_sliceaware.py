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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =========================
# 1) CONFIG
# =========================
# Modes
RUN_TRAIN_FINAL = True
RUN_EXTRACT_CHIMERA = True

# Source cohort: PI-CAI
PICAI_ROOT_DIR = "images"
PICAI_LABELS_CSV = "marksheet-2.csv"
PICAI_GLAND_MASK_DIR = "picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23"

# Target cohort: CHIMERA
CHIMERA_ROOT_DIR = "CHIMERA/radiology/images"

TRAIN_BCR_POS = ['1003', '1010', '1030', '1041', '1064', '1086', '1100', '1106', '1127', '1135', '1137', '1165', '1169', '1185', '1195', '1208', '1214', '1217', '1219', '1240', '1258', '1287']
TRAIN_BCR_NEG = ['1021', '1026', '1028', '1031', '1035', '1048', '1052', '1056', '1060', '1062', '1066', '1071', '1094', '1112', '1114', '1117', '1123', '1129', '1136', '1140', '1141', '1149', '1174', '1179', '1186', '1188', '1192', '1205', '1206', '1207', '1211', '1212', '1216', '1223', '1227', '1260', '1261', '1262', '1264', '1269', '1279', '1282', '1284', '1285', '1290', '1291', '1293', '1294', '1296', '1298', '1299', '1301', '1303', '1304']
TEST_BCR_POS = ['1090', '1122', '1130', '1193', '1295']
TEST_BCR_NEG = ['1011', '1025', '1036', '1037', '1039', '1068', '1107', '1120', '1151', '1184', '1235', '1252', '1267', '1286']

CHIMERA_ALL = sorted(set(TRAIN_BCR_POS + TRAIN_BCR_NEG + TEST_BCR_POS + TEST_BCR_NEG))
CHIMERA_LABELS = {pid: 1 for pid in (TRAIN_BCR_POS + TEST_BCR_POS)}
CHIMERA_LABELS.update({pid: 0 for pid in (TRAIN_BCR_NEG + TEST_BCR_NEG)})

# Output
OUT_DIR = "outputs_csPCa_sliceaware_final_and_CHIMERA_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "best_csPCa_mri_final.pt")

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SHAPE = (48, 96, 96)
SEED = 42
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Embedding mode: choose from {"48x128", "24x128"}
SLICE_EMBED_MODE = "48x128"
SLICE_EMBED_DIM = 128
SLICE_TOKENS = 48 if SLICE_EMBED_MODE == "48x128" else 24

# Architecture pooling kernels
POOL1_KERNEL = (1, 2, 2)
POOL2_KERNEL = (1, 2, 2)
POOL3_KERNEL = (1, 2, 2)  # no depth reduction to preserve 48 along depth


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
    x = torch.from_numpy(volume).unsqueeze(0).float()  # (1, C, D, H, W)
    if mode in ["trilinear", "bilinear", "linear"]:
        x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=target_shape, mode=mode)
    return x.squeeze(0).cpu().numpy()


# =========================
# 4) CASE DISCOVERY
# =========================
def load_picai_labels(csv_path):
    df = pd.read_csv(csv_path)
    need = {"patient_id", "study_id", "case_csPCa"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df[["patient_id", "study_id", "case_csPCa"]].copy()
    df = df.rename(columns={"case_csPCa": "label"})
    df["patient_id"] = df["patient_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)
    bad = ~df["label"].isin([0, 1])
    if bad.any():
        raise ValueError("Found labels outside {0,1} in marksheet-2.csv")
    return df


def find_picai_case_files(root_dir, labels_df, gland_mask_dir):
    items = []
    for _, row in labels_df.iterrows():
        patient_id = str(row["patient_id"])
        study_id = str(row["study_id"])
        label = int(row["label"])

        case_id = f"{patient_id}_{study_id}"
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"[WARN] Missing PI-CAI folder for patient {patient_id}", flush=True)
            continue

        t2w = glob.glob(os.path.join(patient_dir, f"{case_id}_t2w.mha"))
        adc = glob.glob(os.path.join(patient_dir, f"{case_id}_adc.mha"))
        hbv = glob.glob(os.path.join(patient_dir, f"{case_id}_hbv.mha"))
        mask_path = os.path.join(gland_mask_dir, f"{case_id}.nii.gz")

        missing = []
        if len(t2w) != 1:
            missing.append("t2w")
        if len(adc) != 1:
            missing.append("adc")
        if len(hbv) != 1:
            missing.append("hbv")
        if not os.path.isfile(mask_path):
            missing.append("gland_mask")

        if missing:
            print(f"[WARN] Incomplete PI-CAI case {case_id}: missing {', '.join(missing)}", flush=True)
            continue

        items.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "study_id": study_id,
            "label": label,
            "t2w": t2w[0],
            "adc": adc[0],
            "hbv": hbv[0],
            "mask": mask_path,
        })
    return items


def find_chimera_case_files(root_dir, patient_ids, label_map):
    items = []
    for patient_id in patient_ids:
        patient_id = str(patient_id)
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"[WARN] Missing CHIMERA folder for patient {patient_id}", flush=True)
            continue

        case_prefix = f"{patient_id}_0001"
        t2w = glob.glob(os.path.join(patient_dir, f"{case_prefix}_t2w.mha"))
        adc = glob.glob(os.path.join(patient_dir, f"{case_prefix}_adc.mha"))
        hbv = glob.glob(os.path.join(patient_dir, f"{case_prefix}_hbv.mha"))
        mask = glob.glob(os.path.join(patient_dir, f"{case_prefix}_mask.mha"))

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
            print(f"[WARN] Incomplete CHIMERA case {patient_id}: missing {', '.join(missing)}", flush=True)
            continue

        items.append({
            "case_id": patient_id,
            "patient_id": patient_id,
            "study_id": "0001",
            "label": int(label_map[patient_id]),
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

        if min(t2w.shape) == 0 or min(adc.shape) == 0 or min(hbv.shape) == 0:
            raise RuntimeError(f"Empty crop for case {item['case_id']}")

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
            "patient_id": item["patient_id"],
            "study_id": item["study_id"],
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


class MRIEncoder3DSliceAware(nn.Module):
    """
    Slice-aware encoder with outputs:
      - 48x128 or 24x128, depending on num_tokens.

    Input:  (B, 3, 48, 96, 96)
    Output: (B, num_tokens, 128)
    """
    def __init__(self, embed_dim=128, num_tokens=48, dropout=0.3):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.stem_t2w = ConvBlock(1, 8)
        self.stem_adc = ConvBlock(1, 8)
        self.stem_hbv = ConvBlock(1, 8)

        self.pool1 = nn.MaxPool3d(kernel_size=POOL1_KERNEL, stride=POOL1_KERNEL)
        self.enc2 = ConvBlock(24, 16)
        self.pool2 = nn.MaxPool3d(kernel_size=POOL2_KERNEL, stride=POOL2_KERNEL)
        self.enc3 = ConvBlock(16, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=POOL3_KERNEL, stride=POOL3_KERNEL)
        self.enc4 = ConvBlock(32, 64)

        self.dropout = nn.Dropout(dropout)
        self.slice_fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x_t2w = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]
        x_hbv = x[:, 2:3, ...]

        f_t2w = self.pool1(self.stem_t2w(x_t2w))
        f_adc = self.pool1(self.stem_adc(x_adc))
        f_hbv = self.pool1(self.stem_hbv(x_hbv))

        x = torch.cat([f_t2w, f_adc, f_hbv], dim=1)  # (B,24,48,48,48)
        x = self.pool2(self.enc2(x))                 # (B,16,48,24,24)
        x = self.pool3(self.enc3(x))                 # (B,32,48,12,12)
        x = self.enc4(x)                             # (B,64,48,12,12)

        x = x.mean(dim=[3, 4])                       # (B,64,48)

        if self.num_tokens != x.shape[-1]:
            x = F.interpolate(x, size=self.num_tokens, mode="linear", align_corners=False)

        x = x.transpose(1, 2)                        # (B,num_tokens,64)
        x = self.dropout(x)
        z_seq = self.slice_fc(x)                     # (B,num_tokens,128)
        return z_seq


class csPCaClassifierSliceAware(nn.Module):
    def __init__(self, embed_dim=128, num_tokens=48, dropout=0.3):
        super().__init__()
        self.encoder = MRIEncoder3DSliceAware(embed_dim=embed_dim, num_tokens=num_tokens, dropout=dropout)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        z_seq = self.encoder(x)          # (B,T,128)
        z_global = z_seq.mean(dim=1)     # (B,128)
        logits = self.head(z_global)     # (B,2)
        return logits, z_seq, z_global


# =========================
# 7) TRAIN / EVAL
# =========================
def compute_class_weights(items, device):
    labels = np.array([x["label"] for x in items])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return class_counts, class_weights


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print("First training batch loaded", flush=True)

        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds = (prob >= 0.5).long()

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    return total_loss / len(loader.dataset), acc, f1, auc


def eval_full_dataset(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            logits, _, _ = model(x)
            loss = criterion(logits, y)

            prob = torch.softmax(logits, dim=1)[:, 1]
            preds = (prob >= 0.5).long()

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
def extract_slice_embeddings(model, loader, out_prefix):
    model.eval()
    all_seq, all_global = [], []
    case_ids, patient_ids, study_ids, labels = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            ids = batch["case_id"]
            pids = batch["patient_id"]
            sids = batch["study_id"]

            _, z_seq, z_global = model(x)
            z_seq = F.normalize(z_seq, p=2, dim=2).cpu().numpy()      # (B,T,128)
            z_global = F.normalize(z_global, p=2, dim=1).cpu().numpy()  # (B,128)

            all_seq.append(z_seq)
            all_global.append(z_global)
            case_ids.extend(ids)
            patient_ids.extend(pids)
            study_ids.extend(sids)
            labels.extend(y.tolist())

    all_seq = np.concatenate(all_seq, axis=0)
    all_global = np.concatenate(all_global, axis=0)

    np.save(f"{out_prefix}_slice_embeddings.npy", all_seq)
    np.save(f"{out_prefix}_global_embeddings.npy", all_global)

    ids_df = pd.DataFrame({
        "case_id": case_ids,
        "patient_id": patient_ids,
        "study_id": study_ids,
        "label": labels,
    })
    ids_df.to_csv(f"{out_prefix}_embedding_ids.csv", index=False)

    with open(f"{out_prefix}_embedding_meta.json", "w") as f:
        json.dump({
            "num_cases": int(len(case_ids)),
            "slice_embedding_shape": [int(all_seq.shape[1]), int(all_seq.shape[2])],
            "global_embedding_dim": int(all_global.shape[1]),
            "target_shape": list(TARGET_SHAPE),
            "channels": ["t2w", "adc", "hbv"],
            "slice_mode": SLICE_EMBED_MODE,
            "pooling": {
                "pool1": list(POOL1_KERNEL),
                "pool2": list(POOL2_KERNEL),
                "pool3": list(POOL3_KERNEL),
            },
            "notes": "slice embeddings preserve depth tokens and global embedding is mean over tokens",
        }, f, indent=2)


# =========================
# 9) MAIN
# =========================
def main():
    print("Entered main()", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU count visible: {torch.cuda.device_count()}", flush=True)

    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"Labels CSV: {PICAI_LABELS_CSV}", flush=True)
    print(f"Gland mask dir: {PICAI_GLAND_MASK_DIR}", flush=True)
    print(f"CHIMERA_ROOT_DIR = {CHIMERA_ROOT_DIR}", flush=True)
    print(f"CHIMERA exists? {os.path.isdir(CHIMERA_ROOT_DIR)}", flush=True)
    print(f"NUM_WORKERS = {NUM_WORKERS}", flush=True)
    print(f"Slice embedding mode = {SLICE_EMBED_MODE}", flush=True)
    print(f"Slice tokens = {SLICE_TOKENS}, embed dim = {SLICE_EMBED_DIM}", flush=True)

    model = csPCaClassifierSliceAware(embed_dim=SLICE_EMBED_DIM, num_tokens=SLICE_TOKENS).to(DEVICE)

    if RUN_TRAIN_FINAL:
        labels_df = load_picai_labels(PICAI_LABELS_CSV)
        print(f"Loaded rows from CSV: {len(labels_df)}", flush=True)
        print("Raw class distribution from CSV:", flush=True)
        print(labels_df["label"].value_counts().sort_index(), flush=True)

        items = find_picai_case_files(PICAI_ROOT_DIR, labels_df, PICAI_GLAND_MASK_DIR)
        print(f"Loaded usable cases: {len(items)}", flush=True)
        usable_counts = {0: sum(x['label'] == 0 for x in items), 1: sum(x['label'] == 1 for x in items)}
        print(f"Usable class distribution: {usable_counts}", flush=True)

        train_ds = MultiModalMriDataset(items, target_shape=TARGET_SHAPE, augment=True)
        full_eval_ds = MultiModalMriDataset(items, target_shape=TARGET_SHAPE, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        full_loader = DataLoader(full_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        class_counts, class_weights = compute_class_weights(items, DEVICE)
        print(f"Final-train class counts: {class_counts.tolist()}", flush=True)
        print(f"Final-train class weights: {class_weights.detach().cpu().numpy().tolist()}", flush=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_state = None
        best_loss = float("inf")
        patience = 0
        history = []

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
            full_loss, full_acc, full_f1, full_auc = eval_full_dataset(model, full_loader, criterion)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_auc": train_auc,
                "full_loss": full_loss,
                "full_acc": full_acc,
                "full_f1": full_f1,
                "full_auc": full_auc,
            })
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} train_auc={train_auc:.4f} | "
                f"full_loss={full_loss:.4f} full_acc={full_acc:.4f} full_f1={full_f1:.4f} full_auc={full_auc:.4f}",
                flush=True,
            )

            if full_loss < best_loss:
                best_loss = full_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping final training", flush=True)
                    break

        if best_state is None:
            raise RuntimeError("No final model state was saved.")

        torch.save(best_state, FINAL_MODEL_PATH)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "last_csPCa_mri_final.pt"))
        pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "final_training_history.csv"), index=False)
        with open(os.path.join(OUT_DIR, "final_model_meta.json"), "w") as f:
            json.dump({
                "slice_mode": SLICE_EMBED_MODE,
                "slice_tokens": SLICE_TOKENS,
                "slice_embed_dim": SLICE_EMBED_DIM,
                "target_shape": list(TARGET_SHAPE),
                "channels": ["t2w", "adc", "hbv"],
                "device": DEVICE,
            }, f, indent=2)
        print(f"Saved best final model to: {FINAL_MODEL_PATH}", flush=True)
        print(f"Saved last final model to: {os.path.join(OUT_DIR, 'last_csPCa_mri_final.pt')}", flush=True)

    if RUN_EXTRACT_CHIMERA:
        if not os.path.isfile(FINAL_MODEL_PATH):
            raise FileNotFoundError(f"Final model not found: {FINAL_MODEL_PATH}")
        print(f"Loading final model: {FINAL_MODEL_PATH}", flush=True)
        state = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        chimera_items = find_chimera_case_files(CHIMERA_ROOT_DIR, CHIMERA_ALL, CHIMERA_LABELS)
        print(f"Loaded CHIMERA usable cases: {len(chimera_items)}", flush=True)
        if len(chimera_items) == 0:
            print("No CHIMERA cases found. Check CHIMERA_ROOT_DIR.", flush=True)
            return

        chimera_ds = MultiModalMriDataset(chimera_items, target_shape=TARGET_SHAPE, augment=False)
        chimera_loader = DataLoader(chimera_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        extract_slice_embeddings(model, chimera_loader, os.path.join(OUT_DIR, "chimera_all"))
        pd.DataFrame({
            "patient_id": [x["patient_id"] for x in chimera_items],
            "label": [x["label"] for x in chimera_items],
        }).to_csv(os.path.join(OUT_DIR, "chimera_all_labels.csv"), index=False)
        print("Saved CHIMERA slice-aware embeddings and labels.", flush=True)


if __name__ == "__main__":
    main()
