
import os
import json
import glob
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# ============================================================
# CONFIG
# ============================================================
EMBEDDINGS_NPY = "Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_128D_batch_12/chimera_all_global_embeddings.npy"
EMBEDDING_IDS_CSV = "Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_128D_batch_12/chimera_all_embedding_ids.csv"
OUT_DIR = "Pre-trained_PICAI+CHIMERA/performance/outputs_bcr_from_embeddings_hardcoded_128D_batch_12"

SEED = 42
AGGREGATION = "auto"   # auto | mean_tokens | flatten_tokens

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
N_FOLDS = len(FOLD_VAL_POS)

ALL_BCR_POS = sorted({cid for fold in FOLD_VAL_POS for cid in fold}, key=int)
ALL_BCR_NEG = sorted({cid for fold in FOLD_VAL_NEG for cid in fold}, key=int)
ALL_IDS = ALL_BCR_POS + ALL_BCR_NEG
CASE_LABELS = {cid: 1 for cid in ALL_BCR_POS}
CASE_LABELS.update({cid: 0 for cid in ALL_BCR_NEG})


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_patient_id(case_id: str) -> str:
    case_id = str(case_id)
    if "_" in case_id:
        return case_id.split("_")[0]
    return case_id


def load_embeddings(embeddings_npy: str, ids_csv: str, aggregation: str = "auto") -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    Z = np.load(embeddings_npy)
    ids_df = pd.read_csv(ids_csv)

    if "case_id" not in ids_df.columns:
        raise ValueError("IDs CSV must contain a 'case_id' column.")

    ids_df = ids_df.copy()
    ids_df["patient_id"] = ids_df["case_id"].astype(str).map(extract_patient_id)

    meta = {
        "raw_shape": list(Z.shape),
        "aggregation": aggregation,
    }

    if Z.ndim == 2:
        X = Z.astype(np.float32)
        meta["used_shape"] = list(X.shape)
        meta["aggregation_applied"] = "none"
    elif Z.ndim == 3:
        if aggregation == "auto":
            aggregation = "mean_tokens"
        if aggregation == "mean_tokens":
            X = Z.mean(axis=1).astype(np.float32)
        elif aggregation == "flatten_tokens":
            X = Z.reshape(Z.shape[0], -1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported aggregation for 3D embeddings: {aggregation}")
        meta["used_shape"] = list(X.shape)
        meta["aggregation_applied"] = aggregation
    else:
        raise ValueError(f"Unsupported embedding shape: {Z.shape}")

    if len(ids_df) != X.shape[0]:
        raise ValueError(f"Mismatch: ids_csv has {len(ids_df)} rows but embeddings has {X.shape[0]} rows")

    ids_df["label"] = ids_df["patient_id"].map(CASE_LABELS)
    missing = ids_df[ids_df["label"].isna()]
    if len(missing) > 0:
        raise ValueError(f"Some patients are missing labels: {missing['patient_id'].tolist()}")

    ids_df["label"] = ids_df["label"].astype(int)
    return X, ids_df, meta


def build_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
            solver="liblinear",
        )),
    ])


def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
    }
    return out


def find_files_if_needed(embeddings_npy: str, ids_csv: str) -> Tuple[str, str]:
    if os.path.isfile(embeddings_npy) and os.path.isfile(ids_csv):
        return embeddings_npy, ids_csv

    npy_candidates = sorted(glob.glob("**/*embeddings*.npy", recursive=True))
    csv_candidates = sorted(glob.glob("**/*embedding_ids*.csv", recursive=True))

    if not npy_candidates or not csv_candidates:
        raise FileNotFoundError(
            "Could not find embedding files automatically. "
            "Set EMBEDDINGS_NPY and EMBEDDING_IDS_CSV at the top of the script."
        )

    chosen_npy = npy_candidates[0] if not os.path.isfile(embeddings_npy) else embeddings_npy
    chosen_csv = csv_candidates[0] if not os.path.isfile(ids_csv) else ids_csv
    return chosen_npy, chosen_csv


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    embeddings_npy, ids_csv = find_files_if_needed(EMBEDDINGS_NPY, EMBEDDING_IDS_CSV)
    print(f"Using embeddings: {embeddings_npy}")
    print(f"Using ids csv:    {ids_csv}")

    X, ids_df, meta = load_embeddings(embeddings_npy, ids_csv, aggregation=AGGREGATION)
    print(f"Embeddings shape used for classification: {X.shape}")

    patient_to_row = {pid: i for i, pid in enumerate(ids_df["patient_id"].tolist())}

    fold_results = []
    all_val_predictions = []

    for fold in range(N_FOLDS):
        val_ids = FOLD_VAL_POS[fold] + FOLD_VAL_NEG[fold]
        train_ids = [cid for cid in ALL_IDS if cid not in set(val_ids)]

        train_idx = np.array([patient_to_row[cid] for cid in train_ids], dtype=int)
        val_idx = np.array([patient_to_row[cid] for cid in val_ids], dtype=int)

        X_train = X[train_idx]
        X_val = X[val_idx]
        train_df = ids_df.iloc[train_idx].reset_index(drop=True).copy()
        val_df = ids_df.iloc[val_idx].reset_index(drop=True).copy()

        print(f"\n========== FOLD {fold + 1}/{N_FOLDS} ==========")
        print(f"Train N = {len(train_df)} | Val N = {len(val_df)}")

        model = build_classifier()
        model.fit(X_train, train_df["label"].values)

        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]

        train_metrics = compute_metrics(train_df["label"].values, train_prob, threshold=0.5)
        val_metrics = compute_metrics(val_df["label"].values, val_prob, threshold=0.5)

        fold_results.append({
            "fold": fold + 1,
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "n_train": len(train_df),
            "n_val": len(val_df),
        })

        fold_pred_df = val_df.copy()
        fold_pred_df["fold"] = fold + 1
        fold_pred_df["split"] = "val"
        fold_pred_df["y_prob"] = val_prob
        fold_pred_df["y_pred"] = (val_prob >= 0.5).astype(int)
        all_val_predictions.append(fold_pred_df)

        print(
            f"Fold {fold + 1} | "
            f"train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['f1']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

    df = pd.DataFrame(fold_results)
    df.to_csv(os.path.join(OUT_DIR, "hardcoded_fold_metrics.csv"), index=False)

    if all_val_predictions:
        pd.concat(all_val_predictions, axis=0, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "hardcoded_all_val_predictions.csv"), index=False
        )

    summary = pd.DataFrame([
        {
            "metric": "mean",
            "train_acc": df["train_acc"].mean(),
            "train_f1": df["train_f1"].mean(),
            "train_auc": df["train_auc"].mean(),
            "val_acc": df["val_acc"].mean(),
            "val_f1": df["val_f1"].mean(),
            "val_auc": df["val_auc"].mean(),
        },
        {
            "metric": "std",
            "train_acc": df["train_acc"].std(),
            "train_f1": df["train_f1"].std(),
            "train_auc": df["train_auc"].std(),
            "val_acc": df["val_acc"].std(),
            "val_f1": df["val_f1"].std(),
            "val_auc": df["val_auc"].std(),
        },
    ])
    summary.to_csv(os.path.join(OUT_DIR, "hardcoded_cv_summary.csv"), index=False)

    with open(os.path.join(OUT_DIR, "hardcoded_config.json"), "w") as f:
        json.dump({
            "embeddings_npy": embeddings_npy,
            "embedding_ids_csv": ids_csv,
            "aggregation_meta": meta,
            "classifier": "StandardScaler + LogisticRegression(class_weight='balanced', solver='liblinear')",
            "threshold": 0.5,
            "seed": SEED,
            "n_folds": N_FOLDS,
        }, f, indent=2)

    print("\n=== HARDCODED CV RESULTS ===")
    print(df.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
