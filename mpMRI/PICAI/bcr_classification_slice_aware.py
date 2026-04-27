import os
import json
import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================================================
# CONFIG
# =========================================================
# Input slice-aware embeddings from CHIMERA extraction
EMBEDDINGS_NPY = "Pre-trained_PICAI+CHIMERA/embeddings/outputs_csPCa_sliceaware_final_and_CHIMERA_embeddings_48x128D_batch_12/chimera_all_slice_embeddings.npy"
EMBEDDING_IDS_CSV = "Pre-trained_PICAI+CHIMERA/embeddings/outputs_csPCa_sliceaware_final_and_CHIMERA_embeddings_48x128D_batch_12/chimera_all_embedding_ids.csv"

# Which split protocol to run: {"predefined", "hardcoded", "both"}
RUN_MODE = "both"

# How to convert (N, T, D) slice-aware embeddings into patient-level features
# Options:
#   "mean"      -> average over tokens => (N, D)
#   "max"       -> max over tokens => (N, D)
#   "mean_max"  -> concat(mean, max) => (N, 2D)
#   "flatten"   -> flatten all tokens => (N, T*D)
REDUCTION = "mean_max"

# Repro / classifier
SEED = 42
MAX_ITER = 5000
C = 1.0
THRESHOLD = 0.45

# Output
BASE_OUT_DIR = "Pre-trained_PICAI+CHIMERA/performance/outputs_csPCa_sliceaware_final_and_CHIMERA_embeddings_48x128D_batch_12"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# =========================================================
# CHIMERA LABELS / SPLITS
# =========================================================
TRAIN_BCR_POS = ['1003', '1010', '1030', '1041', '1064', '1086', '1100', '1106', '1127', '1135',
                 '1137', '1165', '1169', '1185', '1195', '1208', '1214', '1217', '1219', '1240',
                 '1258', '1287']
TRAIN_BCR_NEG = ['1021', '1026', '1028', '1031', '1035', '1048', '1052', '1056', '1060', '1062',
                 '1066', '1071', '1094', '1112', '1114', '1117', '1123', '1129', '1136', '1140',
                 '1141', '1149', '1174', '1179', '1186', '1188', '1192', '1205', '1206', '1207',
                 '1211', '1212', '1216', '1223', '1227', '1260', '1261', '1262', '1264', '1269',
                 '1279', '1282', '1284', '1285', '1290', '1291', '1293', '1294', '1296', '1298',
                 '1299', '1301', '1303', '1304']
TEST_BCR_POS = ['1090', '1122', '1130', '1193', '1295']
TEST_BCR_NEG = ['1011', '1025', '1036', '1037', '1039', '1068', '1107', '1120', '1151', '1184', '1235', '1252', '1267', '1286']

PREDEFINED_LABELS = {pid: 1 for pid in TRAIN_BCR_POS + TEST_BCR_POS}
PREDEFINED_LABELS.update({pid: 0 for pid in TRAIN_BCR_NEG + TEST_BCR_NEG})

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


# =========================================================
# HELPERS
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def reduce_slice_embeddings(z: np.ndarray, mode: str) -> np.ndarray:
    if z.ndim == 2:
        return z.astype(np.float32)
    if z.ndim != 3:
        raise ValueError(f"Expected embeddings with ndim 2 or 3, got shape {z.shape}")

    if mode == "mean":
        out = z.mean(axis=1)
    elif mode == "max":
        out = z.max(axis=1)
    elif mode == "mean_max":
        out = np.concatenate([z.mean(axis=1), z.max(axis=1)], axis=1)
    elif mode == "flatten":
        out = z.reshape(z.shape[0], -1)
    else:
        raise ValueError(f"Unknown REDUCTION mode: {mode}")
    return out.astype(np.float32)


def load_embeddings(embeddings_npy: str, ids_csv: str, reduction: str) -> pd.DataFrame:
    z = np.load(embeddings_npy)
    ids_df = pd.read_csv(ids_csv)

    if len(ids_df) != z.shape[0]:
        raise ValueError(
            f"Mismatch between ids rows ({len(ids_df)}) and embeddings ({z.shape[0]})"
        )

    if "patient_id" in ids_df.columns:
        patient_ids = ids_df["patient_id"].astype(str).tolist()
    elif "case_id" in ids_df.columns:
        patient_ids = ids_df["case_id"].astype(str).apply(lambda x: x.split("_")[0]).tolist()
    else:
        raise ValueError("IDs CSV must contain patient_id or case_id")

    X = reduce_slice_embeddings(z, reduction)
    feat_cols = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df.insert(0, "patient_id", patient_ids)

    if "label" in ids_df.columns:
        df["label_from_file"] = ids_df["label"].astype(int).values

    # collapse duplicates if any appear unexpectedly
    if df["patient_id"].duplicated().any():
        agg_dict = {c: "mean" for c in feat_cols}
        if "label_from_file" in df.columns:
            agg_dict["label_from_file"] = "first"
        df = df.groupby("patient_id", as_index=False).agg(agg_dict)

    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                random_state=SEED,
                max_iter=MAX_ITER,
                C=C,
                solver="liblinear",
            ),
        ),
    ])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = float("nan")
    return out


def fit_and_predict(X_train, y_train, X_eval):
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_eval)[:, 1]
    return pipe, prob


def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# =========================================================
# PREDEFINED TRAIN/TEST
# =========================================================
def run_predefined(df: pd.DataFrame):
    out_dir = os.path.join(BASE_OUT_DIR, f"predefined_{REDUCTION}")
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df = df[df["patient_id"].isin(PREDEFINED_LABELS)].copy()
    df["label"] = df["patient_id"].map(PREDEFINED_LABELS).astype(int)
    df["split"] = np.where(df["patient_id"].isin(TRAIN_BCR_POS + TRAIN_BCR_NEG), "train", "test")

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["label"].values

    model, train_prob = fit_and_predict(X_train, y_train, X_train)
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, train_prob, THRESHOLD)
    test_metrics = compute_metrics(y_test, test_prob, THRESHOLD)

    train_pred = train_df[["patient_id", "label"]].copy()
    train_pred["y_prob"] = train_prob
    train_pred["y_pred"] = (train_prob >= THRESHOLD).astype(int)
    train_pred.to_csv(os.path.join(out_dir, "train_predictions.csv"), index=False)

    test_pred = test_df[["patient_id", "label"]].copy()
    test_pred["y_prob"] = test_prob
    test_pred["y_pred"] = (test_prob >= THRESHOLD).astype(int)
    test_pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    metrics_df = pd.DataFrame([
        {"split": "train", **train_metrics, "n": len(train_df)},
        {"split": "test", **test_metrics, "n": len(test_df)},
    ])
    metrics_df.to_csv(os.path.join(out_dir, "predefined_metrics.csv"), index=False)

    save_json(
        os.path.join(out_dir, "config.json"),
        {
            "embeddings_npy": EMBEDDINGS_NPY,
            "embedding_ids_csv": EMBEDDING_IDS_CSV,
            "run_mode": "predefined",
            "reduction": REDUCTION,
            "classifier": "StandardScaler + LogisticRegression(class_weight='balanced')",
            "threshold": THRESHOLD,
            "train_n": len(train_df),
            "test_n": len(test_df),
        },
    )

    print("\n=== PREDEFINED SPLIT ===")
    print(metrics_df)


# =========================================================
# HARDCODED 5-FOLD CV
# =========================================================
def run_hardcoded(df: pd.DataFrame):
    out_dir = os.path.join(BASE_OUT_DIR, f"hardcoded_{REDUCTION}")
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df = df[df["patient_id"].isin(ALL_IDS)].copy()
    df["label"] = df["patient_id"].map(CASE_LABELS).astype(int)

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    patient_to_row = {pid: i for i, pid in enumerate(df["patient_id"].tolist())}

    missing_ids = [pid for pid in ALL_IDS if pid not in patient_to_row]
    if missing_ids:
        raise ValueError(f"Missing embeddings for hardcoded patients: {missing_ids}")

    fold_metrics = []
    fold_pred_dfs = []

    for fold in range(N_FOLDS):
        val_ids = FOLD_VAL_POS[fold] + FOLD_VAL_NEG[fold]
        train_ids = [pid for pid in ALL_IDS if pid not in set(val_ids)]

        train_rows = [patient_to_row[pid] for pid in train_ids]
        val_rows = [patient_to_row[pid] for pid in val_ids]

        train_df = df.iloc[train_rows].copy()
        val_df = df.iloc[val_rows].copy()

        X_train = train_df[feat_cols].values
        y_train = train_df["label"].values
        X_val = val_df[feat_cols].values
        y_val = val_df["label"].values

        model, train_prob = fit_and_predict(X_train, y_train, X_train)
        val_prob = model.predict_proba(X_val)[:, 1]

        train_metrics = compute_metrics(y_train, train_prob, THRESHOLD)
        val_metrics = compute_metrics(y_val, val_prob, THRESHOLD)

        fold_metrics.append({
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

        pred_df = val_df[["patient_id", "label"]].copy()
        pred_df.insert(0, "fold", fold + 1)
        pred_df["y_prob"] = val_prob
        pred_df["y_pred"] = (val_prob >= THRESHOLD).astype(int)
        fold_pred_dfs.append(pred_df)

        print(f"\n========== TRAIN/TEST SPLIT (fold {fold + 1}/{N_FOLDS}) ==========")
        print(f"  Train: {train_ids}")
        print(f"  Val:   {val_ids}")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(os.path.join(out_dir, "hardcoded_fold_metrics.csv"), index=False)

    all_val_pred_df = pd.concat(fold_pred_dfs, axis=0, ignore_index=True)
    all_val_pred_df.to_csv(os.path.join(out_dir, "hardcoded_all_val_predictions.csv"), index=False)

    summary_df = pd.DataFrame([
        {
            "metric": "mean",
            "val_acc": fold_metrics_df["val_acc"].mean(),
            "val_f1": fold_metrics_df["val_f1"].mean(),
            "val_auc": fold_metrics_df["val_auc"].mean(),
        },
        {
            "metric": "std",
            "val_acc": fold_metrics_df["val_acc"].std(),
            "val_f1": fold_metrics_df["val_f1"].std(),
            "val_auc": fold_metrics_df["val_auc"].std(),
        },
    ])
    summary_df.to_csv(os.path.join(out_dir, "hardcoded_cv_summary.csv"), index=False)

    save_json(
        os.path.join(out_dir, "config.json"),
        {
            "embeddings_npy": EMBEDDINGS_NPY,
            "embedding_ids_csv": EMBEDDING_IDS_CSV,
            "run_mode": "hardcoded",
            "reduction": REDUCTION,
            "classifier": "StandardScaler + LogisticRegression(class_weight='balanced')",
            "threshold": THRESHOLD,
            "n_folds": N_FOLDS,
        },
    )

    print("\n=== HARDCODED 5-FOLD SUMMARY ===")
    print(fold_metrics_df)
    print("\nSummary:")
    print(summary_df)


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    print(f"Loading slice-aware embeddings: {EMBEDDINGS_NPY}")
    print(f"Loading IDs CSV: {EMBEDDING_IDS_CSV}")
    print(f"Reduction mode: {REDUCTION}")
    print(f"Run mode: {RUN_MODE}")

    df = load_embeddings(EMBEDDINGS_NPY, EMBEDDING_IDS_CSV, REDUCTION)
    print(f"Loaded patient-level feature table: {df.shape}")

    if RUN_MODE in {"predefined", "both"}:
        run_predefined(df)
    if RUN_MODE in {"hardcoded", "both"}:
        run_hardcoded(df)


if __name__ == "__main__":
    main()
