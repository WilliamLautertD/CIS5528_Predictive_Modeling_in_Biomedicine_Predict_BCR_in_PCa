
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
EMBEDDINGS_NPY = "../Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_128D/chimera_all_embeddings.npy"
EMBEDDING_IDS_CSV = "../Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_128D/chimera_all_embedding_ids.csv"
OUT_DIR = "../Pre-trained_PICAI+CHIMERA/performance/outputs_bcr_from_embeddings_predefined_128D"

SEED = 42
AGGREGATION = "auto"   # auto | mean_tokens | flatten_tokens

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
TEST_BCR_NEG = ['1011', '1025', '1036', '1037', '1039', '1068', '1107',
                '1120', '1151', '1184', '1235', '1252', '1267', '1286']

ALL_LABELS = {cid: 1 for cid in TRAIN_BCR_POS + TEST_BCR_POS}
ALL_LABELS.update({cid: 0 for cid in TRAIN_BCR_NEG + TEST_BCR_NEG})


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

    ids_df["label"] = ids_df["patient_id"].map(ALL_LABELS)
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


def evaluate_split(model, X, df, split_name: str, out_dir: str) -> Dict:
    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(df["label"].values, y_prob, threshold=0.5)

    pred_df = df.copy()
    pred_df["split"] = split_name
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = (y_prob >= 0.5).astype(int)
    pred_df.to_csv(os.path.join(out_dir, f"{split_name}_predictions.csv"), index=False)

    return metrics


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

    train_ids = set(TRAIN_BCR_POS + TRAIN_BCR_NEG)
    test_ids = set(TEST_BCR_POS + TEST_BCR_NEG)

    train_mask = ids_df["patient_id"].isin(train_ids).values
    test_mask = ids_df["patient_id"].isin(test_ids).values

    train_df = ids_df.loc[train_mask].reset_index(drop=True)
    test_df = ids_df.loc[test_mask].reset_index(drop=True)

    X_train = X[train_mask]
    X_test = X[test_mask]

    print(f"Embeddings shape used for classification: {X.shape}")
    print(f"Train cases: {len(train_df)}")
    print(f"Test cases:  {len(test_df)}")

    model = build_classifier()
    model.fit(X_train, train_df["label"].values)

    train_metrics = evaluate_split(model, X_train, train_df, "train", OUT_DIR)
    test_metrics = evaluate_split(model, X_test, test_df, "test", OUT_DIR)

    metrics_df = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "test", **test_metrics},
    ])
    metrics_df.to_csv(os.path.join(OUT_DIR, "predefined_metrics.csv"), index=False)

    with open(os.path.join(OUT_DIR, "predefined_config.json"), "w") as f:
        json.dump({
            "embeddings_npy": embeddings_npy,
            "embedding_ids_csv": ids_csv,
            "aggregation_meta": meta,
            "classifier": "StandardScaler + LogisticRegression(class_weight='balanced', solver='liblinear')",
            "threshold": 0.5,
            "seed": SEED,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }, f, indent=2)

    print("\n=== PREDEFINED RESULTS ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
