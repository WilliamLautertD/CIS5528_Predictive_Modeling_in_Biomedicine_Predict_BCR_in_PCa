
"""
Convenience wrapper:
- runs predefined TRAIN/TEST classification
- runs hardcoded 5-fold classification

Edit the two file paths below once, then run this file.
"""

import os
import subprocess
import sys

EMBEDDINGS_NPY = "../Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_64D/all_embeddings.npy"
EMBEDDING_IDS_CSV = "../Pre-trained_PICAI+CHIMERA/embeddings/outputs_chimera_csPCa_embeddings_Dim_64D/all_embedding_ids.csv"

for script_name in [
    "bcr_classification_embeddings_predefined.py",
    "bcr_classification_embeddings_hardcoded.py",
]:
    with open(script_name, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace('EMBEDDINGS_NPY = "all_embeddings.npy"', f'EMBEDDINGS_NPY = "{EMBEDDINGS_NPY}"')
    text = text.replace('EMBEDDING_IDS_CSV = "all_embedding_ids.csv"', f'EMBEDDING_IDS_CSV = "{EMBEDDING_IDS_CSV}"')
    with open(script_name, "w", encoding="utf-8") as f:
        f.write(text)

for script_name in [
    "bcr_classification_embeddings_predefined.py",
    "bcr_classification_embeddings_hardcoded.py",
]:
    print(f"\nRunning {script_name}")
    result = subprocess.run([sys.executable, script_name], check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
