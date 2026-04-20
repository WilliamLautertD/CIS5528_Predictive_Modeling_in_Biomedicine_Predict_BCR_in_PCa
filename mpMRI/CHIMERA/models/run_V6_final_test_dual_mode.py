#!/bin/bash
#SBATCH -p h200
#SBATCH --job-name=chimera_dual_mode_128
#SBATCH --output=chimera_dual_mode_128_%j.log
#SBATCH --error=chimera_dual_mode_128_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mail-user=William.LautertDutra@fccc.edu
#SBATCH --mail-type=END,FAIL


#cd /home/lauterw/project_MRI/radiology

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda run -n ML_CHIMERA_GPU python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

conda run -n ML_CHIMERA_GPU python V6_final_test_dual_mode.py
