#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J c100-r18-pl
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -t 24:00:00
#SBATCH --array=0-3

set -euo pipefail
set +u

# CIFAR-100 ResNet-18 InfoNCE/PL baseline reproduction.
# One array task per pretraining seed. Training entry point: main.py.
# Linear eval uses seeds 1..5. kNN eval uses split seeds 1..5.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs_cifar100_r18_baseline_repro}"
AGG_SCRIPT="${AGG_SCRIPT:-$REPO_ROOT/scripts/aggregate_table23_r18.py}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rlenv}"
SKIP_ENV="${SKIP_ENV:-0}"
SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"

SEEDS=(7 21 63 84)
LINEAR_EVAL_SEEDS=(1 2 3 4 5)
KNN_SPLIT_SEEDS=(1 2 3 4 5)

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#SEEDS[@]}" ]]; then
  echo "TASK_ID=$TASK_ID outside 0-$(( ${#SEEDS[@]} - 1 ))"
  exit 1
fi

SEED="${SEEDS[$TASK_ID]}"
RUN_DIR="$OUTPUT_ROOT/run_pl_seed${SEED}"
CKPT="$RUN_DIR/checkpoint_100.tar"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "SEED=$SEED"

if [[ "$SKIP_ENV" != "1" ]]; then
  module load miniforge/24.3.0-0 || true
  module load cuda/12.4.0 || true
  conda deactivate || true
  conda activate "$CONDA_ENV_NAME"
fi

if [[ "$SKIP_TRAIN_IF_DONE" == "1" && -f "$CKPT" ]]; then
  echo "Found existing checkpoint: $CKPT; skipping training."
else
  python main.py \
    --dataset CIFAR100 \
    --dataset_dir "$DATA_ROOT" \
    --workers 8 \
    --seed "$SEED" \
    --batch_size 256 \
    --image_size 32 \
    --epochs 100 \
    --resnet resnet18 \
    --projection_dim 64 \
    --optimizer Adam \
    --weight_decay 1.0e-6 \
    --temperature 0.5 \
    --policy pl \
    --model_path "$RUN_DIR" \
    --checkpoint_every 0
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Expected checkpoint not found: $CKPT"
  exit 1
fi

# Remove eval_seeds from config so linear_evaluation.py honors the CLI list below.
python - <<PY
from pathlib import Path
import yaml
p = Path("$RUN_DIR") / "config.yaml"
cfg = yaml.safe_load(p.read_text()) or {}
cfg.pop("eval_seeds", None)
cfg["epoch_num"] = 100
cfg["eval_batch_size"] = 256
cfg["eval_epochs"] = 100
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

if [[ "$OVERWRITE_EVAL" == "1" ]]; then
  rm -f "$RUN_DIR/linear_eval_results_v2.csv" "$RUN_DIR/knn_eval_results_v2.csv"
fi

if [[ "$FORCE_EVAL" == "1" || ! -f "$RUN_DIR/linear_eval_results_v2.csv" ]]; then
  python linear_evaluation.py "$CKPT" \
    --eval-seeds "${LINEAR_EVAL_SEEDS[@]}" \
    --eval-batch-size 256 \
    --eval-epochs 100
else
  echo "Found existing linear_eval_results_v2.csv; skipping linear eval."
fi

if [[ "$FORCE_EVAL" == "1" || ! -f "$RUN_DIR/knn_eval_results_v2.csv" ]]; then
  for SPLIT_SEED in "${KNN_SPLIT_SEEDS[@]}"; do
    python knn_eval.py "$CKPT" \
      --eval-batch-size 256 \
      --val-split-ratio 0.2 \
      --split-seed "$SPLIT_SEED" \
      --knn-temperature 0.1 \
      --knn-ks-eval 1 2 5 10 20 50
  done
else
  echo "Found existing knn_eval_results_v2.csv; skipping kNN eval."
fi

echo "Done baseline run: $RUN_DIR"
