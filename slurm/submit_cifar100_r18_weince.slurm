#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J c100-r18-weince
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -t 30:00:00
#SBATCH --array=0-107

set -euo pipefail
set +u

# CIFAR-100 ResNet-18 WEINCE/tlambda_select reproduction.
# Full grid: 4 pretraining seeds x 3 select_aic_margin x 3 select_kappa_rho x 3 select_kappa_aic = 108 jobs.
# Training entry point: main_hard_select.py. Linear eval uses seed 1. kNN eval uses split seeds 1..5.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs_policy_tlambda_select_cifar100_r18_repro}"
AGG_SCRIPT="${AGG_SCRIPT:-$REPO_ROOT/scripts/aggregate_table23_r18.py}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rlenv}"
SKIP_ENV="${SKIP_ENV:-0}"
SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"

SEEDS=(7 21 63 84)
SELECT_AIC_MARGINS=(0 2 6)
SELECT_KAPPA_RHOS=(5 10 20)
SELECT_KAPPA_AICS=(0.5 1.0 2.0)
LINEAR_EVAL_SEEDS=(1)
KNN_SPLIT_SEEDS=(1 2 3 4 5)

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_ROOT"

N_SEEDS="${#SEEDS[@]}"
N_MARGINS="${#SELECT_AIC_MARGINS[@]}"
N_RHOS="${#SELECT_KAPPA_RHOS[@]}"
N_AICS="${#SELECT_KAPPA_AICS[@]}"
TOTAL=$(( N_SEEDS * N_MARGINS * N_RHOS * N_AICS ))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "$TOTAL" ]]; then
  echo "TASK_ID=$TASK_ID outside 0-$(( TOTAL - 1 ))"
  exit 1
fi

SEED_IDX=$(( TASK_ID % N_SEEDS ))
CFG_IDX=$(( TASK_ID / N_SEEDS ))
AIC_IDX=$(( CFG_IDX % N_AICS ))
RHO_IDX=$(( (CFG_IDX / N_AICS) % N_RHOS ))
MARGIN_IDX=$(( CFG_IDX / (N_AICS * N_RHOS) ))

SEED="${SEEDS[$SEED_IDX]}"
SELECT_AIC_MARGIN="${SELECT_AIC_MARGINS[$MARGIN_IDX]}"
SELECT_KAPPA_RHO="${SELECT_KAPPA_RHOS[$RHO_IDX]}"
SELECT_KAPPA_AIC="${SELECT_KAPPA_AICS[$AIC_IDX]}"

tag_float() { echo "$1" | sed 's/\.0$//' | sed 's/\./p/g'; }
MARGIN_TAG="$(tag_float "$SELECT_AIC_MARGIN")"
RHO_TAG="$(tag_float "$SELECT_KAPPA_RHO")"
AIC_TAG="$(tag_float "$SELECT_KAPPA_AIC")"

RUN_DIR="$OUTPUT_ROOT/run_tlambda_seed${SEED}_m${MARGIN_TAG}_rho${RHO_TAG}_aic${AIC_TAG}"
CKPT="$RUN_DIR/checkpoint_100.tar"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "SEED=$SEED"
echo "select_aic_margin=$SELECT_AIC_MARGIN"
echo "select_kappa_rho=$SELECT_KAPPA_RHO"
echo "select_kappa_aic=$SELECT_KAPPA_AIC"

if [[ "$SKIP_ENV" != "1" ]]; then
  module load miniforge/24.3.0-0 || true
  module load cuda/12.4.0 || true
  conda deactivate || true
  conda activate "$CONDA_ENV_NAME"
fi

if [[ "$SKIP_TRAIN_IF_DONE" == "1" && -f "$CKPT" ]]; then
  echo "Found existing checkpoint: $CKPT; skipping training."
else
  python main_hard_select.py \
    --dataset CIFAR100 \
    --dataset_dir "$DATA_ROOT" \
    --workers 6 \
    --seed "$SEED" \
    --batch_size 256 \
    --image_size 32 \
    --epochs 100 \
    --resnet resnet18 \
    --projection_dim 64 \
    --optimizer Adam \
    --weight_decay 1.0e-6 \
    --temperature 0.5 \
    --policy tlambda_select \
    --auto_weib_beta True \
    --use_weib_topm True \
    --weib_lambda 0.5 \
    --weib_top_m 50 \
    --use_pl_topm True \
    --pl_lambda 0.5 \
    --pl_top_m 50 \
    --anchor_tail_frac 0.1 \
    --anchor_min_tail_points 32 \
    --anchor_r2_min 0.2 \
    --shrink_c 100.0 \
    --select_q 0.2 \
    --select_aic_margin "$SELECT_AIC_MARGIN" \
    --select_tail_frac 0.1 \
    --select_min_tail_points 32 \
    --select_weib_top_m 16 \
    --select_kappa_rho "$SELECT_KAPPA_RHO" \
    --select_kappa_aic "$SELECT_KAPPA_AIC" \
    --model_path "$RUN_DIR" \
    --checkpoint_every 0
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Expected checkpoint not found: $CKPT"
  exit 1
fi

# Keep eval_seeds=[1] to mirror the uploaded tlambda outputs. Also pin eval fields.
python - <<PY
from pathlib import Path
import yaml
p = Path("$RUN_DIR") / "config.yaml"
cfg = yaml.safe_load(p.read_text()) or {}
cfg["eval_seeds"] = [1]
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

echo "Done WEINCE run: $RUN_DIR"
