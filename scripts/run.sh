#!/bin/bash
# UG-TTT training launcher.
# Usage: scripts/run.sh <env> [extra args...]
#   env: cp | ac1 | ac2 | erdos
# Examples:
#   scripts/run.sh cp --problem_idx 26
#   scripts/run.sh ac1
#   scripts/run.sh erdos --problem_idx 200
# Baseline (no UG-TTT signal):
#   scripts/run.sh cp --problem_idx 26 --num_ensemble_members 1 --rmi_coef 0 --nnm_coef 0

set -euo pipefail

ENV=${1:?usage: $0 <env> [extra args...]}
shift

export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=1
export RAY_ADDRESS=auto
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=mLoRA:${PYTHONPATH:-} python3 -m ug_ttt.rl.mlora_train \
    --env "${ENV}" \
    --base_model Qwen/Qwen3-8B \
    --precision fp16 \
    --num_ensemble_members 5 \
    --adv_estimator entropic_adaptive_beta \
    --rmi_coef 0.1 \
    --nnm_coef 0.075 \
    --uncertainty_metric true_mi \
    --kl_penalty_coef 0.01 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 4e-5 \
    --group_size 8 \
    --groups_per_batch 8 \
    --num_epochs 10 \
    --max_tokens 260000 \
    --temperature 1.0 \
    --sampler_type puct_backprop \
    --initial_exp_type random \
    --two_phase_sampling \
    --phase1_max_tokens 26000 \
    --streaming_mi \
    --streaming_mi_wrap_budget 6000 \
    --streaming_mi_warmup_epochs 3 \
    --streaming_mi_threshold_percentile 5.0 \
    --wandb_project "ug-ttt" \
    --wandb_name "${ENV}-rmi-nnm" \
    --log_path "./logs/${ENV}_rmi_nnm" \
    --save_every 5 \
    "$@"
