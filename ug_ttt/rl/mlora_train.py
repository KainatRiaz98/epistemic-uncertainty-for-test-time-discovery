"""
UG-TTT: Multi-LoRA Ensemble RL Training Loop with Epistemic Uncertainty.

Trains K LoRA adapters jointly over a frozen base model, served by mLoRA.
Token-level mutual information across the ensemble is added to the policy
advantage as an epistemic exploration bonus, and a nuclear-norm regularizer
on the stacked adapter matrices prevents weight-space collapse.

Pipeline per step:
    1. Roll out groups across all K adapters
    2. Score rollouts with each adapter → token-level MI / RMI
    3. Augmented advantage:  A_t^total = A_t^reward + γ · MI_t
    4. Loss:                 L = clipped-IS surrogate + β · ||U_θ||_*
    5. Optimizer step on all K adapters
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# ── mLoRA ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mLoRA"))
from mlora.model import load_model as mlora_load_model
from mlora.model.tokenizer import Tokenizer as MLoRATokenizer

# ── Our new modules ────────────────────────────────────────────────────────────
from ug_ttt.rl.ensemble import LoRAEnsemble
from ug_ttt.rl.nuclear_norm import compute_nuclear_norm_diversity_loss
from ug_ttt.rl.uncertainty import (
    UNCERTAINTY_FNS,
    compute_rmi,
    compute_true_mi,
    compute_top7_phase_breakdown,
    compute_variance,
    compute_predictive_entropy,
)

# ── Task envs / state / sampler ──────────────────────────────────────────────
from ug_ttt.recipes.ttt.state import State
from ug_ttt.recipes.ttt.sampler import StateSampler
from ug_ttt.utils.ml_log import setup_logging, Logger as MLLogger

logger = logging.getLogger(__name__)

# ── Two-phase generation constants (Qwen3) ────────────────────────────────────
# When Phase 1 (thinking) exhausts its token budget, this teacher-forced prefill
# transitions the model from <think> mode to code output — matching the original
# TwoPhaseTokenCompleter in completers.py.
QWEN3_PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
QWEN3_THINK_CLOSE = "</think>\n\n"


def wrap_qwen3_chat_template(prompt_text: str) -> str:
    """Wrap raw prompt text in Qwen3 chat format with thinking enabled.

    Produces:
        <|im_start|>user
        {prompt_text}<|im_end|>
        <|im_start|>assistant
        <think>

    The model then generates thinking tokens after <think>, and eventually
    produces </think> followed by the actual response (code).
    """
    return (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Config — mirrors ug_ttt/rl/train.py:Config with ensemble additions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Training configuration.

    Mirrors the original Config in train.py, replacing Tinker-specific
    fields with mLoRA equivalents and adding ensemble/uncertainty fields.
    """

    # ── Model (replaces model_name + Tinker service) ──────────────────────
    base_model: str = "Qwen/Qwen3-8B"
    precision: str = "fp16"
    device: str = "cuda"

    # ── Environment ───────────────────────────────────────────────────────
    env: str = "ac1"
    problem_idx: str = "improvement"
    budget_s: int = 1000

    # ── RL hyperparameters (matching original CLIConfig defaults) ─────────
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    num_epochs: int = 50
    max_tokens: int = 26000
    temperature: float = 1.0
    adv_estimator: str = "entropic_adaptive_beta"
    adv_estimator_beta: float = 2.0
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    ppo_clip: float = 0.2
    num_substeps: int = 1
    remove_constant_reward_groups: bool = True

    # ── KL penalty (paper uses λ ∈ {0.01, 0.1}) ───────────────────────
    kl_penalty_coef: float = 0.01
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False

    # ── Additional config fields from original ─────────────────────────
    dynamic_max_tokens: bool = True
    two_phase_sampling: bool = False
    phase1_max_tokens: int = 26000
    context_window: int = 32768     # Total context window (Qwen3-8B default)
    context_buffer: int = 50        # Safety margin to stay under context limit
    eval_timeout: int = 1000
    dataset_timeout: int = 1000
    num_cpus_per_task: int = 1

    # ── Ensemble (NEW) ────────────────────────────────────────────────────
    num_ensemble_members: int = 5
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Dict[str, bool] = field(default_factory=lambda: {
        "q_proj": True, "k_proj": True,
        "v_proj": True, "o_proj": True,
    })

    # ── Uncertainty (NEW) ─────────────────────────────────────────────────
    rmi_coef: float = 0.1           # α: base exploration strength (γ_eff adapts via β-coupling)
    uncertainty_metric: str = "true_mi"  # "true_mi", "rmi", "variance", "predictive_entropy"
    gamma_max_ratio: float = 10.0   # clip γ_eff/rmi_coef to prevent explosion

    # ── Nuclear norm diversity regularization ─────────────────────────────
    nnm_coef: float = 0.0        # 0 = off; try 0.05–0.1 when MI collapses
    nnm_use_fro_norm: bool = False  # Frobenius-normalised variant (scale-invariant)

    # ── Streaming MI early stop (Contribution 2) ─────────────────────────
    streaming_mi_enabled: bool = False
    streaming_mi_window_size: int = 4096     # W: tokens in MI scoring window
    streaming_mi_check_interval: int = 2048  # C: decode steps between checks
    streaming_mi_min_gen: int = 4096         # Minimum tokens before first check
    streaming_mi_warmup_epochs: int = 3      # Epochs with full T_max (build MI history)
    streaming_mi_wrap_budget: int = 6000     # Tokens for code output after early stop
    streaming_mi_threshold_percentile: float = 25.0  # Percentile of post-gen MI

    # ── Performance ──────────────────────────────────────────────────────
    kv_quantize: bool = False  # int8 KV cache: halves bandwidth, ~2× faster batched decode

    # ── Sampler ───────────────────────────────────────────────────────────
    sampler_type: str = "puct_backprop"
    initial_exp_type: str = "random"

    # ── Logging (same as original) ────────────────────────────────────────
    log_path: str = "./logs/mlora_train"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # ── Eval / checkpointing (same as original) ──────────────────────────
    eval_every: int = 3
    save_every: int = 5

    # ── Dataset / task wiring ─────────────────────────────────────────────
    dataset_builder: Optional[Callable] = None  # set by cli_main()


# ═══════════════════════════════════════════════════════════════════════════════
# Advantage computation — reuses logic from train.py:compute_advantages()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_advantages(
    rewards: List[float],
    adv_estimator: str,
    adv_estimator_beta: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute advantages from a group of rewards.
    Direct port from ug_ttt/rl/train.py:compute_advantages(),
    adapted to work on a flat reward list instead of TrajectoryGroup.

    Returns:
        (advantages, metadata) where metadata includes "beta" — the solved
        (or fixed) entropic temperature. Used by train_step for γ-coupling.
    """
    rewards_G = torch.tensor(rewards, dtype=torch.float32)
    k = rewards_G.shape[0]

    if adv_estimator == "mean_baseline":
        return rewards_G - rewards_G.mean(), {"beta": 0.0}

    elif adv_estimator == "entropic":
        s_safe = rewards_G - rewards_G.max()
        e = torch.exp(adv_estimator_beta * s_safe)
        Z = (e.sum() - e) / (k - 1) if k > 1 else e
        return e / (Z + 1e-12) - 1.0, {"beta": float(adv_estimator_beta)}

    elif adv_estimator == "entropic_adaptive_beta":
        delta = np.log(2)
        beta_max = 1e6
        r = rewards_G.float()

        if k < 2:
            beta = r.new_tensor(0.0)
        else:
            def kl_hat(b: float) -> float:
                logits = r.new_tensor(b) * (r - r.max())
                logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                q = torch.exp(logq)
                return float((q * (logq + math.log(k))).sum().item())

            lo, hi = 0.0, 1.0
            if kl_hat(hi) < delta:
                while hi < beta_max and kl_hat(hi) < delta:
                    hi *= 2.0
                if kl_hat(hi) < delta:
                    beta = r.new_tensor(hi)
                else:
                    beta = None
            else:
                beta = None

            if beta is None:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if kl_hat(mid) < delta:
                        lo = mid
                    else:
                        hi = mid
                beta = r.new_tensor(hi)

        e = torch.exp(beta * (r - r.max()))
        Z = (e.sum() - e) / (k - 1) if k > 1 else e
        return e / (Z + 1e-12) - 1.0, {"beta": float(beta.item())}

    else:
        raise ValueError(f"Unknown advantage estimator: {adv_estimator}")


# ═══════════════════════════════════════════════════════════════════════════════
# RL Loss — replaces Tinker's loss_fn (importance_sampling / ppo)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rl_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: "Union[float, torch.Tensor]",
    action_mask: torch.Tensor,
    loss_fn: str = "importance_sampling",
    ppo_clip: float = 0.2,
) -> torch.Tensor:
    """
    Compute RL policy gradient loss for a single trajectory.

    Replaces Tinker's forward_backward() + loss_fn mechanism.

    Args:
        advantage: Either a scalar (trajectory-level) or a (T-1,) per-token
            tensor. When incorporate_kl_penalty is active, per-token advantages
            are passed so that each token sees its own KL adjustment — matching
            the original train.py behaviour where Tinker receives per-token
            advantages via datum.loss_fn_inputs["advantages"].
    """
    if isinstance(advantage, torch.Tensor):
        adv = advantage.to(current_logprobs.device)
    else:
        adv = torch.tensor(advantage, device=current_logprobs.device)
    log_ratio = current_logprobs - old_logprobs
    ratio = log_ratio.exp()

    if loss_fn == "ppo":
        clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
        surrogate = torch.min(ratio * adv, clipped * adv)
        masked = -(surrogate * action_mask)
    else:
        masked = -(ratio * adv * action_mask)

    return masked.sum() / action_mask.sum().clamp(min=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming MI — adaptive early stop during generation (Contribution 2)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamingMIConfig:
    """Configuration for streaming MI-based early stop during Phase 1 generation."""
    enabled: bool = False
    window_size: int = 4096        # W: tokens in MI scoring window
    check_interval: int = 2048     # C: decode steps between checks
    min_gen: int = 4096            # Minimum tokens before first check
    warmup_epochs: int = 3         # Epochs using full T_max (no early stop)
    wrap_budget: int = 6000        # Tokens for code output after early stop
    threshold_percentile: float = 25.0  # Percentile of post-gen MI for threshold


class StreamingMITracker:
    """
    Manages adaptive MI threshold across epochs for streaming early stop.

    During warmup_epochs, streaming MI is disabled but post-generation MI
    values are collected. After warmup, the threshold is set to percentile_25
    of all historical post-generation MI values — auto-adapting to task scale,
    model capacity, and training stage.
    """

    def __init__(self, config: StreamingMIConfig):
        self.config = config
        self.mi_history: List[float] = []
        self.current_threshold: float = float('inf')  # No stopping during warmup
        self.total_sequences: int = 0
        self.mi_stopped_count: int = 0

    def is_active(self, epoch: int) -> bool:
        """Whether streaming MI early stop is active for this epoch."""
        return self.config.enabled and epoch >= self.config.warmup_epochs

    def get_threshold(self) -> float:
        return self.current_threshold

    def record_post_generation_mi(self, mi_values: List[float]):
        """Record post-generation MI values from ensemble scoring.

        Called after every group's rollouts are scored (including warmup),
        building the distribution used for threshold computation.
        """
        self.mi_history.extend(mi_values)

    def record_streaming_result(self, mi_stopped: bool):
        """Record whether a sequence was MI-stopped."""
        self.total_sequences += 1
        if mi_stopped:
            self.mi_stopped_count += 1

    def update_threshold(self, epoch: int):
        """Recompute threshold at end of epoch from all historical MI values."""
        if not self.mi_history:
            return
        self.current_threshold = float(np.percentile(
            self.mi_history, self.config.threshold_percentile
        ))
        logger.info(
            f"StreamingMI: epoch {epoch}, threshold={self.current_threshold:.6f} "
            f"(p{self.config.threshold_percentile:.0f} of {len(self.mi_history)} samples), "
            f"stopped {self.mi_stopped_count}/{self.total_sequences} sequences"
        )

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics for wandb logging."""
        metrics: Dict[str, float] = {
            "streaming_mi/threshold": self.current_threshold
                if self.current_threshold != float('inf') else 0.0,
            "streaming_mi/history_size": float(len(self.mi_history)),
        }
        if self.total_sequences > 0:
            metrics["streaming_mi/stop_rate"] = (
                self.mi_stopped_count / self.total_sequences
            )
        if self.mi_history:
            recent = self.mi_history[-1000:]
            metrics["streaming_mi/mi_mean"] = float(np.mean(recent))
            metrics["streaming_mi/mi_std"] = float(np.std(recent))
        return metrics

    def reset_epoch_counters(self):
        """Reset per-epoch sequence counters (not MI history)."""
        self.total_sequences = 0
        self.mi_stopped_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# KL penalty — port of metrics.py:incorporate_kl_penalty()
# ═══════════════════════════════════════════════════════════════════════════════

def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Discounted sum of future values for each position.
    Port of metrics.py:discounted_future_sum_vectorized().
    """
    import scipy.signal
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)


def incorporate_kl_penalty(
    rollouts: List["Rollout"],
    advantages: torch.Tensor,
    ensemble: "LoRAEnsemble",
    kl_penalty_coef: float,
    kl_discount_factor: float,
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Adjust advantages with per-token KL penalty against the base model.

    Matches metrics.py:incorporate_kl_penalty() exactly: returns per-token
    advantage tensors so that each action token sees its own KL adjustment.

    For each rollout:
        1. Compute base model logprobs (no LoRA) for the generated sequence
        2. kl_diff_t = sampling_logprobs_t - base_logprobs_t
        3. avg_kl_diff = mean(kl_diff) across all rollouts (action tokens only)
        4. kl_advantage_t = kl_coef * mask_t * (avg_kl_diff - kl_diff_t)
        5. Optionally apply temporal discounting
        6. per_token_advantage_t = broadcast(scalar_advantage) + kl_advantage_t

    Args:
        rollouts: List of Rollout objects with sampling_logprobs.
        advantages: (N,) tensor of per-rollout scalar advantages.
        ensemble: LoRAEnsemble (used to get base model for logprob computation).
        kl_penalty_coef: Coefficient for KL penalty.
        kl_discount_factor: Temporal discount factor (0 = disabled).
        device: Device for tensors.

    Returns:
        per_token_advantages: List of (T_i-1,) tensors, one per rollout.
            Each token's advantage = scalar_advantage + per-token KL adjustment.
            Matches how train.py's assemble_training_data broadcasts the scalar
            advantage to all action tokens, then incorporate_kl_penalty adds
            per-token KL adjustments in-place.
        metrics: Dict with 'kl_policy_base' metric.
    """
    # Compute per-token KL diffs for each rollout
    logprob_diffs = []
    float_masks = []

    for rollout in rollouts:
        base_logprobs = _compute_base_logprobs(ensemble, rollout.full_tokens, device)

        sampling_lp = torch.tensor(rollout.sampling_logprobs, device=device)
        mask = rollout.action_mask.to(device)

        # kl_diff = sampling_logprobs - base_logprobs, masked
        diff = (sampling_lp - base_logprobs) * mask
        logprob_diffs.append(diff)
        float_masks.append(mask)

    # Average KL diff across all action tokens in the batch
    total_diff = sum(d.sum() for d in logprob_diffs)
    total_mask = sum(m.sum() for m in float_masks)
    avg_logp_diff = total_diff / total_mask.clamp(min=1)

    # Build per-token advantages: broadcast scalar advantage + per-token KL
    # This matches the original: assemble_training_data broadcasts the scalar
    # advantage to all action tokens, then incorporate_kl_penalty modifies in-place.
    per_token_advantages = []
    for i, rollout in enumerate(rollouts):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.cpu().numpy(), kl_discount_factor),
                device=device,
            )
        # Broadcast scalar advantage to all tokens, add per-token KL adjustment
        token_adv = advantages[i] * float_masks[i] + kl_advantages
        per_token_advantages.append(token_adv)

    return per_token_advantages, {"kl_policy_base": float(avg_logp_diff)}


@torch.no_grad()
def _compute_base_logprobs(
    ensemble: "LoRAEnsemble",
    token_ids: List[int],
    device: str,
) -> torch.Tensor:
    """
    Compute logprobs from the base model (no LoRA adapters).

    This gives us p_base(y_t | y_{<t}) for KL penalty computation.
    We temporarily disable all LoRA adapters and do a forward pass.

    Returns:
        (T-1,) tensor of base model log-probabilities.
    """
    T = len(token_ids)

    # Build data config with no adapter (base model only)
    from mlora.model.args import MLoRAData, MLoRADataConfig, Tokens, Masks

    data_config = MLoRADataConfig(
        adapter_name="__base__",
        adapter_type="",
        start_idx=0,
        end_idx=1,
        expand_fn=ensemble._expand_fn,
        loss_fn=ensemble._noop_loss,
        task_name="base_scoring",
    )

    mlora_data = MLoRAData(
        batch_tokens=[list(token_ids)],
        batch_mask=[[True] * T],
        data_config=[data_config],
    )
    mlora_data.use_flash_causal_ = True

    logits = ensemble.model.forward(mlora_data.model_data())  # (1, T, V)

    # Chunk over the sequence dim to avoid materialising the full (T-1, V) log-prob tensor.
    # For Qwen3-8B: V=152064, T=26K → ~8 GB for log_softmax output if done at once.
    # Chunking keeps peak at (1, T, V) logits + (chunk, V) log_softmax ≈ 8.1 GB instead of ~16 GB.
    # logits[0, start:end, :] is a view (no copy), so only the chunk_lp allocation is new each step.
    dev = logits.device
    target = torch.tensor(token_ids[1:], dtype=torch.long, device=dev)
    chunk_size = 256
    per_token_chunks = []
    for start in range(0, T - 1, chunk_size):
        end = min(start + chunk_size, T - 1)
        chunk_lp = F.log_softmax(logits[0, start:end, :], dim=-1)  # (chunk, V)
        per_token_chunks.append(chunk_lp.gather(1, target[start:end].unsqueeze(1)).squeeze(1))
        del chunk_lp
    del logits
    return torch.cat(per_token_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# KL tracking — port of metrics.py:compute_kl_sample_train()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kl_sample_train(
    rollouts: List["Rollout"],
    training_logprobs: List[torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute KL divergence between sampling and training logprobs.

    Port of metrics.py:compute_kl_sample_train(). This is the primary
    signal for detecting training instability — if KL is large, the
    policy has shifted significantly from the sampling policy.

    Args:
        rollouts: List of Rollout objects with sampling_logprobs.
        training_logprobs: List of (T-1,) tensors from the training forward pass.
        device: Device for tensors.

    Returns:
        Dict with 'optim/kl_sample_train_v1', 'optim/kl_sample_train_v2',
        'optim/entropy' metrics.
    """
    all_diffs: List[torch.Tensor] = []
    all_sampling_logprobs: List[torch.Tensor] = []

    for rollout, train_lp in zip(rollouts, training_logprobs):
        sampling_lp = torch.tensor(rollout.sampling_logprobs, device=device)
        mask = rollout.action_mask.to(device) > 0

        sampling_actions = sampling_lp[mask]
        training_actions = train_lp[mask]

        if len(sampling_actions) > 0:
            logprob_diff = sampling_actions - training_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_actions)

    if not all_diffs:
        return {
            "optim/kl_sample_train_v1": 0.0,
            "optim/kl_sample_train_v2": 0.0,
            "optim/entropy": 0.0,
        }

    flat_diffs = torch.cat(all_diffs)
    flat_sampling = torch.cat(all_sampling_logprobs)

    return {
        "optim/kl_sample_train_v1": flat_diffs.mean().item(),
        "optim/kl_sample_train_v2": 0.5 * (flat_diffs ** 2).mean().item(),
        "optim/entropy": -flat_sampling.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rollout:
    """Single trajectory result. Analogous to original Trajectory + final reward."""
    prompt_tokens: List[int]
    full_tokens: List[int]           # prompt + generated
    sampling_logprobs: List[float]   # logprobs from sampling adapter
    reward_exec: float               # R_exec from task verifier
    reward_rmi: float = 0.0          # uncertainty bonus
    reward_total: float = 0.0        # R_exec + γ * RMI
    adapter_idx: int = 0             # which member generated this
    state: Optional[Any] = None      # parent state
    child_state: Optional[Any] = None  # child state for sampler update
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Two-phase masking: when set, overrides default action_mask.
    # Entries: 0.0 for prompt and teacher-forced prefill, 1.0 for trained tokens.
    _custom_mask: Optional[List[float]] = None
    prefill_info: Optional[Dict[str, int]] = None  # {"start": idx, "len": n}

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def action_mask(self) -> torch.Tensor:
        """Per-token mask. Length = len(full_tokens) - 1.

        Default: 0 for prompt, 1 for generated.
        Two-phase: 0 for prompt + teacher-forced prefill, 1 for thinking + code.
        """
        if self._custom_mask is not None:
            return torch.tensor(self._custom_mask)
        total = len(self.full_tokens) - 1
        mask = torch.zeros(total)
        mask[self.prompt_len - 1:] = 1.0
        return mask



@dataclass
class RolloutGroup:
    """Group of rollouts from the same parent state. Analogous to TrajectoryGroup."""
    rollouts: List[Rollout]
    state: Optional[Any] = None

    def get_total_rewards(self) -> List[float]:
        """Matches TrajectoryGroup.get_total_rewards() interface."""
        return [r.reward_total for r in self.rollouts]

    def get_exec_rewards(self) -> List[float]:
        return [r.reward_exec for r in self.rollouts]


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop — mirrors train.py:do_sync_training() structure
# ═══════════════════════════════════════════════════════════════════════════════

def do_rollout(
    ensemble: LoRAEnsemble,
    tokenizer: MLoRATokenizer,
    prompt_tokens: List[int],
    adapter_idx: int,
    parent_state: Optional[Any],
    step: int,
    process_fn: Callable[[str, Any, int], Dict[str, Any]],
    uncertainty_fn: Optional[Callable],
    rmi_coef: float,
    max_tokens: int,
    temperature: float,
    two_phase_sampling: bool = False,
    phase1_max_tokens: int = 26000,
    context_window: int = 32768,
    context_buffer: int = 50,
    prefill_tokens: Optional[List[int]] = None,
    stop_token_ids: Optional[List[int]] = None,
    kv_quantize: bool = False,
) -> Rollout:
    """
    Generate a single rollout, compute R_exec and RMI.

    Args:
        process_fn: (generated_text, parent_state, step) → dict with keys:
            reward (float), child_state (Optional[State]), correctness (float), metrics (dict)
        two_phase_sampling: If True, use two-phase generation (thinking + code).
        prefill_tokens: Teacher-forced transition tokens (required if two_phase_sampling).
        stop_token_ids: Additional stop tokens for Phase 1 (e.g. [im_end_id]).
    """
    custom_mask = None
    prefill_info = None

    # ── 1. Generate (one adapter) ─────────────────────────────────────────
    if two_phase_sampling and prefill_tokens is not None:
        full_tokens, sampling_logprobs, custom_mask = ensemble.generate_two_phase(
            prompt_tokens=prompt_tokens,
            phase1_max_tokens=phase1_max_tokens,
            context_window=context_window,
            context_buffer=context_buffer,
            prefill_tokens=prefill_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_id_ or 2,
            stop_token_ids=stop_token_ids,
            adapter_idx=adapter_idx,
            kv_quantize=kv_quantize,
        )
    else:
        full_tokens, sampling_logprobs = ensemble.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_id_,
            adapter_idx=adapter_idx,
            kv_quantize=kv_quantize,
        )

    # ── 2. Execute → R_exec (via task verifier) ──────────────────────────
    generated_text = tokenizer.decode(full_tokens[len(prompt_tokens):])
    result = process_fn(generated_text, parent_state, step)
    reward_exec = result["reward"]
    child_state = result.get("child_state")

    # ── 3. Ensemble scoring → uncertainty (all K adapters, single forward pass) ──
    ensemble_logprobs, per_token_true_mi = ensemble.compute_ensemble_logprobs(full_tokens)

    # Primary uncertainty metric for reward_rmi
    if uncertainty_fn is None:
        # true_mi mode: top-7% of tokens (Wang et al. 2025 — high-entropy minority drives RL)
        reward_rmi = float(compute_true_mi(per_token_true_mi))
    else:
        # logprob-based mode (ablation): compute from point estimates
        reward_rmi = float(uncertainty_fn(ensemble_logprobs))

    # Compute ALL uncertainty metrics for ablation comparison logging
    true_mi_val = float(compute_true_mi(per_token_true_mi))
    rmi_val = float(compute_rmi(ensemble_logprobs))
    variance_val = float(compute_variance(ensemble_logprobs))
    pred_entropy_val = float(compute_predictive_entropy(ensemble_logprobs))
    # Top-7% versions: same aggregation as true_mi → isolates metric quality from aggregation
    rmi_top7_val = float(compute_rmi(ensemble_logprobs, top_frac=0.07))
    variance_top7_val = float(compute_variance(ensemble_logprobs, top_frac=0.07))
    pred_entropy_top7_val = float(compute_predictive_entropy(ensemble_logprobs, top_frac=0.07))

    # Ensemble member disagreement stats — proves K adapters actually diverge
    # per_member_mean_lp[k] = mean logprob assigned by member k to the sequence
    per_member_mean_lp = ensemble_logprobs.mean(dim=1)  # (K,)
    member_logprob_spread = float(per_member_mean_lp.std())  # σ across members
    member_logprob_range = float(per_member_mean_lp.max() - per_member_mean_lp.min())

    # Per-token disagreement: how much do members disagree at each position?
    per_token_var = ensemble_logprobs.var(dim=0)  # (T-1,)
    max_token_disagreement = float(per_token_var.max())
    frac_high_disagreement = float((per_token_var > per_token_var.median()).float().mean())

    # ── 4. Augmented reward ──────────────────────────────────────────────
    reward_total = reward_exec + rmi_coef * reward_rmi

    # Pad sampling_logprobs to full-sequence length (T-1) with 0.0 for
    # prompt positions — matches original train.py's trajectory_to_data()
    # which does: [0.0] * ob_len + action_logprobs
    prompt_pad = [0.0] * (len(prompt_tokens) - 1)
    full_sampling_logprobs = prompt_pad + list(sampling_logprobs)

    # Two-phase metrics
    prefill_injected = custom_mask is not None
    gen_total = len(full_tokens) - len(prompt_tokens)
    if prefill_injected and prefill_tokens is not None:
        prefill_len = len(prefill_tokens)
        # Count Phase 1 tokens: from custom_mask, they are 1.0 entries after prompt pad
        phase1_count = sum(1 for v in custom_mask[len(prompt_tokens)-1:] if v == 1.0)
        # Subtract phase2 tokens to get phase1 only
        phase2_count = gen_total - phase1_count - prefill_len  # approximate
        # More precise: phase1 is gen_total - prefill - phase2
        # Phase 1 = tokens before prefill, Phase 2 = tokens after prefill
        # From mask structure: prompt_pad(0) | phase1(1) | prefill(0) | phase2(1)
        p1_ones = 0
        for v in custom_mask[len(prompt_tokens)-1:]:
            if v == 1.0:
                p1_ones += 1
            else:
                break
        phase1_tokens_count = p1_ones
        phase2_tokens_count = gen_total - phase1_tokens_count - prefill_len
        prefill_info = {"start": len(prompt_tokens) + phase1_tokens_count, "len": prefill_len}
    else:
        phase1_tokens_count = gen_total
        phase2_tokens_count = 0

    # ── Top-7% phase breakdown ─────────────────────────────────────────────
    _prefill_count = len(prefill_tokens) if (prefill_injected and prefill_tokens is not None) else 0
    phase_breakdown = compute_top7_phase_breakdown(
        per_token_true_mi,
        prompt_len=len(prompt_tokens),
        phase1_count=phase1_tokens_count,
        prefill_count=_prefill_count,
    )

    return Rollout(
        prompt_tokens=prompt_tokens,
        full_tokens=full_tokens,
        sampling_logprobs=full_sampling_logprobs,
        reward_exec=reward_exec,
        reward_rmi=reward_rmi,
        reward_total=reward_total,
        adapter_idx=adapter_idx,
        child_state=child_state,
        metrics={
            "reward/exec": reward_exec,
            "reward/rmi": reward_rmi,
            "reward/total": reward_total,
            "gen/num_tokens": gen_total,
            "gen/adapter_idx": adapter_idx,
            "gen/prefill_injected": 1.0 if prefill_injected else 0.0,
            "gen/phase1_tokens": phase1_tokens_count,
            "gen/phase2_tokens": phase2_tokens_count,
            "correctness": result.get("correctness", 0.0),
            # ── Uncertainty decomposition (ablation study) ──
            "uncertainty/true_mi": true_mi_val,
            "uncertainty/rmi": rmi_val,
            "uncertainty/variance": variance_val,
            "uncertainty/predictive_entropy": pred_entropy_val,
            # Top-7% baselines — same aggregation as true_mi for fair comparison
            "uncertainty/rmi_top7": rmi_top7_val,
            "uncertainty/variance_top7": variance_top7_val,
            "uncertainty/predictive_entropy_top7": pred_entropy_top7_val,
            # ── Top-7% phase breakdown (thinking vs code) ──
            "uncertainty/top7_pct_thinking": phase_breakdown["top7_pct_thinking"],
            "uncertainty/top7_pct_code": phase_breakdown["top7_pct_code"],
            "uncertainty/top7_mi_thinking": phase_breakdown["top7_mi_thinking"],
            "uncertainty/top7_mi_code": phase_breakdown["top7_mi_code"],
            # ── Ensemble disagreement (proves members diverge) ──
            "ensemble/member_logprob_spread": member_logprob_spread,
            "ensemble/member_logprob_range": member_logprob_range,
            "ensemble/max_token_disagreement": max_token_disagreement,
            "ensemble/frac_high_disagreement": frac_high_disagreement,
        },
        _custom_mask=custom_mask,
        prefill_info=prefill_info if prefill_injected else None,
    )


def _build_rollout_from_scored(
    prompt_tokens: List[int],
    full_tokens: List[int],
    sampling_logprobs: List[float],
    ensemble_logprobs: torch.Tensor,
    per_token_true_mi: torch.Tensor,
    result: Dict[str, Any],
    adapter_idx: int,
    uncertainty_fn: Optional[Callable],
    rmi_coef: float,
    custom_mask: Optional[List[float]] = None,
    prefill_token_count: int = 0,
    mi_meta: Optional[Dict[str, Any]] = None,
) -> Rollout:
    """Build a Rollout from pre-generated tokens and pre-scored ensemble logprobs."""
    reward_exec = result["reward"]
    child_state = result.get("child_state")

    if uncertainty_fn is not None:
        reward_rmi = float(uncertainty_fn(ensemble_logprobs))
    else:
        # top-7% of tokens (Wang et al. 2025 — high-entropy minority drives RL)
        reward_rmi = float(compute_true_mi(per_token_true_mi))

    true_mi_val = float(compute_true_mi(per_token_true_mi))
    rmi_val = float(compute_rmi(ensemble_logprobs))
    variance_val = float(compute_variance(ensemble_logprobs))
    pred_entropy_val = float(compute_predictive_entropy(ensemble_logprobs))
    # Top-7% versions: same aggregation as true_mi → isolates metric quality from aggregation
    rmi_top7_val = float(compute_rmi(ensemble_logprobs, top_frac=0.07))
    variance_top7_val = float(compute_variance(ensemble_logprobs, top_frac=0.07))
    pred_entropy_top7_val = float(compute_predictive_entropy(ensemble_logprobs, top_frac=0.07))

    per_member_mean_lp = ensemble_logprobs.mean(dim=1)
    member_logprob_spread = float(per_member_mean_lp.std())
    member_logprob_range = float(per_member_mean_lp.max() - per_member_mean_lp.min())

    per_token_var = ensemble_logprobs.var(dim=0)
    max_token_disagreement = float(per_token_var.max())
    frac_high_disagreement = float((per_token_var > per_token_var.median()).float().mean())

    reward_total = reward_exec + rmi_coef * reward_rmi

    prompt_pad = [0.0] * (len(prompt_tokens) - 1)
    full_sampling_logprobs = prompt_pad + list(sampling_logprobs)

    # Two-phase metrics
    prefill_injected = custom_mask is not None
    gen_total = len(full_tokens) - len(prompt_tokens)
    if prefill_injected and prefill_token_count > 0:
        # Count Phase 1 tokens from mask: 1.0 entries after prompt pad, before first 0.0 block
        p1_ones = 0
        for v in custom_mask[len(prompt_tokens)-1:]:
            if v == 1.0:
                p1_ones += 1
            else:
                break
        phase1_tokens_count = p1_ones
        phase2_tokens_count = gen_total - phase1_tokens_count - prefill_token_count
        prefill_info = {"start": len(prompt_tokens) + phase1_tokens_count, "len": prefill_token_count}
    else:
        phase1_tokens_count = gen_total
        phase2_tokens_count = 0
        prefill_info = None

    # ── Top-7% phase breakdown ─────────────────────────────────────────────
    phase_breakdown = compute_top7_phase_breakdown(
        per_token_true_mi,
        prompt_len=len(prompt_tokens),
        phase1_count=phase1_tokens_count,
        prefill_count=prefill_token_count if prefill_injected else 0,
    )

    return Rollout(
        prompt_tokens=prompt_tokens,
        full_tokens=full_tokens,
        sampling_logprobs=full_sampling_logprobs,
        reward_exec=reward_exec,
        reward_rmi=reward_rmi,
        reward_total=reward_total,
        adapter_idx=adapter_idx,
        child_state=child_state,
        metrics={
            "reward/exec": reward_exec,
            "reward/rmi": reward_rmi,
            "reward/total": reward_total,
            "gen/num_tokens": gen_total,
            "gen/adapter_idx": adapter_idx,
            "gen/prefill_injected": 1.0 if prefill_injected else 0.0,
            "gen/phase1_tokens": phase1_tokens_count,
            "gen/phase2_tokens": phase2_tokens_count,
            "correctness": result.get("correctness", 0.0),
            "uncertainty/true_mi": true_mi_val,
            "uncertainty/rmi": rmi_val,
            "uncertainty/variance": variance_val,
            "uncertainty/predictive_entropy": pred_entropy_val,
            # Top-7% baselines — same aggregation as true_mi for fair comparison
            "uncertainty/rmi_top7": rmi_top7_val,
            "uncertainty/variance_top7": variance_top7_val,
            "uncertainty/predictive_entropy_top7": pred_entropy_top7_val,
            # ── Top-7% phase breakdown (thinking vs code) ──
            "uncertainty/top7_pct_thinking": phase_breakdown["top7_pct_thinking"],
            "uncertainty/top7_pct_code": phase_breakdown["top7_pct_code"],
            "uncertainty/top7_mi_thinking": phase_breakdown["top7_mi_thinking"],
            "uncertainty/top7_mi_code": phase_breakdown["top7_mi_code"],
            "ensemble/member_logprob_spread": member_logprob_spread,
            "ensemble/member_logprob_range": member_logprob_range,
            "ensemble/max_token_disagreement": max_token_disagreement,
            "ensemble/frac_high_disagreement": frac_high_disagreement,
            # ── Streaming MI early stop metrics ──
            "streaming_mi/stopped": 1.0 if (mi_meta and mi_meta.get("mi_stopped")) else 0.0,
            "streaming_mi/stop_step": mi_meta["mi_stop_step"] if mi_meta else -1,
            "streaming_mi/num_checks": len(mi_meta.get("mi_values", [])) if mi_meta else 0,
            "streaming_mi/tokens_saved": (
                max(0, gen_total - (mi_meta["mi_stop_step"] if mi_meta and mi_meta.get("mi_stopped") else gen_total))
            ),
        },
        _custom_mask=custom_mask,
        prefill_info=prefill_info,
    )


def do_group_rollout_batched(
    ensemble: LoRAEnsemble,
    tokenizer: MLoRATokenizer,
    prompt_tokens: List[int],
    group_size: int,
    parent_state: Optional[Any],
    epoch: int,
    group_idx: int,
    process_fn: Callable[[str, Any, int], Dict[str, Any]],
    uncertainty_fn: Optional[Callable],
    rmi_coef: float,
    max_tokens: int,
    temperature: float,
    reward_workers: concurrent.futures.ThreadPoolExecutor,
    kv_quantize: bool = False,
    two_phase_sampling: bool = False,
    phase1_max_tokens: int = 26000,
    context_window: int = 32768,
    context_buffer: int = 50,
    prefill_tokens: Optional[List[int]] = None,
    stop_token_ids: Optional[List[int]] = None,
    mi_tracker: Optional[StreamingMITracker] = None,
    streaming_mi_cfg: Optional[Config] = None,
) -> List[Rollout]:
    """
    Generate all rollouts for a group using batched generation + parallel
    reward computation + batched ensemble scoring.

    3-layer parallelism:
        1. GPU: generate_batch() — all group_size sequences in parallel
        2. CPU: ThreadPoolExecutor — reward computation in parallel
        3. GPU: compute_ensemble_logprobs_batch() — score all at once

    Args:
        reward_workers: Thread pool for parallel CPU reward computation.
        two_phase_sampling: If True, use two-phase generation (thinking + code).
        prefill_tokens: Teacher-forced transition tokens (required if two_phase_sampling).
        stop_token_ids: Additional stop tokens for Phase 1.

    Returns:
        List of Rollout objects for this group.
    """
    K = ensemble.K

    # ── 1. Multi-adapter batched generation (GPU) ────────────────────────
    adapter_assignments = [(epoch * group_size + i) % K for i in range(group_size)]
    t_gen_start = time.time()

    # Determine if streaming MI is active for this epoch.
    # Gated only by mi_tracker (non-None iff streaming_mi_enabled=True in cfg).
    # Intentionally NOT gated on rmi_coef — Condition C of the ablation uses
    # rmi_coef=0 with streaming MI enabled (dynamic tokens, no exploration bonus).
    mi_active = (mi_tracker is not None
                 and mi_tracker.is_active(epoch))

    if two_phase_sampling and prefill_tokens is not None:
        gen_results_4 = ensemble.generate_batch_multi_adapter_two_phase(
            prompt_tokens=prompt_tokens,
            adapter_assignments=adapter_assignments,
            phase1_max_tokens=phase1_max_tokens,
            context_window=context_window,
            context_buffer=context_buffer,
            prefill_tokens=prefill_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_id_ or 2,
            stop_token_ids=stop_token_ids,
            kv_quantize=kv_quantize,
            streaming_mi_enabled=mi_active,
            mi_window_size=streaming_mi_cfg.streaming_mi_window_size if streaming_mi_cfg else 4096,
            mi_check_interval=streaming_mi_cfg.streaming_mi_check_interval if streaming_mi_cfg else 2048,
            mi_min_gen=streaming_mi_cfg.streaming_mi_min_gen if streaming_mi_cfg else 4096,
            mi_threshold=mi_tracker.get_threshold() if mi_tracker else 0.0,
            mi_wrap_budget=streaming_mi_cfg.streaming_mi_wrap_budget if streaming_mi_cfg else 6000,
        )
        # Unpack 4-tuples into gen_results (tokens, logprobs) + per-seq masks + MI metadata
        gen_results = [(ft, lp) for ft, lp, _, _ in gen_results_4]
        custom_masks = [mask for _, _, mask, _ in gen_results_4]
        mi_metas = [meta for _, _, _, meta in gen_results_4]
    else:
        gen_results = ensemble.generate_batch_multi_adapter(
            prompt_tokens=prompt_tokens,
            adapter_assignments=adapter_assignments,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_id_,
            kv_quantize=kv_quantize,
        )
        custom_masks = [None] * len(gen_results)
        mi_metas = [None] * len(gen_results)

    t_gen = time.time() - t_gen_start
    logger.info(f"  Group {group_idx}: multi-adapter generation {group_size} seqs in {t_gen:.1f}s")

    # ── 2. Parallel reward computation (CPU) ─────────────────────────────
    t_reward_start = time.time()
    reward_futures = []
    for i, (full_tokens, sampling_logprobs) in enumerate(gen_results):
        step_idx = epoch * 1000 + group_idx * group_size + i
        generated_text = tokenizer.decode(full_tokens[len(prompt_tokens):])
        future = reward_workers.submit(process_fn, generated_text, parent_state, step_idx)
        reward_futures.append(future)

    # ── 3. Batched ensemble scoring (GPU, overlaps with CPU rewards) ─────
    all_full_tokens = [ft for ft, _ in gen_results]
    if rmi_coef > 0:
        t_score_start = time.time()
        torch.cuda.empty_cache()
        all_scored = ensemble.compute_ensemble_logprobs_batch(all_full_tokens)
        t_score = time.time() - t_score_start
        logger.info(f"  Group {group_idx}: batched ensemble scoring in {t_score:.1f}s")
    else:
        # Baseline mode: skip ensemble scoring — rmi_coef=0 so uncertainty
        # has no effect on reward or training; use zero tensors for logging.
        all_scored = [
            (torch.zeros(ensemble.K, len(ft) - 1), torch.zeros(len(ft) - 1))
            for ft in all_full_tokens
        ]

    # ── 4. Collect rewards and build Rollout objects ─────────────────────
    reward_results = [f.result() for f in reward_futures]
    t_reward = time.time() - t_reward_start
    logger.info(f"  Group {group_idx}: rewards computed in {t_reward:.1f}s")

    # Feed post-generation MI to tracker for threshold adaptation (every epoch)
    if mi_tracker is not None:
        post_gen_mis = [float(compute_true_mi(mi)) for _, mi in all_scored]
        mi_tracker.record_post_generation_mi(post_gen_mis)

    prefill_len = len(prefill_tokens) if prefill_tokens else 0
    rollouts = []
    for i, (full_tokens, sampling_logprobs) in enumerate(gen_results):
        ensemble_logprobs_i, per_token_mi_i = all_scored[i]
        rollout = _build_rollout_from_scored(
            prompt_tokens=prompt_tokens,
            full_tokens=full_tokens,
            sampling_logprobs=sampling_logprobs,
            ensemble_logprobs=ensemble_logprobs_i,
            per_token_true_mi=per_token_mi_i,
            result=reward_results[i],
            adapter_idx=adapter_assignments[i],
            uncertainty_fn=uncertainty_fn,
            rmi_coef=rmi_coef,
            custom_mask=custom_masks[i],
            prefill_token_count=prefill_len,
            mi_meta=mi_metas[i],
        )
        rollout.state = parent_state
        rollouts.append(rollout)

        # Record streaming MI result in tracker
        if mi_tracker is not None and mi_metas[i] is not None:
            mi_tracker.record_streaming_result(mi_metas[i].get("mi_stopped", False))

        # Per-rollout logging
        n_gen = len(full_tokens) - len(prompt_tokens)
        correct = rollout.metrics.get("correctness", 0.0)
        phase_info = ""
        if rollout.prefill_info:
            phase_info = (
                f" p1={rollout.metrics.get('gen/phase1_tokens', 0)}"
                f" p2={rollout.metrics.get('gen/phase2_tokens', 0)}"
            )
        true_mi_log = rollout.metrics.get("uncertainty/true_mi", 0.0)
        logger.info(
            f"    [g{group_idx}][r{i+1}/{group_size}] adapter={rollout.adapter_idx} "
            f"R_exec={rollout.reward_exec:.4f} true_MI={true_mi_log:.5f} RMI={rollout.reward_rmi:.4f} "
            f"R_total={rollout.reward_total:.4f} correct={correct:.0f} "
            f"tokens={n_gen}{phase_info}"
        )

    # Group-level normalized MI summary (mean and std across all rollouts in this group)
    group_true_mis = [r.metrics.get("uncertainty/true_mi", 0.0) for r in rollouts]
    if group_true_mis:
        import statistics
        g_mean = sum(group_true_mis) / len(group_true_mis)
        g_std = statistics.stdev(group_true_mis) if len(group_true_mis) > 1 else 0.0
        g_max = max(group_true_mis)
        # Normalized: each rollout's MI relative to group mean (avoids scale dependence)
        norm_mis = [v / g_mean if g_mean > 1e-9 else 0.0 for v in group_true_mis]
        norm_str = " ".join(f"{v:.2f}" for v in norm_mis)
        logger.info(
            f"  [g{group_idx}] group true_MI: mean={g_mean:.5f} std={g_std:.5f} max={g_max:.5f} "
            f"normalized=[{norm_str}]"
        )

    return rollouts


def _split_list(lst: list, n: int) -> List[list]:
    """Split list into n roughly equal chunks. Port of train.py:split_list()."""
    if n <= 0:
        return []
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _write_rollout_logs(
    log_path: str,
    rollouts: List["Rollout"],
    tokenizer,
    epoch: int,
    group_idx: int,
    prompt_text: str,
) -> None:
    """Append per-rollout data to trajectories.jsonl for offline analysis.

    Mirrors train.py's _write_trajectory_logs() but for Rollout objects,
    extended with all mlora-specific uncertainty/ensemble metrics.
    """
    os.makedirs(log_path, exist_ok=True)
    out_path = os.path.join(log_path, "trajectories.jsonl")
    with open(out_path, "a") as f:
        for i, rollout in enumerate(rollouts):
            gen_tokens = rollout.full_tokens[rollout.prompt_len:]
            generated_text = tokenizer.decode(gen_tokens)
            payload = {
                # Identification
                "epoch": epoch,
                "group_idx": group_idx,
                "rollout_idx": i,
                "adapter_idx": rollout.adapter_idx,
                # Text
                "prompt_text": prompt_text,
                "generated_text": generated_text,
                "num_tokens": len(gen_tokens),
                # Rewards
                "reward_exec": rollout.reward_exec,
                "reward_rmi": rollout.reward_rmi,
                "reward_total": rollout.reward_total,
                # Task
                "correctness": rollout.metrics.get("correctness", 0.0),
                # Generation (two-phase)
                "prefill_injected": rollout.metrics.get("gen/prefill_injected", 0.0),
                "phase1_tokens": rollout.metrics.get("gen/phase1_tokens", 0),
                "phase2_tokens": rollout.metrics.get("gen/phase2_tokens", 0),
                # Uncertainty decomposition — all 4 for ablation (mean and top-7%)
                "uncertainty_true_mi": rollout.metrics.get("uncertainty/true_mi", 0.0),
                "uncertainty_rmi": rollout.metrics.get("uncertainty/rmi", 0.0),
                "uncertainty_variance": rollout.metrics.get("uncertainty/variance", 0.0),
                "uncertainty_predictive_entropy": rollout.metrics.get("uncertainty/predictive_entropy", 0.0),
                "uncertainty_rmi_top7": rollout.metrics.get("uncertainty/rmi_top7", 0.0),
                "uncertainty_variance_top7": rollout.metrics.get("uncertainty/variance_top7", 0.0),
                "uncertainty_predictive_entropy_top7": rollout.metrics.get("uncertainty/predictive_entropy_top7", 0.0),
                # Top-7% phase breakdown (thinking vs code)
                "top7_pct_thinking": rollout.metrics.get("uncertainty/top7_pct_thinking", 0.0),
                "top7_pct_code": rollout.metrics.get("uncertainty/top7_pct_code", 0.0),
                "top7_mi_thinking": rollout.metrics.get("uncertainty/top7_mi_thinking", 0.0),
                "top7_mi_code": rollout.metrics.get("uncertainty/top7_mi_code", 0.0),
                # Ensemble disagreement
                "ensemble_member_logprob_spread": rollout.metrics.get("ensemble/member_logprob_spread", 0.0),
                "ensemble_member_logprob_range": rollout.metrics.get("ensemble/member_logprob_range", 0.0),
                "ensemble_max_token_disagreement": rollout.metrics.get("ensemble/max_token_disagreement", 0.0),
                "ensemble_frac_high_disagreement": rollout.metrics.get("ensemble/frac_high_disagreement", 0.0),
                # Streaming MI early stop
                "streaming_mi_stopped": rollout.metrics.get("streaming_mi/stopped", 0.0),
                "streaming_mi_stop_step": rollout.metrics.get("streaming_mi/stop_step", -1),
                "streaming_mi_num_checks": rollout.metrics.get("streaming_mi/num_checks", 0),
                "streaming_mi_tokens_saved": rollout.metrics.get("streaming_mi/tokens_saved", 0),
            }
            f.write(json.dumps(payload) + "\n")


def _collect_rollout_metrics(rollout: "Rollout", accum: Dict[str, List[float]]):
    """Collect per-rollout metrics into the accumulator dict."""
    accum["reward/exec"].append(rollout.reward_exec)
    accum["reward/rmi"].append(rollout.reward_rmi)
    accum["reward/total"].append(rollout.reward_total)
    accum["gen/num_tokens"].append(rollout.metrics.get("gen/num_tokens", 0))
    accum["correctness"].append(rollout.metrics.get("correctness", 0.0))
    # Two-phase metrics
    accum["gen/prefill_injected"].append(rollout.metrics.get("gen/prefill_injected", 0.0))
    accum["gen/phase1_tokens"].append(rollout.metrics.get("gen/phase1_tokens", 0))
    accum["gen/phase2_tokens"].append(rollout.metrics.get("gen/phase2_tokens", 0))
    # Uncertainty decomposition — all 4 for ablation comparison (mean and top-7%)
    accum["uncertainty/true_mi"].append(rollout.metrics.get("uncertainty/true_mi", 0.0))
    accum["uncertainty/rmi"].append(rollout.metrics.get("uncertainty/rmi", 0.0))
    accum["uncertainty/variance"].append(rollout.metrics.get("uncertainty/variance", 0.0))
    accum["uncertainty/predictive_entropy"].append(
        rollout.metrics.get("uncertainty/predictive_entropy", 0.0)
    )
    accum["uncertainty/rmi_top7"].append(rollout.metrics.get("uncertainty/rmi_top7", 0.0))
    accum["uncertainty/variance_top7"].append(rollout.metrics.get("uncertainty/variance_top7", 0.0))
    accum["uncertainty/predictive_entropy_top7"].append(
        rollout.metrics.get("uncertainty/predictive_entropy_top7", 0.0)
    )
    # Top-7% phase breakdown (thinking vs code)
    accum["uncertainty/top7_pct_thinking"].append(
        rollout.metrics.get("uncertainty/top7_pct_thinking", 0.0)
    )
    accum["uncertainty/top7_pct_code"].append(
        rollout.metrics.get("uncertainty/top7_pct_code", 0.0)
    )
    accum["uncertainty/top7_mi_thinking"].append(
        rollout.metrics.get("uncertainty/top7_mi_thinking", 0.0)
    )
    accum["uncertainty/top7_mi_code"].append(
        rollout.metrics.get("uncertainty/top7_mi_code", 0.0)
    )
    # Ensemble disagreement — proves members actually diverge
    accum["ensemble/member_logprob_spread"].append(
        rollout.metrics.get("ensemble/member_logprob_spread", 0.0)
    )
    accum["ensemble/member_logprob_range"].append(
        rollout.metrics.get("ensemble/member_logprob_range", 0.0)
    )
    accum["ensemble/max_token_disagreement"].append(
        rollout.metrics.get("ensemble/max_token_disagreement", 0.0)
    )
    accum["ensemble/frac_high_disagreement"].append(
        rollout.metrics.get("ensemble/frac_high_disagreement", 0.0)
    )
    # Streaming MI early stop
    accum["streaming_mi/stopped"].append(rollout.metrics.get("streaming_mi/stopped", 0.0))
    accum["streaming_mi/tokens_saved"].append(rollout.metrics.get("streaming_mi/tokens_saved", 0))


def train_step(
    ensemble: LoRAEnsemble,
    rollout_groups: List[RolloutGroup],
    cfg: Config,
) -> Dict[str, float]:
    """
    Train all K adapters on a batch of rollout groups.

    Mirrors the original train_step() → forward_backward() → optim_step() flow,
    but uses mLoRA's batched forward and trains all K adapters per step.

    Flow:
        1. For each group: compute advantages from augmented rewards
        2. Apply KL penalty if kl_penalty_coef > 0
        3. Split into num_substeps mini-batches
        4. For each substep:
           a. Forward-backward on mini-batch
           b. Optimizer step
        5. Compute KL tracking metrics
    """
    metrics_accum: Dict[str, List[float]] = {
        "reward/exec": [], "reward/rmi": [], "reward/total": [],
        "advantage/mean": [], "advantage/std": [],
        "advantage/min": [], "advantage/max": [],
        "beta": [], "gamma_eff": [],
        "gen/num_tokens": [], "correctness": [],
        # Two-phase generation tracking
        "gen/prefill_injected": [], "gen/phase1_tokens": [], "gen/phase2_tokens": [],
        # Uncertainty decomposition (all 4 metrics — critical for ablation)
        "uncertainty/true_mi": [], "uncertainty/rmi": [], "uncertainty/variance": [],
        "uncertainty/predictive_entropy": [],
        # Top-7% baselines — same aggregation as true_mi for fair ablation comparison
        "uncertainty/rmi_top7": [], "uncertainty/variance_top7": [],
        "uncertainty/predictive_entropy_top7": [],
        # Top-7% phase breakdown (thinking vs code)
        "uncertainty/top7_pct_thinking": [], "uncertainty/top7_pct_code": [],
        "uncertainty/top7_mi_thinking": [], "uncertainty/top7_mi_code": [],
        # Ensemble disagreement (proves K members actually diverge)
        "ensemble/member_logprob_spread": [],
        "ensemble/member_logprob_range": [],
        "ensemble/max_token_disagreement": [],
        "ensemble/frac_high_disagreement": [],
        # Per-group: RMI-reward correlation (proves RMI isn't just noise)
        "group/rmi_exec_corr": [],
        # Streaming MI early stop
        "streaming_mi/stopped": [], "streaming_mi/tokens_saved": [],
    }

    # ── Phase 1: Compute advantages for all groups, collect trainable rollouts ──
    # When kl_penalty_coef > 0, advantages are per-token tensors (matching train.py).
    # When kl_penalty_coef == 0, advantages are scalars (float).
    trainable_rollouts: List[Tuple["Rollout", "Union[float, torch.Tensor]"]] = []

    for group in rollout_groups:
        # β sees R_exec ONLY — preserves max-reward-seeking gradient
        scalar_advantages, adv_meta = compute_advantages(
            group.get_exec_rewards(),
            adv_estimator=cfg.adv_estimator,
            adv_estimator_beta=cfg.adv_estimator_beta,
        )

        # ── RMI advantage-space bonus (after β, before KL) ──
        # γ_eff is coupled to β: exploration pressure increases as policy converges
        # β_ref = adv_estimator_beta so γ_eff = rmi_coef when β equals the fixed default
        beta = adv_meta.get("beta", cfg.adv_estimator_beta)
        if cfg.rmi_coef > 0:
            rmi_values = torch.tensor(
                [r.reward_rmi for r in group.rollouts], dtype=torch.float32
            )
            # Replace NaN RMI values with 0 (neutral) to prevent contamination
            rmi_values = torch.where(torch.isnan(rmi_values), torch.zeros_like(rmi_values), rmi_values)
            rmi_centered = rmi_values - rmi_values.mean()
            # Normalize RMI to unit scale so γ_eff directly controls strength
            # relative to A_exec, regardless of raw RMI magnitude
            rmi_std = rmi_centered.std()
            if rmi_std > 1e-8:
                rmi_centered = rmi_centered / rmi_std
            rmi_centered = rmi_centered.clamp(-3, 3)
            gamma_eff = cfg.rmi_coef * min(beta / cfg.adv_estimator_beta, cfg.gamma_max_ratio)
            # Scale RMI bonus proportional to advantage magnitude
            adv_scale = scalar_advantages.abs().mean().clamp(min=1e-8)
            scalar_advantages = scalar_advantages + gamma_eff * adv_scale * rmi_centered
        else:
            gamma_eff = 0.0

        metrics_accum["beta"].append(float(beta))
        metrics_accum["gamma_eff"].append(float(gamma_eff))

        # ── Per-group: RMI-reward correlation (proves RMI targets novelty, not noise) ──
        if len(group.rollouts) >= 3:
            exec_rewards = [r.reward_exec for r in group.rollouts]
            rmi_rewards = [r.reward_rmi for r in group.rollouts]
            if np.std(exec_rewards) > 1e-8 and np.std(rmi_rewards) > 1e-8:
                corr = float(np.corrcoef(exec_rewards, rmi_rewards)[0, 1])
                if not np.isnan(corr):
                    metrics_accum["group/rmi_exec_corr"].append(corr)

        # ── KL penalty adjustment → promotes to per-token advantages ──
        kl_metrics = {}
        if cfg.kl_penalty_coef > 0:
            per_token_advs, kl_metrics = incorporate_kl_penalty(
                rollouts=group.rollouts,
                advantages=scalar_advantages,
                ensemble=ensemble,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
                device=cfg.device,
            )
            metrics_accum["advantage/mean"].append(scalar_advantages.mean().item())
            metrics_accum["advantage/std"].append(scalar_advantages.std().item())
            metrics_accum["advantage/min"].append(scalar_advantages.min().item())
            metrics_accum["advantage/max"].append(scalar_advantages.max().item())

            for rollout, scalar_adv, token_adv in zip(
                group.rollouts, scalar_advantages, per_token_advs
            ):
                _collect_rollout_metrics(rollout, metrics_accum)

                if abs(scalar_adv.item()) < 1e-8:
                    continue

                trainable_rollouts.append((rollout, token_adv))
        else:
            metrics_accum["advantage/mean"].append(scalar_advantages.mean().item())
            metrics_accum["advantage/std"].append(scalar_advantages.std().item())
            metrics_accum["advantage/min"].append(scalar_advantages.min().item())
            metrics_accum["advantage/max"].append(scalar_advantages.max().item())

            for rollout, adv in zip(group.rollouts, scalar_advantages):
                _collect_rollout_metrics(rollout, metrics_accum)

                if abs(adv.item()) < 1e-8:
                    continue

                trainable_rollouts.append((rollout, adv.item()))

    # ── Phase 2: Substep training (matching original train_step batch splitting) ──
    num_substeps = min(cfg.num_substeps, len(trainable_rollouts)) if trainable_rollouts else 0
    substep_batches = _split_list(trainable_rollouts, num_substeps) if num_substeps > 0 else []

    total_loss_val = 0.0
    nnm_loss_val = 0.0
    nnm_nuc_val = 0.0
    all_training_logprobs: List[torch.Tensor] = []
    all_trained_rollouts: List[Rollout] = []

    for substep_batch in substep_batches:
        n_batch = max(len(substep_batch), 1)

        for rollout, adv in substep_batch:
            action_mask = rollout.action_mask.to(cfg.device)
            old_logprobs = torch.tensor(
                rollout.sampling_logprobs, device=cfg.device
            )

            # Sequential per-adapter forward+backward: peak activation memory is
            # O(T × L × D) instead of O(K × T × L × D) from the joint K-adapter pass.
            # Each adapter's computation graph is freed by backward() before the next
            # forward, keeping only 1/K of the joint approach's activation memory.
            rollout_loss_val = 0.0
            first_adapter_logprobs = None

            for k in range(ensemble.K):
                _hidden_k, logprobs_k = ensemble.compute_training_logprobs_single(
                    k, rollout.full_tokens
                )  # logprobs_k: (T-1,)
                del _hidden_k

                if k == 0:
                    first_adapter_logprobs = logprobs_k.detach()

                loss_k = compute_rl_loss(
                    current_logprobs=logprobs_k,
                    old_logprobs=old_logprobs,
                    advantage=adv,
                    action_mask=action_mask,
                    loss_fn=cfg.loss_fn,
                    ppo_clip=cfg.ppo_clip,
                )
                # Divide by K (adapters) and n_batch (rollouts) so gradient magnitude
                # matches the original joint backward.
                scaled_k = loss_k / (ensemble.K * n_batch)
                scaled_k.backward()  # frees this adapter's computation graph
                rollout_loss_val += scaled_k.item()
                del logprobs_k, loss_k, scaled_k

            # Store first adapter's training logprobs for KL tracking
            all_training_logprobs.append(first_adapter_logprobs)
            all_trained_rollouts.append(rollout)
            total_loss_val += rollout_loss_val

        if len(substep_batch) > 0:
            # Nuclear norm diversity regularization — touches all K lora_A tensors
            # simultaneously after RL gradients have accumulated. Controlled by
            # --nnm_coef (0 = disabled, no overhead). Gradient magnitude normalised
            # by n_batch to match the RL loss scaling above.
            if cfg.nnm_coef > 0.0:
                _nnm_loss, nnm_nuc_val, _ = compute_nuclear_norm_diversity_loss(
                    ensemble, cfg.nnm_use_fro_norm
                )
                _weighted = cfg.nnm_coef * _nnm_loss / n_batch
                _weighted.backward()
                nnm_loss_val = _weighted.item()

            ensemble.step_all_optimizers()

    # ── Phase 3: KL tracking metrics ──
    kl_track_metrics = {}
    if all_trained_rollouts:
        kl_track_metrics = compute_kl_sample_train(
            all_trained_rollouts, all_training_logprobs, cfg.device
        )

    # ── Aggregate metrics ──
    metrics: Dict[str, float] = {}

    # Rewards: mean, std, min, max
    for key in ["reward/exec", "reward/rmi", "reward/total"]:
        vals = metrics_accum[key]
        if vals:
            metrics[f"train/{key}/mean"] = float(np.mean(vals))
            metrics[f"train/{key}/std"] = float(np.std(vals))
            metrics[f"train/{key}/min"] = float(np.min(vals))
            metrics[f"train/{key}/max"] = float(np.max(vals))

    # Advantages: already per-group mean/std/min/max, aggregate across groups
    if metrics_accum["advantage/mean"]:
        metrics["train/advantage/mean"] = float(np.mean(metrics_accum["advantage/mean"]))
        metrics["train/advantage/std"] = float(np.mean(metrics_accum["advantage/std"]))
    if metrics_accum["advantage/min"]:
        metrics["train/advantage/min"] = float(np.min(metrics_accum["advantage/min"]))
    if metrics_accum["advantage/max"]:
        metrics["train/advantage/max"] = float(np.max(metrics_accum["advantage/max"]))

    # Beta and gamma_eff
    if metrics_accum["beta"]:
        metrics["train/beta/mean"] = float(np.mean(metrics_accum["beta"]))
        metrics["train/beta/std"] = float(np.std(metrics_accum["beta"]))
    if metrics_accum["gamma_eff"]:
        metrics["train/gamma_eff/mean"] = float(np.mean(metrics_accum["gamma_eff"]))

    # Generation stats
    if metrics_accum["gen/num_tokens"]:
        metrics["train/gen/num_tokens/mean"] = float(np.mean(metrics_accum["gen/num_tokens"]))
        metrics["train/gen/num_tokens/std"] = float(np.std(metrics_accum["gen/num_tokens"]))
        metrics["train/gen/num_tokens/max"] = float(np.max(metrics_accum["gen/num_tokens"]))

    # Two-phase generation stats
    if metrics_accum["gen/prefill_injected"]:
        metrics["train/two_phase/prefill_rate"] = float(np.mean(metrics_accum["gen/prefill_injected"]))
    if metrics_accum["gen/phase1_tokens"]:
        metrics["train/two_phase/phase1_tokens/mean"] = float(np.mean(metrics_accum["gen/phase1_tokens"]))
    if metrics_accum["gen/phase2_tokens"]:
        metrics["train/two_phase/phase2_tokens/mean"] = float(np.mean(metrics_accum["gen/phase2_tokens"]))

    # Correctness (fraction of rollouts that got reward > 0)
    if metrics_accum["correctness"]:
        metrics["train/correctness/mean"] = float(np.mean(metrics_accum["correctness"]))
        metrics["train/correctness/rate"] = float(
            np.mean([1.0 if c > 0 else 0.0 for c in metrics_accum["correctness"]])
        )

    # ── Uncertainty decomposition (ablation: all metrics, mean and top-7%) ──
    for key in [
        "uncertainty/true_mi",
        "uncertainty/rmi", "uncertainty/variance", "uncertainty/predictive_entropy",
        "uncertainty/rmi_top7", "uncertainty/variance_top7", "uncertainty/predictive_entropy_top7",
    ]:
        vals = metrics_accum[key]
        if vals:
            metrics[f"train/{key}/mean"] = float(np.mean(vals))
            metrics[f"train/{key}/std"] = float(np.std(vals))
            metrics[f"train/{key}/min"] = float(np.min(vals))
            metrics[f"train/{key}/max"] = float(np.max(vals))

    # ── Top-7% phase breakdown (thinking vs code generation) ──
    for key in [
        "uncertainty/top7_pct_thinking", "uncertainty/top7_pct_code",
        "uncertainty/top7_mi_thinking", "uncertainty/top7_mi_code",
    ]:
        vals = metrics_accum[key]
        if vals:
            metrics[f"train/{key}/mean"] = float(np.mean(vals))
            metrics[f"train/{key}/std"] = float(np.std(vals))

    # ── Ensemble disagreement (proves K members actually diverge) ──
    for key in ["ensemble/member_logprob_spread", "ensemble/member_logprob_range",
                "ensemble/max_token_disagreement", "ensemble/frac_high_disagreement"]:
        vals = metrics_accum[key]
        if vals:
            metrics[f"train/{key}/mean"] = float(np.mean(vals))

    # ── RMI-reward correlation (is uncertainty tracking novelty or noise?) ──
    if metrics_accum["group/rmi_exec_corr"]:
        corrs = metrics_accum["group/rmi_exec_corr"]
        metrics["train/group/rmi_exec_corr/mean"] = float(np.mean(corrs))
        metrics["train/group/rmi_exec_corr/std"] = float(np.std(corrs))

    # ── Streaming MI early stop stats ──
    if metrics_accum["streaming_mi/stopped"]:
        stopped_vals = metrics_accum["streaming_mi/stopped"]
        saved_vals = metrics_accum["streaming_mi/tokens_saved"]
        metrics["train/streaming_mi/stop_rate"] = float(np.mean(stopped_vals))
        metrics["train/streaming_mi/tokens_saved/mean"] = float(np.mean(saved_vals))
        metrics["train/streaming_mi/tokens_saved/total"] = float(np.sum(saved_vals))

    # Core training stats
    metrics["train/loss"] = total_loss_val
    metrics["train/nnm_loss"] = nnm_loss_val        # 0.0 when --nnm_coef is not set
    metrics["train/nnm_nuclear_norm"] = nnm_nuc_val
    metrics["train/num_rollouts"] = float(len(trainable_rollouts))
    metrics["train/num_rollouts_total"] = float(
        sum(len(g.rollouts) for g in rollout_groups)
    )
    metrics["train/num_groups"] = float(len(rollout_groups))

    metrics.update(kl_track_metrics)
    if kl_metrics:
        metrics.update(kl_metrics)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# main() — mirrors train.py:main()
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    cfg: Config,
    prompt_fn: Callable[[Optional[Any]], str],
    process_fn: Callable[[str, Any, int], Dict[str, Any]],
    sampler: StateSampler,
):
    """
    Main training entry point.

    Args:
        cfg: Training configuration.
        prompt_fn: (state) → prompt text string.
        process_fn: (generated_text, parent_state, step) → dict with
            reward, child_state, correctness, metrics.
        sampler: State sampler (greedy, PUCT, etc.).
    """
    # ── 1. Setup logging (reuses existing ml_log) ─────────────────────────
    ml_logger = setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    # Log which loggers are active so user knows immediately if wandb was skipped
    if hasattr(ml_logger, 'loggers'):
        active = [type(l).__name__ for l in ml_logger.loggers]
        logger.info(f"Active loggers: {active}")
    else:
        logger.info(f"Active logger: {type(ml_logger).__name__}")

    # ── Configure wandb custom x-axes ────────────────────────────────────
    # Each metric family gets its own step_metric so per-rollout, per-group,
    # and per-epoch logs don't collide (wandb requires monotonic global step).
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.define_metric("rollout/*", step_metric="rollout/_global_step")
            _wandb.define_metric("rollout_group/*", step_metric="rollout_group/_global_step")
            _wandb.define_metric("train/*", step_metric="progress/epoch")
            _wandb.define_metric("progress/*", step_metric="progress/epoch")
            _wandb.define_metric("time/*", step_metric="progress/epoch")
            _wandb.define_metric("gpu/*", step_metric="progress/epoch")
            _wandb.define_metric("optim/*", step_metric="progress/epoch")
            _wandb.define_metric("kl_policy_base", step_metric="progress/epoch")
    except ImportError:
        pass

    # Helper: log metrics to wandb/json only (skip PrettyPrint Rich tables)
    # Used for per-group rollout metrics to avoid flooding console with tables
    def _log_metrics_quiet(metrics: Dict[str, Any], step: int | None = None):
        """Log to wandb + json but skip PrettyPrint (no Rich table spam)."""
        if hasattr(ml_logger, 'loggers'):
            for l in ml_logger.loggers:
                if type(l).__name__ != 'PrettyPrintLogger':
                    l.log_metrics(metrics, step=step)
        else:
            ml_logger.log_metrics(metrics, step=step)

    # ── 2. Load model + ensemble ──────────────────────────────────────────
    logger.info(f"Loading base model: {cfg.base_model} ({cfg.precision})")
    args = argparse.Namespace(
        base_model=cfg.base_model,
        device=cfg.device,
        precision=cfg.precision,
        model_type="llama",
        pipeline=False,
        balance=None,
        rank=None,
    )
    tokenizer, model = mlora_load_model(args)

    ensemble = LoRAEnsemble(
        model=model,
        num_members=cfg.num_ensemble_members,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        learning_rate=cfg.learning_rate,
        optimizer="adamw",
    )

    if cfg.uncertainty_metric == "true_mi":
        # True MI is computed from full vocab distributions during scoring;
        # None signals callers to use pre-computed per_token_true_mi
        uncertainty_fn = None
    else:
        uncertainty_fn = UNCERTAINTY_FNS[cfg.uncertainty_metric]

    # ── Streaming MI tracker (Contribution 2) ────────────────────────────
    mi_tracker: Optional[StreamingMITracker] = None
    if cfg.streaming_mi_enabled:
        mi_config = StreamingMIConfig(
            enabled=True,
            window_size=cfg.streaming_mi_window_size,
            check_interval=cfg.streaming_mi_check_interval,
            min_gen=cfg.streaming_mi_min_gen,
            warmup_epochs=cfg.streaming_mi_warmup_epochs,
            wrap_budget=cfg.streaming_mi_wrap_budget,
            threshold_percentile=cfg.streaming_mi_threshold_percentile,
        )
        mi_tracker = StreamingMITracker(mi_config)
        logger.info(
            f"Streaming MI early stop enabled: W={mi_config.window_size}, "
            f"C={mi_config.check_interval}, min_gen={mi_config.min_gen}, "
            f"warmup={mi_config.warmup_epochs} epochs, wrap={mi_config.wrap_budget}"
        )

    # ── 3. Resume ─────────────────────────────────────────────────────────
    start_epoch = 0
    checkpoint_file = os.path.join(cfg.log_path, "last_epoch.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            start_epoch = int(f.read().strip()) + 1
        ensemble.load(cfg.log_path, start_epoch - 1)
        logger.info(f"Resumed from epoch {start_epoch}")

        # Rebuild MI history from metrics.jsonl so threshold is correct on resume
        if mi_tracker is not None:
            metrics_path = os.path.join(cfg.log_path, "metrics.jsonl")
            if os.path.exists(metrics_path):
                mi_values_from_log = []
                with open(metrics_path) as mf:
                    for line in mf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        mi_val = rec.get("rollout/uncertainty_true_mi")
                        if mi_val is not None:
                            mi_values_from_log.append(float(mi_val))
                if mi_values_from_log:
                    mi_tracker.record_post_generation_mi(mi_values_from_log)
                    mi_tracker.update_threshold(start_epoch - 1)
                    logger.info(
                        f"Rebuilt MI history from {len(mi_values_from_log)} rollouts, "
                        f"threshold={mi_tracker.current_threshold:.6f}"
                    )

    os.makedirs(cfg.log_path, exist_ok=True)

    # ── 4. Training loop ─────────────────────────────────────────────────
    num_batches_per_epoch = cfg.groups_per_batch
    num_batches_total = num_batches_per_epoch * cfg.num_epochs
    logger.info(
        f"Training: {cfg.num_epochs} epochs, "
        f"{cfg.groups_per_batch} groups/batch × {cfg.group_size} rollouts/group, "
        f"K={cfg.num_ensemble_members} adapters, γ={cfg.rmi_coef}"
    )
    logger.info(
        f"Parallelism: batched generation ({cfg.group_size} seqs/batch), "
        f"threaded rewards ({cfg.group_size} workers), batched ensemble scoring"
    )

    best_reward_exec = float("-inf")
    best_reward_total = float("-inf")

    # Thread pool for parallel CPU reward computation
    reward_workers = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(cfg.group_size, 8)
    )

    # ── Two-phase setup (compute once before training loop) ──────────────
    prefill_tokens_list: Optional[List[int]] = None
    stop_token_ids: Optional[List[int]] = None
    if cfg.two_phase_sampling:
        prefill_text = QWEN3_PHASE2_PREFILL + QWEN3_THINK_CLOSE
        prefill_tokens_list = tokenizer.encode(prefill_text, bos=False, eos=False)
        # <|im_end|> is the stop token for Qwen3 chat format
        im_end_tokens = tokenizer.tokenizer_.encode("<|im_end|>", add_special_tokens=False)
        if im_end_tokens:
            stop_token_ids = [im_end_tokens[0]]
        logger.info(
            f"Two-phase sampling enabled: phase1_max={cfg.phase1_max_tokens}, "
            f"context_window={cfg.context_window}, prefill_len={len(prefill_tokens_list)}, "
            f"stop_ids={stop_token_ids}"
        )

    for epoch in range(start_epoch, cfg.num_epochs):
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "progress/epoch": epoch,
            "progress/done_frac": (epoch + 1) / cfg.num_epochs,
        }

        logger.info(
            f"[Epoch {epoch}/{cfg.num_epochs}] Starting rollout phase: "
            f"{cfg.groups_per_batch} groups × {cfg.group_size} rollouts"
        )

        # ── Rollout phase (batched) ──────────────────────────────────
        rollout_groups: List[RolloutGroup] = []
        parent_states = sampler.sample_states(cfg.groups_per_batch)
        num_groups_total = len(parent_states)

        for g, parent_state in enumerate(parent_states):
            t_group = time.time()
            prompt_text = prompt_fn(parent_state)
            if cfg.two_phase_sampling:
                prompt_text = wrap_qwen3_chat_template(prompt_text)
            prompt_tokens = tokenizer.encode(prompt_text, bos=True, eos=False)

            group_rollouts = do_group_rollout_batched(
                ensemble=ensemble,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                group_size=cfg.group_size,
                parent_state=parent_state,
                epoch=epoch,
                group_idx=g,
                process_fn=process_fn,
                uncertainty_fn=uncertainty_fn,
                rmi_coef=cfg.rmi_coef,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                reward_workers=reward_workers,
                kv_quantize=cfg.kv_quantize,
                two_phase_sampling=cfg.two_phase_sampling,
                phase1_max_tokens=cfg.phase1_max_tokens,
                context_window=cfg.context_window,
                context_buffer=cfg.context_buffer,
                prefill_tokens=prefill_tokens_list,
                stop_token_ids=stop_token_ids,
                mi_tracker=mi_tracker,
                streaming_mi_cfg=cfg if cfg.streaming_mi_enabled else None,
            )

            # ── Local JSONL rollout log (written FIRST so it survives any downstream crash) ──
            _write_rollout_logs(
                log_path=cfg.log_path,
                rollouts=group_rollouts,
                tokenizer=tokenizer,
                epoch=epoch,
                group_idx=g,
                prompt_text=prompt_text,
            )

            # Per-rollout wandb logging
            for i, rollout in enumerate(group_rollouts):
                rollout_step = (epoch * num_groups_total + g) * cfg.group_size + i
                _log_metrics_quiet({
                    "rollout/_global_step": rollout_step,
                    "rollout/epoch": epoch,
                    "rollout/group_idx": g,
                    "rollout/idx": i,
                    "rollout/adapter_idx": rollout.adapter_idx,
                    "rollout/reward_exec": rollout.reward_exec,
                    "rollout/reward_rmi": rollout.reward_rmi,
                    "rollout/reward_total": rollout.reward_total,
                    "rollout/correctness": rollout.metrics.get("correctness", 0.0),
                    "rollout/num_tokens": len(rollout.full_tokens) - len(rollout.prompt_tokens),
                    "rollout/prefill_injected": rollout.metrics.get("gen/prefill_injected", 0.0),
                    "rollout/phase1_tokens": rollout.metrics.get("gen/phase1_tokens", 0),
                    "rollout/phase2_tokens": rollout.metrics.get("gen/phase2_tokens", 0),
                    "rollout/uncertainty_true_mi": rollout.metrics.get("uncertainty/true_mi", 0.0),
                    "rollout/uncertainty_rmi": rollout.metrics.get("uncertainty/rmi", 0.0),
                    "rollout/uncertainty_variance": rollout.metrics.get("uncertainty/variance", 0.0),
                    "rollout/uncertainty_predictive_entropy": rollout.metrics.get("uncertainty/predictive_entropy", 0.0),
                    "rollout/uncertainty_rmi_top7": rollout.metrics.get("uncertainty/rmi_top7", 0.0),
                    "rollout/uncertainty_variance_top7": rollout.metrics.get("uncertainty/variance_top7", 0.0),
                    "rollout/uncertainty_predictive_entropy_top7": rollout.metrics.get("uncertainty/predictive_entropy_top7", 0.0),
                    # Top-7% phase breakdown (thinking vs code generation)
                    "rollout/top7_pct_thinking": rollout.metrics.get("uncertainty/top7_pct_thinking", 0.0),
                    "rollout/top7_pct_code": rollout.metrics.get("uncertainty/top7_pct_code", 0.0),
                    "rollout/top7_mi_thinking": rollout.metrics.get("uncertainty/top7_mi_thinking", 0.0),
                    "rollout/top7_mi_code": rollout.metrics.get("uncertainty/top7_mi_code", 0.0),
                    "rollout/ensemble_member_logprob_spread": rollout.metrics.get("ensemble/member_logprob_spread", 0.0),
                    "rollout/ensemble_member_logprob_range": rollout.metrics.get("ensemble/member_logprob_range", 0.0),
                    "rollout/ensemble_max_token_disagreement": rollout.metrics.get("ensemble/max_token_disagreement", 0.0),
                    "rollout/ensemble_frac_high_disagreement": rollout.metrics.get("ensemble/frac_high_disagreement", 0.0),
                    "rollout/streaming_mi_stopped": rollout.metrics.get("streaming_mi/stopped", 0.0),
                    "rollout/streaming_mi_tokens_saved": rollout.metrics.get("streaming_mi/tokens_saved", 0),
                })

            # Per-group metrics
            group_rewards_exec = [r.reward_exec for r in group_rollouts]
            group_rewards_total = [r.reward_total for r in group_rollouts]
            group_rmi = [r.reward_rmi for r in group_rollouts]
            t_group_elapsed = time.time() - t_group

            # Filter constant-reward groups (same as original)
            rewards = [r.reward_total for r in group_rollouts]
            kept = not (cfg.remove_constant_reward_groups and len(set(rewards)) <= 1)

            if kept:
                rollout_groups.append(
                    RolloutGroup(rollouts=group_rollouts, state=parent_state)
                )

            # ── Per-group console log ────────────────────────────────
            status = "KEPT" if kept else "FILTERED"
            gpu_msg = ""
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                gpu_msg = f" | GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
            logger.info(
                f"  [{epoch}][{g+1}/{num_groups_total}] {status} | "
                f"R_exec={np.mean(group_rewards_exec):.4f} (max={np.max(group_rewards_exec):.4f}) "
                f"RMI={np.mean(group_rmi):.4f} "
                f"R_total={np.mean(group_rewards_total):.4f} | "
                f"{t_group_elapsed:.1f}s{gpu_msg}"
            )

            # ── Per-group wandb + json logging (skip Rich table) ─────
            global_group_step = epoch * num_groups_total + g
            group_metrics: Dict[str, Any] = {
                "rollout_group/_global_step": global_group_step,
                "rollout_group/epoch": epoch,
                "rollout_group/group_idx": g,
                "rollout_group/reward_exec_mean": float(np.mean(group_rewards_exec)),
                "rollout_group/reward_exec_max": float(np.max(group_rewards_exec)),
                "rollout_group/reward_exec_min": float(np.min(group_rewards_exec)),
                "rollout_group/reward_total_mean": float(np.mean(group_rewards_total)),
                "rollout_group/rmi_mean": float(np.mean(group_rmi)),
                "rollout_group/kept": float(kept),
                "rollout_group/time_s": t_group_elapsed,
                "rollout_group/streaming_mi_stop_rate": float(np.mean([
                    r.metrics.get("streaming_mi/stopped", 0.0) for r in group_rollouts
                ])),
                "rollout_group/streaming_mi_tokens_saved_mean": float(np.mean([
                    r.metrics.get("streaming_mi/tokens_saved", 0) for r in group_rollouts
                ])),
            }
            if torch.cuda.is_available():
                group_metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                group_metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            _log_metrics_quiet(group_metrics)

            # Free any residual KV cache / scoring tensors before the next group
            torch.cuda.empty_cache()

        metrics["time/rollout"] = time.time() - t_start

        # ── Train step ────────────────────────────────────────────────
        logger.info(
            f"[Epoch {epoch}] Rollout phase done in {metrics['time/rollout']:.1f}s. "
            f"Training on {len(rollout_groups)}/{num_groups_total} groups..."
        )
        t_train = time.time()
        torch.cuda.empty_cache()  # release fragmented reserved-but-unallocated pool before backward
        train_metrics = train_step(ensemble, rollout_groups, cfg)
        metrics.update(train_metrics)
        metrics["time/train"] = time.time() - t_train
        torch.cuda.empty_cache()  # release post-backward gradient/activation pool before next epoch's rollout

        # ── Streaming MI threshold update ────────────────────────────
        if mi_tracker is not None:
            mi_tracker.update_threshold(epoch)
            mi_metrics = mi_tracker.get_metrics()
            for k, v in mi_metrics.items():
                metrics[f"train/{k}"] = v
            mi_tracker.reset_epoch_counters()

        # ── Epoch-level metrics ──────────────────────────────────────
        # Track best rewards seen so far (running max)
        epoch_exec_max = metrics.get("train/reward/exec/max", float("-inf"))
        epoch_total_max = metrics.get("train/reward/total/max", float("-inf"))
        best_reward_exec = max(best_reward_exec, epoch_exec_max)
        best_reward_total = max(best_reward_total, epoch_total_max)
        metrics["train/reward/exec/best"] = best_reward_exec
        metrics["train/reward/total/best"] = best_reward_total

        # How many groups were filtered out (constant reward)
        num_groups_attempted = len(parent_states)
        num_groups_kept = len(rollout_groups)
        metrics["rollout/groups_attempted"] = float(num_groups_attempted)
        metrics["rollout/groups_kept"] = float(num_groups_kept)
        metrics["rollout/groups_filtered"] = float(num_groups_attempted - num_groups_kept)

        # Reward diversity within groups (important for advantage estimation)
        if rollout_groups:
            group_reward_stds = []
            for group in rollout_groups:
                rews = [r.reward_exec for r in group.rollouts]
                group_reward_stds.append(float(np.std(rews)))
            metrics["rollout/intra_group_reward_std"] = float(np.mean(group_reward_stds))

        # ── Update sampler buffer with child states ──────────────────
        for group in rollout_groups:
            child_states = []
            parent_states_for_update = []
            for rollout in group.rollouts:
                if rollout.child_state is not None:
                    child_states.append(rollout.child_state)
                    parent_states_for_update.append(rollout.state)
                elif hasattr(sampler, 'record_failed_rollout'):
                    sampler.record_failed_rollout(rollout.state)
            if child_states:
                sampler.update_states(child_states, parent_states_for_update, save=False)
        sampler.flush(step=epoch)

        # ── Logging ───────────────────────────────────────────────────
        metrics["time/total"] = time.time() - t_start
        if torch.cuda.is_available():
            metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            metrics["gpu/memory_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        ml_logger.log_metrics(metrics)

        r_exec = metrics.get("train/reward/exec/mean", 0)
        r_max = metrics.get("train/reward/exec/max", 0)
        r_best = metrics.get("train/reward/exec/best", 0)
        r_rmi = metrics.get("train/reward/rmi/mean", 0)
        loss = metrics.get("train/loss", 0)
        beta = metrics.get("train/beta/mean", 0)
        corr = metrics.get("train/correctness/rate", 0)
        nuc_norm = metrics.get("train/nnm_nuclear_norm", 0)
        nnm_info = f" nuc={nuc_norm:.3f}" if cfg.nnm_coef > 0.0 else ""
        logger.info(
            f"[Epoch {epoch}] R_exec={r_exec:.4f} (max={r_max:.4f} best={r_best:.4f}) "
            f"RMI={r_rmi:.4f} beta={beta:.2f} corr={corr:.1%} loss={loss:.4f}"
            f"{nnm_info} groups={num_groups_kept}/{num_groups_attempted} time={metrics['time/total']:.1f}s"
        )

        # ── Checkpointing ─────────────────────────────────────────────
        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ensemble.save(cfg.log_path, epoch)
            with open(checkpoint_file, "w") as f:
                f.write(str(epoch))
            logger.info(f"Checkpoint saved at epoch {epoch}")

    # ── 5. Cleanup + final checkpoint ────────────────────────────────────
    reward_workers.shutdown(wait=False)
    ensemble.save(cfg.log_path, cfg.num_epochs - 1)
    with open(checkpoint_file, "w") as f:
        f.write(str(cfg.num_epochs - 1))

    ml_logger.close()
    logger.info("Training complete")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — mirrors ug_ttt/recipes/ttt/train.py:cli_main()
# ═══════════════════════════════════════════════════════════════════════════════

def cli_main():
    """
    CLI entry point. Parses args → Config, then calls main().

    Task/prompt/sampler wiring should happen here, matching the pattern
    in ug_ttt/recipes/ttt/train.py:cli_main().
    """
    parser = argparse.ArgumentParser(description="Multi-LoRA Ensemble RL Training")

    # Model
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--precision", default="fp16",
                        choices=["nf4", "fp4", "int8", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")

    # Environment
    parser.add_argument("--env", default="ac1")
    parser.add_argument("--problem_idx", default="improvement")
    parser.add_argument("--budget_s", type=int, default=1000)

    # RL
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--groups_per_batch", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=26000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adv_estimator", default="entropic_adaptive_beta")
    parser.add_argument("--loss_fn", default="importance_sampling")

    # Performance
    parser.add_argument("--kv_quantize", action="store_true", default=False,
                        help="Int8 KV cache: halves memory bandwidth for faster batched decode")
    parser.add_argument("--no_kv_quantize", dest="kv_quantize", action="store_false")

    # Ensemble
    parser.add_argument("--num_ensemble_members", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=4e-5)

    # Uncertainty
    parser.add_argument("--rmi_coef", type=float, default=0.1)
    parser.add_argument("--uncertainty_metric", default="true_mi",
                        choices=["true_mi", "rmi", "variance", "predictive_entropy"])

    # Nuclear norm diversity regularization
    parser.add_argument("--nnm_coef", type=float, default=0.0,
                        help="Nuclear norm diversity regularisation weight (0=off, try 0.05-0.1)")
    parser.add_argument("--nnm_use_fro_norm", action="store_true", default=False,
                        help="Use Frobenius-normalised nuclear norm (scale-invariant)")

    # Streaming MI early stop
    parser.add_argument("--streaming_mi", action="store_true", default=False,
                        help="Enable streaming MI-based early stop during Phase 1")
    parser.add_argument("--streaming_mi_window_size", type=int, default=4096)
    parser.add_argument("--streaming_mi_check_interval", type=int, default=2048)
    parser.add_argument("--streaming_mi_min_gen", type=int, default=4096)
    parser.add_argument("--streaming_mi_warmup_epochs", type=int, default=3)
    parser.add_argument("--streaming_mi_wrap_budget", type=int, default=6000)
    parser.add_argument("--streaming_mi_threshold_percentile", type=float, default=25.0)

    # Sampler
    # KL penalty
    parser.add_argument("--kl_penalty_coef", type=float, default=0.01)
    parser.add_argument("--kl_discount_factor", type=float, default=0.0)
    parser.add_argument("--compute_post_kl", action="store_true", default=False)

    # Additional config fields from original
    parser.add_argument("--num_substeps", type=int, default=1)
    parser.add_argument("--dynamic_max_tokens", action="store_true", default=True)
    parser.add_argument("--two_phase_sampling", action="store_true", default=False)
    parser.add_argument("--phase1_max_tokens", type=int, default=26000)
    parser.add_argument("--context_window", type=int, default=32768)
    parser.add_argument("--context_buffer", type=int, default=50)
    parser.add_argument("--eval_timeout", type=int, default=1000)
    parser.add_argument("--dataset_timeout", type=int, default=1000)
    parser.add_argument("--num_cpus_per_task", type=int, default=1)
    parser.add_argument("--remove_constant_reward_groups", action="store_true", default=True)

    parser.add_argument("--sampler_type", default="puct_backprop")
    parser.add_argument("--initial_exp_type", default="random")

    # Logging
    parser.add_argument("--log_path", default="./logs/mlora_train")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    cfg = Config(
        base_model=args.base_model,
        precision=args.precision,
        device=args.device,
        env=args.env,
        problem_idx=args.problem_idx,
        budget_s=args.budget_s,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_epochs=args.num_epochs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        adv_estimator=args.adv_estimator,
        loss_fn=args.loss_fn,
        num_substeps=args.num_substeps,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        compute_post_kl=args.compute_post_kl,
        dynamic_max_tokens=args.dynamic_max_tokens,
        two_phase_sampling=args.two_phase_sampling,
        phase1_max_tokens=args.phase1_max_tokens,
        context_window=args.context_window,
        context_buffer=args.context_buffer,
        eval_timeout=args.eval_timeout,
        dataset_timeout=args.dataset_timeout,
        num_cpus_per_task=args.num_cpus_per_task,
        num_ensemble_members=args.num_ensemble_members,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        rmi_coef=args.rmi_coef,
        uncertainty_metric=args.uncertainty_metric,
        nnm_coef=args.nnm_coef,
        nnm_use_fro_norm=args.nnm_use_fro_norm,
        streaming_mi_enabled=args.streaming_mi,
        streaming_mi_window_size=args.streaming_mi_window_size,
        streaming_mi_check_interval=args.streaming_mi_check_interval,
        streaming_mi_min_gen=args.streaming_mi_min_gen,
        streaming_mi_warmup_epochs=args.streaming_mi_warmup_epochs,
        streaming_mi_wrap_budget=args.streaming_mi_wrap_budget,
        streaming_mi_threshold_percentile=args.streaming_mi_threshold_percentile,
        sampler_type=args.sampler_type,
        initial_exp_type=args.initial_exp_type,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    # ── Wire task/prompt/sampler ──────────────────────────────────────────

    from ug_ttt.recipes.ttt.sampler import create_sampler
    from ug_ttt.recipes.ttt.env_ttt import last_codeblock_postprocess
    from ug_ttt.recipes.ttt.state import InequalitiesState, CirclePackingState, ErdosState

    os.makedirs(cfg.log_path, exist_ok=True)

    # ── Build prompt_fn and process_fn based on env type ──────────────────
    if cfg.env in ("ac1", "ac2"):
        from ug_ttt.recipes.ttt.env_ac import build_ac_prompt, verify_ac

        def prompt_fn(state: InequalitiesState) -> str:
            return build_ac_prompt(
                state, env_name=cfg.env,
                problem_idx=cfg.problem_idx,
                budget_s=cfg.budget_s,
                num_cpus_per_task=cfg.num_cpus_per_task,
            )

        def process_fn(generated_text: str, parent_state: InequalitiesState, step: int) -> Dict[str, Any]:
            parsed_code = last_codeblock_postprocess(generated_text, ["python"], keep_separators=True)
            if not parsed_code or not parsed_code.strip():
                return {"reward": 0.0, "child_state": None, "correctness": 0.0, "metrics": {}}

            outs = verify_ac(
                parsed_code, step,
                num_cpus_per_task=cfg.num_cpus_per_task,
                env_name=cfg.env,
                eval_timeout=cfg.eval_timeout,
                log_path=cfg.log_path,
                state=parent_state,
            )
            correctness = outs.get("correctness", 0.0)

            # Reward — same as AutoCorrInequalityEnv._compute_reward()
            if correctness > 0:
                if cfg.env == "ac1":
                    reward = 1.0 / (-outs["performance"]) if outs["performance"] != 0 else 0.0
                else:
                    reward = outs["performance"]
            else:
                reward = 0.0

            # Child state — same as AutoCorrInequalityEnv._create_next_state()
            child_state = None
            result_construction = outs.get("result_construction")
            if correctness > 0 and result_construction is not None:
                child_state = InequalitiesState(
                    timestep=step,
                    construction=result_construction,
                    code=parsed_code,
                    value=outs["performance"],
                    observation=outs.get("stdout", ""),
                )

            return {
                "reward": reward,
                "child_state": child_state,
                "correctness": correctness,
                "metrics": {
                    "score": outs.get("score", 0.0),
                    "correctness": correctness,
                    "performance": outs.get("performance", 0.0),
                },
            }

    elif cfg.env == "cp":
        from ug_ttt.recipes.ttt.env_cp import build_cp_prompt, verify_cp, parse_cp_problem_idx

        n, _ = parse_cp_problem_idx(cfg.problem_idx)

        def prompt_fn(state: CirclePackingState) -> str:
            return build_cp_prompt(state, n=n)

        def process_fn(generated_text: str, parent_state: CirclePackingState, step: int) -> Dict[str, Any]:
            parsed_code = last_codeblock_postprocess(generated_text, ["python"], keep_separators=True)
            if not parsed_code or not parsed_code.strip():
                return {"reward": 0.0, "child_state": None, "correctness": 0.0, "metrics": {}}

            outs = verify_cp(
                parsed_code, step,
                num_cpus_per_task=cfg.num_cpus_per_task,
                problem_idx=n,
                eval_timeout=cfg.eval_timeout,
                log_path=cfg.log_path,
                state=parent_state,
            )
            correctness = outs.get("correctness", 0.0)

            # Reward — same as CirclePackingEnv._compute_reward()
            sum_radii = outs.get("sum_radii", outs.get("score", 0.0))
            reward = sum_radii if correctness > 0 else 0.0

            # Child state — same as CirclePackingEnv._create_next_state()
            child_state = None
            if correctness > 0:
                parent_values = ([parent_state.value] + parent_state.parent_values
                                 if parent_state.value is not None else [])
                child_state = CirclePackingState(
                    timestep=step,
                    construction=outs.get("result_construction"),
                    code=parsed_code,
                    value=sum_radii,
                    parent_values=parent_values,
                    observation=outs.get("stdout", ""),
                )

            return {
                "reward": reward,
                "child_state": child_state,
                "correctness": correctness,
                "metrics": {
                    "score": outs.get("score", 0.0),
                    "correctness": correctness,
                    "sum_radii": sum_radii,
                },
            }

    elif cfg.env == "erdos":
        from ug_ttt.recipes.ttt.env_erdos import build_erdos_prompt, verify_erdos

        def prompt_fn(state: ErdosState) -> str:
            return build_erdos_prompt(
                state,
                budget_s=cfg.budget_s,
                num_cpus_per_task=cfg.num_cpus_per_task,
                problem_idx=cfg.problem_idx,
            )

        def process_fn(generated_text: str, parent_state: ErdosState, step: int) -> Dict[str, Any]:
            parsed_code = last_codeblock_postprocess(generated_text, ["python"], keep_separators=True)
            if not parsed_code or not parsed_code.strip():
                return {"reward": 0.0, "child_state": None, "correctness": 0.0, "metrics": {}}

            outs = verify_erdos(
                parsed_code, step,
                num_cpus_per_task=cfg.num_cpus_per_task,
                eval_timeout=cfg.eval_timeout,
                log_path=cfg.log_path,
                state=parent_state,
            )
            correctness = outs.get("correctness", 0.0)
            performance = outs.get("performance")

            # Reward — same as ErdosMinOverlapEnv._compute_reward()
            if cfg.adv_estimator in ("entropic", "entropic_adaptive_beta"):
                current_bound = -performance if performance is not None else float('inf')
                reward = 1/current_bound if (correctness > 0 and current_bound > 0) else 0.0
            else:
                reward = outs["score"]

            # Child state — same as ErdosMinOverlapEnv._create_next_state()
            child_state = None
            if correctness > 0 and performance is not None:
                parent_values = ([parent_state.value] + parent_state.parent_values
                                 if parent_state.value is not None else [])
                child_state = ErdosState(
                    timestep=step,
                    code=parsed_code,
                    value=performance,
                    c5_bound=outs.get("c5_bound"),
                    construction=outs.get("construction"),
                    parent_values=parent_values,
                    observation=outs.get("stdout", ""),
                )

            return {
                "reward": reward,
                "child_state": child_state,
                "correctness": correctness,
                "metrics": {
                    "score": outs.get("score", 0.0),
                    "correctness": correctness,
                    "c5_bound": outs.get("c5_bound"),
                    "performance": performance,
                },
            }

    else:
        raise ValueError(f"Unsupported env: {cfg.env}. Supported: ac1, ac2, cp, erdos")

    # ── Create sampler (with resume support) ─────────────────────────────
    resume_step = None
    checkpoint_file = os.path.join(cfg.log_path, "last_epoch.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            resume_step = int(f.read().strip())
    sampler = create_sampler(
        sampler_type=cfg.sampler_type,
        log_path=cfg.log_path,
        env_type=cfg.env,
        initial_exp_type=cfg.initial_exp_type,
        batch_size=cfg.groups_per_batch,
        resume_step=resume_step,
    )

    main(
        cfg=cfg,
        prompt_fn=prompt_fn,
        process_fn=process_fn,
        sampler=sampler,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cli_main()
