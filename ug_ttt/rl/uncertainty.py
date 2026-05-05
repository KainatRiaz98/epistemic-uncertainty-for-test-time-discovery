"""
Uncertainty metrics for epistemic exploration in UG-TTT.

Primary metric: True MI computed from full vocabulary distributions during
ensemble scoring (H[mixture] - E_k[H[member]]). This captures distributional
disagreement across the entire vocabulary, not just at generated tokens.

Ablation baselines: Jensen-gap MI, variance, and token-level predictive entropy
computed from per-token logprobs at generated tokens only.

Reference: He et al., "Survey of uncertainty estimation in LLMs" (2026), Eq. 6
           Malinin & Gales, "Uncertainty estimation in autoregressive structured
           prediction" (ICLR 2020)
"""

import math
from typing import Dict, Optional

import torch


# ── Shared aggregation helper ─────────────────────────────────────────────────


def _top_frac_mean(vals: torch.Tensor, top_frac: float = 0.07) -> torch.Tensor:
    """Average the highest top_frac fraction of values.

    Shared by compute_true_mi and the ablation baselines so that all metrics
    can be aggregated identically for a fair apples-to-apples comparison.
    """
    T = vals.numel()
    k = max(1, int(T * top_frac))
    if T <= k:
        return vals.mean()
    topk_vals, _ = torch.topk(vals, k)
    return topk_vals.mean()


# ── Phase breakdown of top-7% tokens ─────────────────────────────────────────


def compute_top7_phase_breakdown(
    per_token_mi: torch.Tensor,
    prompt_len: int,
    phase1_count: int,
    prefill_count: int = 0,
    top_frac: float = 0.07,
) -> Dict[str, float]:
    """
    For the top-7% highest-MI tokens, compute what fraction came from
    Phase 1 (thinking tokens) vs Phase 2 (code generation tokens).

    Token layout in per_token_mi (index t predicts full_tokens[t+1]):
      prompt tokens      → t ∈ [0,          prompt_len-1)
      thinking (Phase 1) → t ∈ [prompt_len-1, prompt_len-1+phase1_count)
      teacher prefill    → t ∈ [prompt_len-1+phase1_count,
                                 prompt_len-1+phase1_count+prefill_count)
      code gen (Phase 2) → t ∈ [prompt_len-1+phase1_count+prefill_count, T-1)

    When two_phase_sampling is off, phase2_count==0 and phase1_count covers all
    generated tokens → top7_pct_code will be 0.0.

    Returns dict with:
        top7_pct_thinking  fraction of top-7% tokens that are Phase 1
        top7_pct_code      fraction of top-7% tokens that are Phase 2
        top7_mi_thinking   mean MI of Phase-1 tokens within top-7% (0 if none)
        top7_mi_code       mean MI of Phase-2 tokens within top-7% (0 if none)
    """
    T_minus_1 = per_token_mi.numel()
    k = max(1, int(T_minus_1 * top_frac))
    if T_minus_1 <= k:
        topk_indices = torch.arange(T_minus_1, device=per_token_mi.device)
    else:
        _, topk_indices = torch.topk(per_token_mi, k)

    p1_start = prompt_len - 1
    p1_end   = prompt_len - 1 + phase1_count          # exclusive
    p2_start = prompt_len - 1 + phase1_count + prefill_count  # inclusive

    in_p1 = (topk_indices >= p1_start) & (topk_indices < p1_end)
    in_p2 = topk_indices >= p2_start

    n_top7     = topk_indices.numel()
    n_thinking = int(in_p1.sum().item())
    n_code     = int(in_p2.sum().item())

    return {
        "top7_pct_thinking": n_thinking / n_top7 if n_top7 > 0 else 0.0,
        "top7_pct_code":     n_code     / n_top7 if n_top7 > 0 else 0.0,
        "top7_mi_thinking":  float(per_token_mi[topk_indices[in_p1]].mean().item())
                             if n_thinking > 0 else 0.0,
        "top7_mi_code":      float(per_token_mi[topk_indices[in_p2]].mean().item())
                             if n_code     > 0 else 0.0,
    }


# ── Primary metric ───────────────────────────────────────────────────────────


def compute_true_mi(per_token_mi: torch.Tensor, top_frac: float = 0.07) -> torch.Tensor:
    """
    Aggregate per-token true MI by averaging the top-7% highest-MI positions.

    True MI is the Bayesian decomposition of predictive uncertainty:

        MI(y; Θ|x) = H[p̄(·|x)] - E_k[H[p_k(·|x)]]

    where H is entropy over the full vocabulary V at each position,
    p̄(v) = (1/K) Σ_k p_k(v) is the arithmetic mixture, and the
    expectation is over K ensemble members.

    Per-position true MI is computed during the chunked LM head pass in
    LoRAEnsemble._chunked_logprobs(compute_mi=True), which has access to
    the full (K, chunk, V) logit tensor before it is discarded.

    Rather than averaging all T positions (which buries signal in boilerplate
    tokens), we concentrate on the high-uncertainty minority. Wang et al.
    "Beyond the 80/20 Rule" (2025) argues the RL signal should be
    concentrated in high-entropy minority tokens. AdaDec (He et al. 2025)
    empirically measures ~6.75% of decoding steps as genuinely uncertain in
    code generation (range 0.85%–12.17%); 7% sits at the empirical mean.

    Top-7% adaptive cutoff: for a 10k-token sequence this scores ~700
    positions. Scales correctly across the 7k–15k token range of code
    generation, unlike a fixed top-k count.

    Args:
        per_token_mi: (T,) tensor of per-position true MI values,
            pre-computed from full vocabulary distributions.
        top_frac: fraction of highest-MI positions to average (default 0.07).

    Returns:
        Scalar: top-7% mean true MI (non-negative by construction).
    """
    return _top_frac_mean(per_token_mi, top_frac)


# ── Ablation baselines (computed from per-token logprobs at generated tokens) ─


def compute_rmi(
    ensemble_logprobs: torch.Tensor,
    top_frac: Optional[float] = None,
) -> torch.Tensor:
    """
    Jensen-gap MI estimate from ensemble logprobs (ablation baseline).

    Approximation that only considers the probability of the generated token
    at each position, NOT the full vocabulary distribution:

        JG(t) = log p̄(y_t) - E_k[log p_k(y_t)]

    where p̄(y_t) = (1/K) Σ_k p_k(y_t) is the arithmetic mean probability.
    Non-negative by Jensen's inequality (log is concave).

    This misses distributional disagreement at non-generated tokens. Use
    compute_true_mi for full-distribution epistemic uncertainty.

    Args:
        ensemble_logprobs: (K, T) tensor. Entry [k, t] is
            log p_k(y_t | y_{<t}, x) for ensemble member k.
        top_frac: if set, aggregate via top-frac mean (same as compute_true_mi)
            instead of the plain mean, for a fair ablation comparison.

    Returns:
        Scalar tensor: per-token Jensen-gap MI aggregated by mean or top-frac.
    """
    K = ensemble_logprobs.shape[0]
    # Numerically stable: log((1/K) Σ_k p_k) = logsumexp(log_p_k, dim=0) - log(K)
    log_avg_probs = torch.logsumexp(ensemble_logprobs, dim=0) - math.log(K)
    mean_member_logprobs = ensemble_logprobs.mean(dim=0)
    mi_per_token = log_avg_probs - mean_member_logprobs
    if top_frac is not None:
        return _top_frac_mean(mi_per_token, top_frac)
    return mi_per_token.mean()


def compute_variance(
    ensemble_logprobs: torch.Tensor,
    top_frac: Optional[float] = None,
) -> torch.Tensor:
    """
    Variance of per-token logprobs across ensemble members.
    Ablation baseline: simpler proxy for ensemble disagreement.

    Args:
        ensemble_logprobs: (K, T) tensor.
        top_frac: if set, aggregate via top-frac mean instead of plain mean.

    Returns:
        Scalar: variance across tokens (mean or top-frac aggregated).
    """
    per_token_var = ensemble_logprobs.var(dim=0)
    if top_frac is not None:
        return _top_frac_mean(per_token_var, top_frac)
    return per_token_var.mean()


def compute_predictive_entropy(
    ensemble_logprobs: torch.Tensor,
    top_frac: Optional[float] = None,
) -> torch.Tensor:
    """
    Entropy of the mixture predictive at the generated token (approximate).
    Ablation baseline: captures total uncertainty (epistemic + aleatoric)
    but only at the generated token, not the full vocabulary.

    H_approx(t) = -p̄(y_t) log p̄(y_t)

    Args:
        ensemble_logprobs: (K, T) tensor.
        top_frac: if set, aggregate via top-frac mean instead of plain mean.

    Returns:
        Scalar: token-level predictive entropy (mean or top-frac aggregated).
    """
    K = ensemble_logprobs.shape[0]
    log_avg_probs = torch.logsumexp(ensemble_logprobs, dim=0) - math.log(K)
    avg_probs = log_avg_probs.exp()
    entropy = -(avg_probs * log_avg_probs)
    if top_frac is not None:
        return _top_frac_mean(entropy, top_frac)
    return entropy.mean()


# Logprob-based metrics (take (K, T) ensemble_logprobs).
# compute_true_mi has a different signature (takes pre-computed (T,) MI)
# and is handled separately by callers.
UNCERTAINTY_FNS = {
    "rmi": compute_rmi,
    "variance": compute_variance,
    "predictive_entropy": compute_predictive_entropy,
}
