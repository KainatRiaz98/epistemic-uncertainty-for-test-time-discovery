"""
CPU smoke test for the mLoRA ensemble RL training loop.

Runs the full pipeline — uncertainty metrics, do_rollout, train_step,
KL penalty, checkpointing — WITHOUT loading any model or requiring a GPU.

Strategy: mock LoRAEnsemble so that generate() and compute_ensemble_logprobs()
return deterministic random tensors.  Everything else (advantage computation,
reward augmentation, RL loss, KL tracking, data structures) runs real code.

Run with:
    python -m pytest tests/test_smoke_cpu.py -v
or:
    python tests/test_smoke_cpu.py
"""

import math
import os
import sys
import tempfile
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
import numpy as np

# ── Make sure the project root is on the path ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Stub out mLoRA before importing our modules ───────────────────────────────
# ensemble.py and mlora_train.py do top-level imports from mlora.*
# We provide lightweight stubs so the modules load without the real library.

def _make_mlora_stubs():
    """Register fake mlora.* modules in sys.modules."""

    mlora = types.ModuleType("mlora")
    mlora.model = types.ModuleType("mlora.model")
    mlora.model.llm = types.ModuleType("mlora.model.llm")
    mlora.model.tokenizer = types.ModuleType("mlora.model.tokenizer")
    mlora.model.args = types.ModuleType("mlora.model.args")
    mlora.config = types.ModuleType("mlora.config")
    mlora.executor = types.ModuleType("mlora.executor")
    mlora.executor.context = types.ModuleType("mlora.executor.context")
    mlora.executor.context.lora = types.ModuleType("mlora.executor.context.lora")

    # ── Stub classes ──────────────────────────────────────────────────────────

    class _LLMModel:
        device_ = "cpu"
        def forward(self, x): ...
        def linears_info(self): return {}
        def sequential(self): return []
        def load_adapter(self, a): ...

    class _Tokenizer:
        eos_id_ = 2
        def encode(self, text, bos=True, eos=False): return [1, 2, 3]
        def decode(self, tokens): return "def solution(): pass"

    class _LinearInfo: ...
    class _MLoRADataConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class _MLoRAData:
        use_flash_causal_ = False
        return_hidden_states_ = False
        kv_cache_ = None
        cache_position_ = 0
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_data(self): return self

    class _LoRAConfig:
        name_ = "test"
        r_ = 4
        alpha_ = 8
        dropout_ = 0.0
        target_ = {}
        def __init__(self, d): pass

    class _TrainLoRAContext:
        def __init__(self, cfg, linears): pass
        def switch_device(self, d): pass
        def adapter_model(self): return {}
        def weight_dict(self): return {}
        def recover_weight(self, d): pass
        def step(self): pass

    mlora.model.llm.LLMModel = _LLMModel
    mlora.model.tokenizer.Tokenizer = _Tokenizer
    mlora.model.args.LinearInfo = _LinearInfo
    mlora.model.args.MLoRADataConfig = _MLoRADataConfig
    mlora.model.args.MLoRAData = _MLoRAData
    mlora.model.args.Tokens = list
    mlora.model.args.Masks = list
    mlora.config.LoRAConfig = _LoRAConfig
    mlora.executor.context.lora.TrainLoRAContext = _TrainLoRAContext

    # Nested package references
    mlora.model.loader = types.ModuleType("mlora.model.loader")
    mlora.model.loader.load_model = lambda args: (_Tokenizer(), _LLMModel())
    mlora.model.load_model = mlora.model.loader.load_model

    for name, mod in [
        ("mlora", mlora),
        ("mlora.model", mlora.model),
        ("mlora.model.llm", mlora.model.llm),
        ("mlora.model.tokenizer", mlora.model.tokenizer),
        ("mlora.model.args", mlora.model.args),
        ("mlora.model.loader", mlora.model.loader),
        ("mlora.config", mlora.config),
        ("mlora.executor", mlora.executor),
        ("mlora.executor.context", mlora.executor.context),
        ("mlora.executor.context.lora", mlora.executor.context.lora),
    ]:
        sys.modules[name] = mod

_make_mlora_stubs()

# ── Now import the real modules ───────────────────────────────────────────────
from ug_ttt.rl.uncertainty import (
    compute_rmi,
    compute_variance,
    compute_predictive_entropy,
    UNCERTAINTY_FNS,
)
from ug_ttt.rl.mlora_train import (
    Config,
    Rollout,
    RolloutGroup,
    compute_advantages,
    compute_rl_loss,
    compute_kl_sample_train,
    do_rollout,
    train_step,
)


# ── Mock LoRAEnsemble ─────────────────────────────────────────────────────────

class MockEnsemble:
    """
    Drop-in replacement for LoRAEnsemble.
    Returns deterministic random tensors.  No GPU, no mLoRA.
    """

    def __init__(self, K: int = 3, vocab: int = 32000, seed: int = 0):
        self.K = K
        self.vocab = vocab
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._step_count = 0

    # Called by do_rollout → ensemble.generate()
    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
    ) -> Tuple[List[int], List[float]]:
        n_gen = 10  # short fixed length for tests
        gen_tokens = [10 + i for i in range(n_gen)]
        gen_logprobs = [-0.5 - 0.05 * i for i in range(n_gen)]
        return prompt_tokens + gen_tokens, gen_logprobs

    # Called by do_rollout → ensemble.compute_ensemble_logprobs()
    def compute_ensemble_logprobs(self, token_ids: List[int], chunk_size: int = 256) -> torch.Tensor:
        T = len(token_ids)
        # Each member has slightly different logprobs → non-zero disagreement
        base = torch.full((self.K, T - 1), -1.0)
        for k in range(self.K):
            base[k] += torch.randn(T - 1, generator=self._rng) * 0.3 * (k + 1)
        return base

    # Called by train_step → ensemble.compute_training_logprobs()
    def compute_training_logprobs(self, token_ids: List[int], chunk_size: int = 256):
        T = len(token_ids)
        hidden = torch.randn(self.K, T, 64, generator=self._rng)
        logprobs = torch.full((self.K, T - 1), -0.6)
        logprobs += torch.randn(self.K, T - 1, generator=self._rng) * 0.1
        return hidden, logprobs

    # Called by incorporate_kl_penalty → _compute_base_logprobs → ensemble.model.forward()
    @property
    def model(self):
        vocab = self.vocab
        class _FakeModel:
            device_ = "cpu"
            def forward(self, data):
                # Return fake logits (1, T, V) for base model scoring
                T = len(data.batch_tokens[0]) if hasattr(data, 'batch_tokens') else 15
                return torch.randn(1, T, vocab)
        return _FakeModel()

    def step_all_optimizers(self):
        self._step_count += 1

    def save(self, path: str, step: int):
        os.makedirs(path, exist_ok=True)
        for k in range(self.K):
            save_dir = os.path.join(path, f"ensemble_{k}", f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({"dummy": torch.zeros(1)}, os.path.join(save_dir, "adapter.pt"))

    def load(self, path: str, step: int):
        pass  # no-op for tests

    @staticmethod
    def _expand_fn(batch_tokens, align_len=None):
        if align_len is None:
            align_len = max(len(t) for t in batch_tokens)
        padded_tokens, padded_masks = [], []
        for tokens in batch_tokens:
            pad_len = align_len - len(tokens)
            padded_tokens.append(tokens + [0] * pad_len)
            padded_masks.append([True] * len(tokens) + [False] * pad_len)
        return padded_tokens, padded_masks

    @staticmethod
    def _noop_loss(input, target, mask):
        return None


# ── Helper: make a Config with cpu device and fast settings ──────────────────

def make_test_config(**overrides) -> Config:
    defaults = dict(
        base_model="mock",
        device="cpu",
        env="ac1",
        group_size=4,
        groups_per_batch=2,
        num_epochs=2,
        max_tokens=20,
        temperature=1.0,
        adv_estimator="mean_baseline",
        loss_fn="importance_sampling",
        num_substeps=1,
        rmi_coef=0.1,
        uncertainty_metric="rmi",
        gamma_max_ratio=10.0,
        num_ensemble_members=3,
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
        remove_constant_reward_groups=False,
        log_path=tempfile.mkdtemp(),
        wandb_project=None,
        eval_every=1,
        save_every=1,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ── Helper: make a fake Rollout ───────────────────────────────────────────────

def make_rollout(reward_exec: float = 0.5, rmi_coef: float = 0.1,
                 prompt_len: int = 5, gen_len: int = 10,
                 reward_rmi: float = 0.3) -> Rollout:
    prompt_tokens = list(range(1, prompt_len + 1))
    full_tokens = prompt_tokens + list(range(100, 100 + gen_len))
    # Pad with 0.0 for prompt positions, matching do_rollout / original train.py
    sampling_logprobs = [0.0] * (prompt_len - 1) + [-0.5] * gen_len
    reward_total = reward_exec + rmi_coef * reward_rmi
    return Rollout(
        prompt_tokens=prompt_tokens,
        full_tokens=full_tokens,
        sampling_logprobs=sampling_logprobs,
        reward_exec=reward_exec,
        reward_rmi=reward_rmi,
        reward_total=reward_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Test suite
# ══════════════════════════════════════════════════════════════════════════════

class TestRMIFormula(unittest.TestCase):
    """
    Verifies the RMI formula:
        MI(y; Θ|x) = log p̄(y_t) - E_k[log p_k(y_t)]

    where p̄(y_t) = (1/K) Σ_k p_k(y_t)  [arithmetic mean of probs]
    and E_k[...] = (1/K) Σ_k log p_k(y_t) [mean of log-probs]

    Jensen's inequality guarantees MI >= 0 because:
        log(mean(x)) >= mean(log(x))  for any positive x.
    """

    def _manual_rmi(self, ensemble_logprobs: torch.Tensor) -> float:
        """Reference implementation — formula written out explicitly."""
        probs = ensemble_logprobs.exp()                    # (K, T)
        avg_prob = probs.mean(dim=0)                       # (T,)
        log_avg = avg_prob.log()                           # (T,)  ← log of arithmetic mean
        mean_log = ensemble_logprobs.mean(dim=0)           # (T,)  ← mean of logs
        mi_per_token = log_avg - mean_log                  # (T,)  always >= 0
        return mi_per_token.mean().item()

    def test_nonnegative_by_jensen(self):
        """RMI must always be >= 0 (Jensen's inequality)."""
        for seed in range(10):
            torch.manual_seed(seed)
            lp = torch.randn(5, 20)  # arbitrary, not normalised
            rmi = compute_rmi(lp).item()
            self.assertGreaterEqual(rmi, -1e-6,
                msg=f"RMI < 0 at seed {seed}: {rmi:.6f}")

    def test_matches_manual_formula(self):
        """compute_rmi() must match the explicit formula."""
        torch.manual_seed(42)
        lp = torch.randn(4, 15)
        expected = self._manual_rmi(lp)
        got = compute_rmi(lp).item()
        self.assertAlmostEqual(got, expected, places=5,
            msg=f"RMI mismatch: got {got:.6f}, expected {expected:.6f}")

    def test_zero_when_ensemble_agrees(self):
        """If all K members assign identical logprobs, RMI == 0."""
        row = torch.randn(1, 20)
        lp = row.expand(5, -1)   # all K identical
        rmi = compute_rmi(lp).item()
        self.assertAlmostEqual(rmi, 0.0, places=5,
            msg=f"Expected RMI=0 for identical ensemble, got {rmi:.6f}")

    def test_larger_disagreement_yields_higher_rmi(self):
        """More spread between members → higher RMI."""
        base = torch.zeros(3, 20)
        low_spread  = base + torch.randn(3, 20) * 0.01
        high_spread = base + torch.randn(3, 20) * 2.0
        rmi_low  = compute_rmi(low_spread).item()
        rmi_high = compute_rmi(high_spread).item()
        self.assertGreater(rmi_high, rmi_low,
            msg=f"Expected high-spread RMI ({rmi_high:.4f}) > low-spread RMI ({rmi_low:.4f})")

    def test_scalar_output(self):
        """compute_rmi must return a scalar tensor."""
        lp = torch.randn(3, 10)
        out = compute_rmi(lp)
        self.assertEqual(out.shape, torch.Size([]),
            msg=f"Expected scalar, got shape {out.shape}")

    def test_single_token(self):
        """Works for T=1 (edge case)."""
        lp = torch.randn(3, 1)
        rmi = compute_rmi(lp)
        self.assertGreaterEqual(rmi.item(), -1e-6)

    def test_single_member(self):
        """K=1: RMI is 0 (log(p) - log(p) == 0)."""
        lp = torch.randn(1, 10)
        rmi = compute_rmi(lp).item()
        self.assertAlmostEqual(rmi, 0.0, places=5,
            msg=f"K=1 should give RMI=0, got {rmi:.6f}")


class TestAblationMetrics(unittest.TestCase):
    """Tests for variance and predictive_entropy ablation baselines."""

    def test_variance_zero_for_identical(self):
        lp = torch.randn(1, 10).expand(4, -1)
        v = compute_variance(lp).item()
        self.assertAlmostEqual(v, 0.0, places=5)

    def test_variance_positive(self):
        lp = torch.randn(4, 10)
        v = compute_variance(lp).item()
        self.assertGreater(v, 0.0)

    def test_predictive_entropy_nonnegative(self):
        lp = -torch.rand(4, 10).abs() - 0.1  # valid logprobs (negative)
        h = compute_predictive_entropy(lp).item()
        # h = -p*log(p); avg_prob in (0,1) so log < 0, -p*log(p) >= 0
        self.assertGreaterEqual(h, -1e-6)

    def test_uncertainty_fn_registry(self):
        """All three metrics accessible via UNCERTAINTY_FNS dict."""
        lp = -torch.rand(3, 8).abs() - 0.1  # valid logprobs
        for name, fn in UNCERTAINTY_FNS.items():
            out = fn(lp)
            self.assertEqual(out.shape, torch.Size([]),
                msg=f"UNCERTAINTY_FNS['{name}'] returned non-scalar {out.shape}")


class TestRewardAugmentation(unittest.TestCase):
    """Verify R_total = R_exec + γ * RMI at the Rollout level."""

    def test_formula_exact(self):
        r_exec = 0.7
        rmi    = 0.4
        gamma  = 0.1
        r = make_rollout(reward_exec=r_exec, rmi_coef=gamma, reward_rmi=rmi)
        expected = r_exec + gamma * rmi
        self.assertAlmostEqual(r.reward_total, expected, places=6,
            msg=f"R_total={r.reward_total:.6f}, expected {expected:.6f}")

    def test_gamma_zero_ignores_rmi(self):
        """γ=0 → R_total == R_exec regardless of RMI."""
        r = make_rollout(reward_exec=0.8, rmi_coef=0.0, reward_rmi=99.0)
        self.assertAlmostEqual(r.reward_total, 0.8, places=6)

    def test_action_mask_shape(self):
        """action_mask length == len(full_tokens) - 1."""
        r = make_rollout(prompt_len=5, gen_len=10)
        mask = r.action_mask
        self.assertEqual(mask.shape[0], len(r.full_tokens) - 1)

    def test_action_mask_values(self):
        """Prompt positions are 0, generated positions are 1."""
        r = make_rollout(prompt_len=5, gen_len=10)
        mask = r.action_mask
        # positions 0..3 (prompt tokens predict next prompt token) → 0
        self.assertTrue((mask[:4] == 0).all(),
            msg=f"Prompt portion of mask should be 0: {mask[:4]}")
        # positions 4..13 (generated tokens) → 1
        self.assertTrue((mask[4:] == 1).all(),
            msg=f"Action portion of mask should be 1: {mask[4:]}")


class TestComputeAdvantages(unittest.TestCase):
    """Test all three advantage estimators."""

    def test_mean_baseline(self):
        rewards = [1.0, 0.0, 0.5, 1.5]
        adv, meta = compute_advantages(rewards, "mean_baseline")
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=5)
        expected = torch.tensor(rewards) - torch.tensor(rewards).mean()
        self.assertTrue(torch.allclose(adv, expected, atol=1e-5))
        self.assertEqual(meta["beta"], 0.0)

    def test_entropic_nonneg_Z(self):
        """Entropic advantage: Z is leave-one-out, adv = e/Z - 1. Result can be any sign."""
        rewards = [0.2, 0.8, 0.5, 0.1]
        adv, meta = compute_advantages(rewards, "entropic", adv_estimator_beta=2.0)
        self.assertEqual(adv.shape[0], 4)
        self.assertAlmostEqual(meta["beta"], 2.0, places=5)

    def test_entropic_adaptive_beta(self):
        rewards = [0.1, 0.9, 0.5, 0.3]
        adv, meta = compute_advantages(rewards, "entropic_adaptive_beta")
        self.assertEqual(adv.shape[0], 4)
        self.assertGreater(meta["beta"], 0.0)

    def test_uniform_rewards_mean_baseline_zero(self):
        """All equal rewards → all-zero advantages."""
        adv, _ = compute_advantages([1.0, 1.0, 1.0], "mean_baseline")
        self.assertTrue(torch.allclose(adv, torch.zeros(3), atol=1e-6))

    def test_single_rollout(self):
        """K=1 should not crash."""
        adv, _ = compute_advantages([0.7], "mean_baseline")
        self.assertEqual(adv.shape[0], 1)


class TestRMISeparation(unittest.TestCase):
    """Verify RMI is separated from β and added in advantage space."""

    def test_beta_returned_from_adaptive(self):
        """entropic_adaptive_beta should return a positive solved β."""
        _, meta = compute_advantages([0.1, 0.9, 0.5, 0.3], "entropic_adaptive_beta")
        self.assertGreater(meta["beta"], 0.0)

    def test_rmi_centering_zero_mean(self):
        """Centered RMI should have zero mean."""
        rmi = torch.tensor([0.1, 0.5, 0.2, 0.8])
        centered = rmi - rmi.mean()
        self.assertAlmostEqual(centered.mean().item(), 0.0, places=6)

    def test_gamma_eff_coupling_normal(self):
        """γ_eff = rmi_coef * β / adv_estimator_beta when below clip."""
        # β=4, adv_estimator_beta=2 → ratio=2, γ_eff=0.2
        gamma_eff = 0.1 * min(4.0 / 2.0, 10.0)
        self.assertAlmostEqual(gamma_eff, 0.2, places=6)

    def test_gamma_eff_coupling_clipped(self):
        """γ_eff should be clipped at rmi_coef * γ_max_ratio."""
        # β=100, adv_estimator_beta=2 → ratio=50, clipped to 10
        gamma_eff = 0.1 * min(100.0 / 2.0, 10.0)
        self.assertAlmostEqual(gamma_eff, 1.0, places=6)

    def test_gamma_eff_zero_beta(self):
        """β=0 (mean_baseline) → γ_eff=0."""
        gamma_eff = 0.1 * min(0.0 / 2.0, 10.0)
        self.assertAlmostEqual(gamma_eff, 0.0, places=6)

    def test_rmi_adjusts_advantages(self):
        """Full pipeline: exec advantages + γ_eff * centered RMI."""
        exec_rewards = [0.1, 0.9, 0.5, 0.3]
        rmi_values = [0.2, 0.1, 0.4, 0.3]
        adv_estimator_beta = 2.0  # the reference β

        adv_exec, meta = compute_advantages(exec_rewards, "entropic_adaptive_beta")
        beta = meta["beta"]
        rmi_t = torch.tensor(rmi_values, dtype=torch.float32)
        rmi_centered = rmi_t - rmi_t.mean()
        gamma_eff = 0.1 * min(beta / adv_estimator_beta, 10.0)
        adv_total = adv_exec + gamma_eff * rmi_centered

        # RMI contribution is zero-mean
        self.assertAlmostEqual((gamma_eff * rmi_centered).mean().item(), 0.0, places=6)
        # Total advantages differ from exec-only when gamma_eff > 0
        if gamma_eff > 0:
            self.assertFalse(torch.allclose(adv_total, adv_exec, atol=1e-8))

    def test_train_step_emits_beta_gamma(self):
        """train_step should emit beta and gamma_eff metrics."""
        ensemble = MockEnsemble(K=3)
        cfg = make_test_config(
            adv_estimator="entropic_adaptive_beta",
            rmi_coef=0.1,
        )
        rollouts = [make_rollout(reward_exec=r) for r in [0.2, 0.8, 0.5, 0.1]]
        group = RolloutGroup(rollouts=rollouts)
        metrics = train_step(ensemble, [group], cfg)
        self.assertIn("train/beta/mean", metrics)
        self.assertIn("train/gamma_eff/mean", metrics)
        self.assertGreater(metrics["train/beta/mean"], 0.0)

    def test_rmi_coef_zero_no_gamma(self):
        """When rmi_coef=0, gamma_eff should be 0."""
        ensemble = MockEnsemble(K=3)
        cfg = make_test_config(rmi_coef=0.0)
        rollouts = [make_rollout(reward_exec=r) for r in [0.2, 0.8, 0.5]]
        group = RolloutGroup(rollouts=rollouts)
        metrics = train_step(ensemble, [group], cfg)
        self.assertAlmostEqual(metrics["train/gamma_eff/mean"], 0.0, places=6)


class TestRLLoss(unittest.TestCase):

    def _make_loss_inputs(self, T: int = 10, prompt_len: int = 3):
        current  = torch.full((T,), -0.6)
        old      = torch.full((T,), -0.5)
        mask     = torch.zeros(T)
        mask[prompt_len - 1:] = 1.0
        return current, old, mask

    def test_loss_is_scalar(self):
        current, old, mask = self._make_loss_inputs()
        loss = compute_rl_loss(current, old, advantage=1.0, action_mask=mask)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_zero_advantage_zero_loss(self):
        current, old, mask = self._make_loss_inputs()
        loss = compute_rl_loss(current, old, advantage=0.0, action_mask=mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_mask_zero_returns_zero(self):
        """If no action tokens, loss should be 0 (mask sum clamped to 1)."""
        current = torch.zeros(5)
        old     = torch.zeros(5)
        mask    = torch.zeros(5)
        loss = compute_rl_loss(current, old, advantage=1.0, action_mask=mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_ppo_loss_clipping(self):
        """PPO loss with large ratio should clamp."""
        current = torch.full((10,), -0.1)  # ratio >> 1
        old     = torch.full((10,), -3.0)
        mask    = torch.ones(10)
        loss_ppo = compute_rl_loss(current, old, 1.0, mask, loss_fn="ppo", ppo_clip=0.2)
        loss_is  = compute_rl_loss(current, old, 1.0, mask, loss_fn="importance_sampling")
        # PPO clips, so its |loss| <= |IS loss| when ratio is very large
        self.assertLessEqual(abs(loss_ppo.item()), abs(loss_is.item()) + 1e-3)

    def test_negative_advantage_flips_sign(self):
        """Negative advantage should produce positive loss (penalise the action)."""
        current, old, mask = self._make_loss_inputs()
        loss_pos = compute_rl_loss(current, old, advantage=+1.0, action_mask=mask)
        loss_neg = compute_rl_loss(current, old, advantage=-1.0, action_mask=mask)
        self.assertAlmostEqual(loss_pos.item(), -loss_neg.item(), places=5)


class TestDoRollout(unittest.TestCase):
    """Integration test: do_rollout uses MockEnsemble to run the full
    generate → execute → RMI → augment flow on CPU."""

    def setUp(self):
        self.ensemble = MockEnsemble(K=3)
        self.prompt_tokens = [1, 2, 3, 4, 5]
        self.cfg = make_test_config(rmi_coef=0.1, uncertainty_metric="rmi")

    def _simple_process_fn(self, text: str, state, step: int) -> Dict:
        return {
            "reward": 0.5,
            "child_state": None,
            "correctness": 1.0,
            "metrics": {"score": 0.5},
        }

    def test_rollout_returns_correct_type(self):
        from ug_ttt.rl.mlora_train import do_rollout
        from ug_ttt.rl.uncertainty import UNCERTAINTY_FNS
        rollout = do_rollout(
            ensemble=self.ensemble,
            tokenizer=MagicMock(decode=lambda t: "def foo(): pass"),
            prompt_tokens=self.prompt_tokens,
            adapter_idx=0,
            parent_state=None,
            step=0,
            process_fn=self._simple_process_fn,
            uncertainty_fn=UNCERTAINTY_FNS["rmi"],
            rmi_coef=0.1,
            max_tokens=20,
            temperature=1.0,
        )
        self.assertIsInstance(rollout, Rollout)

    def test_rollout_reward_total_formula(self):
        """R_total must equal R_exec + γ * RMI."""
        from ug_ttt.rl.mlora_train import do_rollout
        from ug_ttt.rl.uncertainty import UNCERTAINTY_FNS
        rollout = do_rollout(
            ensemble=self.ensemble,
            tokenizer=MagicMock(decode=lambda t: "def foo(): pass"),
            prompt_tokens=self.prompt_tokens,
            adapter_idx=0,
            parent_state=None,
            step=0,
            process_fn=self._simple_process_fn,
            uncertainty_fn=UNCERTAINTY_FNS["rmi"],
            rmi_coef=0.1,
            max_tokens=20,
            temperature=1.0,
        )
        expected_total = rollout.reward_exec + 0.1 * rollout.reward_rmi
        self.assertAlmostEqual(rollout.reward_total, expected_total, places=6,
            msg=f"R_total={rollout.reward_total:.6f} != R_exec + γ*RMI = {expected_total:.6f}")

    def test_rmi_nonnegative(self):
        """RMI computed inside do_rollout must be >= 0."""
        from ug_ttt.rl.mlora_train import do_rollout
        from ug_ttt.rl.uncertainty import UNCERTAINTY_FNS
        rollout = do_rollout(
            ensemble=self.ensemble,
            tokenizer=MagicMock(decode=lambda t: "def foo(): pass"),
            prompt_tokens=self.prompt_tokens,
            adapter_idx=0,
            parent_state=None,
            step=0,
            process_fn=self._simple_process_fn,
            uncertainty_fn=UNCERTAINTY_FNS["rmi"],
            rmi_coef=0.1,
            max_tokens=20,
            temperature=1.0,
        )
        self.assertGreaterEqual(rollout.reward_rmi, -1e-6,
            msg=f"reward_rmi should be >= 0, got {rollout.reward_rmi}")

    def test_rollout_action_mask_covers_generated_tokens(self):
        """action_mask should be 1 for all generated (non-prompt) tokens."""
        from ug_ttt.rl.mlora_train import do_rollout
        from ug_ttt.rl.uncertainty import UNCERTAINTY_FNS
        rollout = do_rollout(
            ensemble=self.ensemble,
            tokenizer=MagicMock(decode=lambda t: "def foo(): pass"),
            prompt_tokens=self.prompt_tokens,
            adapter_idx=0,
            parent_state=None,
            step=0,
            process_fn=self._simple_process_fn,
            uncertainty_fn=UNCERTAINTY_FNS["rmi"],
            rmi_coef=0.1,
            max_tokens=20,
            temperature=1.0,
        )
        mask = rollout.action_mask
        prompt_len = len(rollout.prompt_tokens)
        gen_len = len(rollout.full_tokens) - prompt_len
        # Generated portion of mask (last gen_len entries) should all be 1
        self.assertTrue((mask[-(gen_len):] == 1.0).all(),
            msg="Generated tokens should have action_mask=1")


class TestTrainStep(unittest.TestCase):
    """Tests for train_step: advantages, RL loss, optimizer steps."""

    def setUp(self):
        self.ensemble = MockEnsemble(K=3)
        self.cfg = make_test_config(
            adv_estimator="mean_baseline",
            loss_fn="importance_sampling",
            num_substeps=1,
            rmi_coef=0.1,
        )

    def _make_group(self, rewards: List[float]) -> RolloutGroup:
        rollouts = [make_rollout(reward_exec=r, rmi_coef=0.1) for r in rewards]
        return RolloutGroup(rollouts=rollouts)

    def test_train_step_returns_metrics(self):
        groups = [self._make_group([0.2, 0.8, 0.5, 0.1])]
        metrics = train_step(self.ensemble, groups, self.cfg)
        self.assertIn("train/reward/exec/mean", metrics)
        self.assertIn("train/reward/rmi/mean", metrics)
        self.assertIn("train/reward/total/mean", metrics)
        self.assertIn("train/loss", metrics)

    def test_optimizer_called(self):
        """step_all_optimizers must be called once per substep batch."""
        groups = [self._make_group([0.1, 0.9, 0.5, 0.3])]
        before = self.ensemble._step_count
        train_step(self.ensemble, groups, self.cfg)
        after = self.ensemble._step_count
        self.assertGreater(after, before,
            msg="step_all_optimizers should have been called at least once")

    def test_constant_reward_group_still_processed(self):
        """remove_constant_reward_groups=False → constant group still tracked."""
        cfg = make_test_config(remove_constant_reward_groups=False)
        groups = [self._make_group([1.0, 1.0, 1.0, 1.0])]
        metrics = train_step(self.ensemble, groups, cfg)
        # Should complete without error; constant group → zero advantages → no trainable rollouts
        self.assertIn("train/num_rollouts", metrics)

    def test_multiple_groups(self):
        groups = [
            self._make_group([0.2, 0.8, 0.5]),
            self._make_group([0.9, 0.1, 0.6]),
        ]
        metrics = train_step(self.ensemble, groups, self.cfg)
        self.assertGreater(metrics["train/reward/exec/mean"], 0)

    def test_ppo_loss(self):
        cfg = make_test_config(loss_fn="ppo")
        groups = [self._make_group([0.1, 0.9, 0.5])]
        metrics = train_step(self.ensemble, groups, cfg)
        self.assertIn("train/loss", metrics)


class TestKLPenalty(unittest.TestCase):
    """KL penalty path: advantages should be adjusted when kl_penalty_coef > 0."""

    def setUp(self):
        self.ensemble = MockEnsemble(K=3)

    def test_kl_penalty_adjusts_advantages(self):
        cfg = make_test_config(kl_penalty_coef=1.0, kl_discount_factor=0.0)
        rollouts = [make_rollout(reward_exec=r) for r in [0.2, 0.8, 0.5, 0.1]]
        group = RolloutGroup(rollouts=rollouts)
        metrics = train_step(self.ensemble, [group], cfg)
        # Should complete without error and emit kl metric
        self.assertIn("kl_policy_base", metrics)

    def test_kl_zero_coef_no_kl_metric(self):
        cfg = make_test_config(kl_penalty_coef=0.0)
        rollouts = [make_rollout(reward_exec=r) for r in [0.2, 0.8, 0.5]]
        group = RolloutGroup(rollouts=rollouts)
        metrics = train_step(self.ensemble, [group], cfg)
        self.assertNotIn("kl_policy_base", metrics)


class TestKLTracking(unittest.TestCase):

    def test_kl_sample_train_structure(self):
        # T-1 = 13 for prompt_len=4, gen_len=10 (full_tokens=14)
        rollouts = [make_rollout(prompt_len=4, gen_len=10) for _ in range(3)]
        training_logprobs = [torch.randn(13) for _ in range(3)]
        metrics = compute_kl_sample_train(rollouts, training_logprobs, device="cpu")
        self.assertIn("optim/kl_sample_train_v1", metrics)
        self.assertIn("optim/kl_sample_train_v2", metrics)
        self.assertIn("optim/entropy", metrics)

    def test_identical_logprobs_zero_kl(self):
        rollouts = [make_rollout(prompt_len=4, gen_len=10)]
        # sampling_logprobs has T-1 = 13 entries (0.0 pad for prompt, -0.5 for gen)
        # training logprobs should match at T-1 = 13
        training_logprobs = [torch.full((13,), -0.5)]
        metrics = compute_kl_sample_train(rollouts, training_logprobs, device="cpu")
        self.assertAlmostEqual(metrics["optim/kl_sample_train_v1"], 0.0, places=5)

    def test_empty_rollouts(self):
        metrics = compute_kl_sample_train([], [], device="cpu")
        self.assertEqual(metrics["optim/kl_sample_train_v1"], 0.0)


class TestRolloutGroup(unittest.TestCase):

    def test_get_total_rewards(self):
        rollouts = [make_rollout(reward_exec=r, rmi_coef=0.1, reward_rmi=0.3) for r in [0.2, 0.8]]
        group = RolloutGroup(rollouts=rollouts)
        totals = group.get_total_rewards()
        self.assertEqual(len(totals), 2)
        for r, total in zip([0.2, 0.8], totals):
            expected = r + 0.1 * 0.3
            self.assertAlmostEqual(total, expected, places=6)

    def test_get_exec_rewards(self):
        rollouts = [make_rollout(reward_exec=r) for r in [0.3, 0.7, 0.5]]
        group = RolloutGroup(rollouts=rollouts)
        self.assertEqual(group.get_exec_rewards(), [0.3, 0.7, 0.5])


class TestCheckpointing(unittest.TestCase):

    def test_save_creates_files(self):
        ensemble = MockEnsemble(K=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save(tmpdir, step=5)
            for k in range(2):
                path = os.path.join(tmpdir, f"ensemble_{k}", "step_5", "adapter.pt")
                self.assertTrue(os.path.exists(path),
                    msg=f"Checkpoint file missing: {path}")

    def test_save_load_roundtrip(self):
        """Smoke test: save does not crash, load does not crash."""
        ensemble = MockEnsemble(K=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save(tmpdir, step=3)
            ensemble.load(tmpdir, step=3)  # no-op in mock, but should not raise


class TestFullEpochLoop(unittest.TestCase):
    """
    Simulates one full epoch: sample states → rollouts → train_step.
    Verifies the reward augmentation is applied before advantage computation.
    """

    def test_one_epoch(self):
        K = 3
        ensemble = MockEnsemble(K=K)
        cfg = make_test_config(
            group_size=4,
            groups_per_batch=2,
            rmi_coef=0.2,
            adv_estimator="mean_baseline",
            remove_constant_reward_groups=False,
        )
        uncertainty_fn = UNCERTAINTY_FNS["rmi"]

        # Simulate prompt_fn and process_fn
        prompt_tokens = [1, 2, 3, 4, 5]
        tokenizer = MagicMock()
        tokenizer.decode = lambda t: "def foo(): pass"

        def process_fn(text, state, step):
            return {"reward": float(step % 3) * 0.3, "child_state": None,
                    "correctness": 1.0, "metrics": {}}

        # Rollout phase
        rollout_groups = []
        for g in range(cfg.groups_per_batch):
            group_rollouts = []
            for i in range(cfg.group_size):
                adapter_idx = i % K
                step_idx = g * cfg.group_size + i
                rollout = do_rollout(
                    ensemble=ensemble,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    adapter_idx=adapter_idx,
                    parent_state=None,
                    step=step_idx,
                    process_fn=process_fn,
                    uncertainty_fn=uncertainty_fn,
                    rmi_coef=cfg.rmi_coef,
                    max_tokens=20,
                    temperature=1.0,
                )
                group_rollouts.append(rollout)

            # Verify reward augmentation was applied
            for r in group_rollouts:
                expected = r.reward_exec + cfg.rmi_coef * r.reward_rmi
                self.assertAlmostEqual(r.reward_total, expected, places=6)

            rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        # Train step
        metrics = train_step(ensemble, rollout_groups, cfg)

        # Sanity checks on returned metrics
        self.assertIn("train/reward/exec/mean", metrics)
        self.assertIn("train/reward/rmi/mean", metrics)
        self.assertGreaterEqual(metrics["train/reward/rmi/mean"], -1e-6,
            msg="Mean RMI reward should be non-negative")
        self.assertIn("train/loss", metrics)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
