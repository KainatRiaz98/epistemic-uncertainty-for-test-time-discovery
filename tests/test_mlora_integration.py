"""
mLoRA API integration test — validates that our code calls mLoRA correctly.

Does NOT require a GPU or model weights. Uses the real mLoRA classes
(not stubs) to verify:
  1. LoRAConfig accepts our dict format
  2. MLoRADataConfig / MLoRAData constructors match our calling convention
  3. AdamW optimizer config parses beta1/beta2/weight_decay correctly
  4. model_data() produces a valid ModelData with our custom fields

Run with (must be run separately from test_smoke_cpu.py which uses mLoRA stubs):
    PYTHONPATH=mLoRA:$PYTHONPATH python -m pytest tests/test_mlora_integration.py -v
"""

import os
import sys
import unittest

import torch

# ── Ensure both project root and mLoRA are on the path ──────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLORA_ROOT = os.path.join(PROJECT_ROOT, "mLoRA")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MLORA_ROOT)



class TestLoRAConfigParsing(unittest.TestCase):
    """Verify mLoRA's LoRAConfig accepts the dict we pass from ensemble.py."""

    def test_ensemble_config_dict(self):
        from mlora.config import LoRAConfig

        config_dict = {
            "type": "lora",
            "name": "ensemble_0",
            "path": "./adapters/ensemble_0",
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": {"q_proj": True, "o_proj": True},
            "optimizer": "adamw",
            "lr": 3e-4,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
        }
        cfg = LoRAConfig(config_dict)

        self.assertEqual(cfg.name_, "ensemble_0")
        self.assertEqual(cfg.r_, 32)
        self.assertEqual(cfg.alpha_, 64)
        self.assertAlmostEqual(cfg.dropout_, 0.05)
        self.assertEqual(cfg.target_, {"q_proj": True, "o_proj": True})

    def test_optimizer_config_created(self):
        from mlora.config import LoRAConfig

        config_dict = {
            "type": "lora",
            "name": "test",
            "path": "./test",
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": {"q_proj": True},
            "optimizer": "adamw",
            "lr": 3e-4,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
        }
        cfg = LoRAConfig(config_dict)

        # Verify optimizer config is created and has correct params
        self.assertIsNotNone(cfg.optimizer_config_)
        fn_params = cfg.optimizer_config_.to_fn_parameters()

        self.assertAlmostEqual(fn_params["lr"], 3e-4)
        self.assertEqual(fn_params["betas"], (0.9, 0.95))
        self.assertAlmostEqual(fn_params["weight_decay"], 0.0)

    def test_default_adamw_params_backward_compat(self):
        """Verify that omitting beta1/beta2/weight_decay uses defaults."""
        from mlora.config import LoRAConfig

        config_dict = {
            "type": "lora",
            "name": "test_default",
            "path": "./test",
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": {"q_proj": True},
            "optimizer": "adamw",
            "lr": 1e-4,
        }
        cfg = LoRAConfig(config_dict)
        fn_params = cfg.optimizer_config_.to_fn_parameters()

        # Should use PyTorch defaults
        self.assertEqual(fn_params["betas"], (0.9, 0.999))
        self.assertAlmostEqual(fn_params["weight_decay"], 0.01)


class TestMLoRADataConfig(unittest.TestCase):
    """Verify MLoRADataConfig accepts our calling convention."""

    def test_constructor_kwargs(self):
        from mlora.model.args import MLoRADataConfig

        expand_fn = lambda tokens, align_len=None: (tokens, [[True]])
        loss_fn = lambda i, t, m: None

        cfg = MLoRADataConfig(
            adapter_name="ensemble_0",
            adapter_type="lora",
            start_idx=0,
            end_idx=1,
            expand_fn=expand_fn,
            loss_fn=loss_fn,
            task_name="score_0",
        )

        self.assertEqual(cfg.adapter_name_, "ensemble_0")
        self.assertEqual(cfg.adapter_type_, "lora")
        self.assertEqual(cfg.batch_start_idx_, 0)
        self.assertEqual(cfg.batch_end_idx_, 1)
        self.assertEqual(cfg.task_name_, "score_0")

    def test_model_data_config_conversion(self):
        """MLoRADataConfig.model_data_config() must produce a ModelDataConfig."""
        from mlora.model.args import MLoRADataConfig, ModelDataConfig

        cfg = MLoRADataConfig(
            adapter_name="ensemble_2",
            adapter_type="lora",
            start_idx=2,
            end_idx=3,
            expand_fn=lambda t, a=None: (t, [[True]]),
            loss_fn=lambda i, t, m: None,
            task_name="train_2",
        )

        mdc = cfg.model_data_config()
        self.assertIsInstance(mdc, ModelDataConfig)
        self.assertEqual(mdc.adapter_name_, "ensemble_2")
        self.assertEqual(mdc.batch_start_idx_, 2)


class TestMLoRAData(unittest.TestCase):
    """Verify MLoRAData construction and model_data() output."""

    def _make_data(self, K=3, T=20):
        from mlora.model.args import MLoRAData, MLoRADataConfig

        expand_fn = lambda tokens, align_len=None: (tokens, [[True] * len(tokens[0])])
        loss_fn = lambda i, t, m: None

        configs = [
            MLoRADataConfig(
                adapter_name=f"ensemble_{k}",
                adapter_type="lora",
                start_idx=k,
                end_idx=k + 1,
                expand_fn=expand_fn,
                loss_fn=loss_fn,
                task_name=f"score_{k}",
            )
            for k in range(K)
        ]

        token_ids = list(range(T))
        data = MLoRAData(
            batch_tokens=[token_ids] * K,
            batch_mask=[[True] * T] * K,
            data_config=configs,
        )
        return data

    def test_construction(self):
        data = self._make_data(K=5, T=100)
        self.assertEqual(data.batch_size(), 5)
        self.assertEqual(data.token_len(), 100)

    def test_model_data_fields(self):
        """model_data() must produce ModelData with our custom fields."""
        from mlora.model.args import ModelData

        data = self._make_data(K=3, T=50)

        # Set custom fields that ensemble.py uses
        data.use_flash_causal_ = True
        data.return_hidden_states_ = True

        md = data.model_data()
        self.assertIsInstance(md, ModelData)
        self.assertTrue(md.use_flash_causal_)
        self.assertTrue(md.return_hidden_states_)
        self.assertEqual(len(md.data_config_), 3)
        self.assertEqual(len(md.task_name_), 3)

    def test_kv_cache_field(self):
        """model_data() must pass through kv_cache_ and cache_position_."""
        data = self._make_data(K=1, T=10)
        data.use_flash_causal_ = True
        data.kv_cache_ = [None] * 32  # simulating 32 layers
        data.cache_position_ = 5

        md = data.model_data()
        self.assertEqual(len(md.kv_cache_), 32)
        self.assertEqual(md.cache_position_, 5)


class TestQwen3Compatibility(unittest.TestCase):
    """Verify that Qwen3 is in the supported model types list."""

    def test_qwen3_in_compatible_types(self):
        from mlora.model.llm.model_llama import LlamaCompatibleModelTypes
        self.assertIn("qwen3", LlamaCompatibleModelTypes)

    def test_qwen2_in_compatible_types(self):
        from mlora.model.llm.model_llama import LlamaCompatibleModelTypes
        self.assertIn("qwen2", LlamaCompatibleModelTypes)


class TestEnsembleHelpers(unittest.TestCase):
    """Test ensemble.py helper functions with real mLoRA types."""

    def test_expand_fn_padding(self):
        """LoRAEnsemble._expand_fn must pad tokens and masks correctly."""
        from ug_ttt.rl.ensemble import LoRAEnsemble

        batch = [[1, 2, 3], [1, 2, 3, 4, 5]]
        padded_tokens, padded_masks = LoRAEnsemble._expand_fn(batch)

        self.assertEqual(len(padded_tokens[0]), 5)  # padded to max len
        self.assertEqual(len(padded_tokens[1]), 5)
        self.assertEqual(padded_masks[0], [True, True, True, False, False])
        self.assertEqual(padded_masks[1], [True, True, True, True, True])

    def test_noop_loss_returns_none(self):
        from ug_ttt.rl.ensemble import LoRAEnsemble

        result = LoRAEnsemble._noop_loss(
            torch.zeros(1), torch.zeros(1), torch.zeros(1)
        )
        self.assertIsNone(result)


class TestConfigDataclass(unittest.TestCase):
    """Verify our Config dataclass fields have correct defaults."""

    def test_default_values(self):
        from ug_ttt.rl.mlora_train import Config

        cfg = Config()
        # Core RL
        self.assertEqual(cfg.adv_estimator, "entropic_adaptive_beta")
        self.assertEqual(cfg.adv_estimator_beta, 2.0)
        self.assertEqual(cfg.loss_fn, "importance_sampling")
        # Uncertainty (our additions)
        self.assertIsInstance(cfg.rmi_coef, float)
        self.assertIsInstance(cfg.gamma_max_ratio, float)
        # KL penalty
        self.assertIsInstance(cfg.kl_penalty_coef, float)
        self.assertIsInstance(cfg.kl_discount_factor, float)

    def test_rmi_separation_fields(self):
        """gamma_max_ratio must exist for β-γ coupling."""
        from ug_ttt.rl.mlora_train import Config

        cfg = Config(rmi_coef=0.1, gamma_max_ratio=5.0, adv_estimator_beta=2.0)
        # γ_eff = rmi_coef * min(β / adv_estimator_beta, gamma_max_ratio)
        # At β = adv_estimator_beta: γ_eff = 0.1 * min(1.0, 5.0) = 0.1
        gamma_eff = cfg.rmi_coef * min(1.0 / 1.0, cfg.gamma_max_ratio)
        self.assertAlmostEqual(gamma_eff, 0.1)


class TestRolloutLogprobsPadding(unittest.TestCase):
    """Verify sampling_logprobs alignment matches original train.py."""

    def test_action_mask_and_logprobs_same_length(self):
        from ug_ttt.rl.mlora_train import Rollout

        prompt = [1, 2, 3, 4, 5]
        gen = [100, 101, 102, 103, 104]
        full = prompt + gen
        # Padded logprobs: 0.0 * (prompt_len - 1) + gen_logprobs
        sampling_lp = [0.0] * (len(prompt) - 1) + [-0.5] * len(gen)

        r = Rollout(
            prompt_tokens=prompt,
            full_tokens=full,
            sampling_logprobs=sampling_lp,
            reward_exec=0.5,
            reward_rmi=0.1,
            reward_total=0.6,
        )

        mask = r.action_mask
        self.assertEqual(len(r.sampling_logprobs), len(mask))
        self.assertEqual(len(mask), len(full) - 1)

        # Prompt positions should have 0.0 logprobs AND 0.0 mask
        for i in range(len(prompt) - 1):
            self.assertEqual(r.sampling_logprobs[i], 0.0)
            self.assertEqual(mask[i].item(), 0.0)

        # Gen positions should have real logprobs AND 1.0 mask
        for i in range(len(prompt) - 1, len(mask)):
            self.assertEqual(r.sampling_logprobs[i], -0.5)
            self.assertEqual(mask[i].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
