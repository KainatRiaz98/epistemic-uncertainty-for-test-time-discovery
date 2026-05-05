"""
LoRA Ensemble Manager — wraps mLoRA for multi-adapter lifecycle.

Handles:
    - Initializing K LoRA adapters with different random seeds
    - Batched forward passes for ensemble scoring (single pass, all K adapters)
    - Per-adapter logprob extraction
    - Autoregressive text generation from a single adapter
    - Optimizer stepping for all K adapters
    - Checkpoint save/load

This module is the bridge between mLoRA and the RL training loop.
The training loop (mlora_train.py) should never call mLoRA directly.
"""

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ug_ttt.rl.uncertainty import compute_true_mi
from mlora.model.llm import LLMModel
from mlora.model.args import (
    LinearInfo,
    MLoRAData,
    MLoRADataConfig,
    Tokens,
    Masks,
)
from mlora.config import LoRAConfig
from mlora.executor.context.lora import TrainLoRAContext

logger = logging.getLogger(__name__)


class LoRAEnsemble:
    """
    Manages K LoRA adapters on a shared base model via mLoRA.

    Each adapter is initialized with a different random seed (Kaiming normal),
    creating a "chorus of experts" that represents different hypotheses about
    optimal parameters — the basis for Bayesian epistemic uncertainty estimation.
    """

    def __init__(
        self,
        model: LLMModel,
        num_members: int = 5,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: Optional[Dict[str, bool]] = None,
        learning_rate: float = 4e-5,
        optimizer: str = "adamw",
    ):
        self.model = model
        self.K = num_members
        self.contexts: List[TrainLoRAContext] = []

        if target_modules is None:
            target_modules = {
                "q_proj": True, "k_proj": True,
                "v_proj": True, "o_proj": True,
            }

        self._init_adapters(
            lora_rank, lora_alpha, lora_dropout,
            target_modules, learning_rate, optimizer,
        )

    def _init_adapters(
        self,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: Dict[str, bool],
        lr: float,
        optimizer: str,
    ):
        """Create K LoRA adapters with different random seeds."""
        linears_info = self.model.linears_info()

        for k in range(self.K):
            # Different seed → different Kaiming initialization → different hypothesis
            torch.manual_seed(42 + k * 1000)

            lora_config = LoRAConfig({
                "type": "lora",
                "name": f"ensemble_{k}",
                "path": f"./adapters/ensemble_{k}",
                "r": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": target_modules,
                "optimizer": optimizer,
                "lr": lr,
                # Match original train.py: AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
                "beta1": 0.9,
                "beta2": 0.95,
                "weight_decay": 0.0,
            })

            context = TrainLoRAContext(lora_config, linears_info)
            # Randomize lora_B instead of leaving it at zero.
            # Standard LoRA uses B=0 so all adapters start identical, diverging only
            # via gradient noise. Small random init forces initial diversity so
            # epoch-0 MI is non-zero. std=0.01 is small enough not to destabilise
            # early training (Wang et al. 2023).
            with torch.no_grad():
                for module in context.adapter_model_.values():
                    torch.nn.init.normal_(module.lora_b_, mean=0.0, std=0.01)
            context.switch_device(self.model.device_)
            self.contexts.append(context)
            self.model.load_adapter(context.adapter_model())

        logger.info(f"Initialized {self.K} LoRA ensemble members (rank={rank})")

    # ── LM head access ───────────────────────────────────────────────────

    def _get_lm_head(self) -> torch.nn.Linear:
        """Extract LM head from the model's sequential modules."""
        for module in self.model.sequential():
            if hasattr(module, 'wrapper_module_') and hasattr(module.wrapper_module_, 'lm_head_'):
                return module.wrapper_module_.lm_head_
        raise RuntimeError("Could not find LM head in model")

    # ── Ensemble scoring ──────────────────────────────────────────────────

    def _build_mlora_data(
        self, token_ids: List[int], task_prefix: str
    ) -> MLoRAData:
        """Build MLoRAData for K adapters scoring the same sequence."""
        K = self.K
        T = len(token_ids)
        data_configs = [
            MLoRADataConfig(
                adapter_name=f"ensemble_{k}",
                adapter_type="lora",
                start_idx=k,
                end_idx=k + 1,
                expand_fn=self._expand_fn,
                loss_fn=self._noop_loss,
                task_name=f"{task_prefix}_{k}",
            )
            for k in range(K)
        ]
        mlora_data = MLoRAData(
            batch_tokens=[list(token_ids)] * K,
            batch_mask=[[True] * T] * K,
            data_config=data_configs,
        )
        mlora_data.use_flash_causal_ = True
        return mlora_data

    def _chunked_logprobs(
        self,
        hidden: torch.Tensor,
        token_ids: List[int],
        chunk_size: int = 256,
        compute_mi: bool = False,
    ):
        """
        Extract per-token logprobs from hidden states without materializing
        the full (K, T, V) logits tensor.

        When compute_mi=True, also computes true mutual information from the
        full vocabulary distribution at each position:

            MI(t) = H[p̄(·|y_{<t})] - E_k[H[p_k(·|y_{<t})]]

        The MI computation reuses the log_softmax already computed for logprob
        extraction, adding only entropy sums (<0.03% of LM head matmul cost).

        Applies the LM head in chunks over the sequence dimension.
        Peak logit memory: K × chunk_size × V × 4 bytes.

        Returns:
            If compute_mi=False: (K, T-1) tensor of per-token logprobs.
            If compute_mi=True: tuple of (K, T-1) logprobs and (T-1,) true MI.
        """
        K = hidden.shape[0]
        T = len(token_ids)
        lm_head = self._get_lm_head()
        target = torch.tensor(token_ids[1:], dtype=torch.long, device=hidden.device)

        logprob_chunks = []
        mi_chunks = [] if compute_mi else None

        for start in range(0, T - 1, chunk_size):
            end = min(start + chunk_size, T - 1)
            # hidden[:, start:end] predicts tokens[start+1:end+1]
            chunk_logits = lm_head(hidden[:, start:end, :]).float()  # (K, chunk, V)
            chunk_lp = F.log_softmax(chunk_logits, dim=-1)

            # Gather logprobs at generated tokens
            chunk_target = target[start:end].unsqueeze(0).unsqueeze(-1).expand(K, -1, 1)
            logprob_chunks.append(chunk_lp.gather(2, chunk_target).squeeze(-1))

            if compute_mi:
                # True MI = H[mixture] - E_k[H[member]] per position
                #
                # Per-member entropy: H[p_k] = -Σ_v p_k(v) log p_k(v)
                # chunk_lp is finite (log_softmax is stable), exp() may underflow
                # to 0 but 0 * finite = 0 in IEEE float, so product is safe.
                chunk_probs = chunk_lp.exp()  # (K, chunk, V)
                H_members = -(chunk_probs * chunk_lp).sum(dim=-1)  # (K, chunk)

                # Mixture log-probs: log p̄(v) = logsumexp(log p_k(v)) - log(K)
                log_mix = torch.logsumexp(chunk_lp, dim=0) - math.log(K)  # (chunk, V)
                mix_probs = log_mix.exp()  # (chunk, V)
                # Mixture entropy: H[p̄] = -Σ_v p̄(v) log p̄(v)
                H_mixture = -(mix_probs * log_mix).sum(dim=-1)  # (chunk,)

                mi_chunks.append(H_mixture - H_members.mean(dim=0))  # (chunk,)
                del chunk_probs, H_members, log_mix, mix_probs, H_mixture

            del chunk_logits, chunk_lp

        logprobs = torch.cat(logprob_chunks, dim=1)  # (K, T-1)
        if compute_mi:
            per_token_mi = torch.cat(mi_chunks, dim=0)  # (T-1,)
            return logprobs, per_token_mi
        return logprobs

    def compute_ensemble_logprobs(
        self,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a token sequence through all K adapters in a single forward pass.

        Uses flash causal attention (O(n) memory) and chunked logit extraction
        to avoid materializing the full (K, T, V) logits tensor. Also computes
        true mutual information from the full vocabulary distribution.

        Args:
            token_ids: Complete token sequence (prompt + generated), length T.
            chunk_size: Tokens per chunk for LM head. Lower = less memory.

        Returns:
            Tuple of:
                - (K, T-1) tensor of log p_k(y_t | y_{<t}) for each adapter k
                - (T-1,) tensor of per-position true MI
        """
        mlora_data = self._build_mlora_data(token_ids, "score")
        mlora_data.return_hidden_states_ = True

        self.model.seq_module_.eval()
        try:
            with torch.no_grad():
                hidden = self.model.forward(mlora_data.model_data())  # (K, T, D)
            return self._chunked_logprobs(hidden, token_ids, chunk_size, compute_mi=True)
        finally:
            self.model.seq_module_.train()

    def compute_training_logprobs(
        self,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like compute_ensemble_logprobs but WITH gradient tracking.
        Used during the RL training step.

        Returns:
            hidden: (K, T, dim) hidden states with grad attached
            per_token_logprobs: (K, T-1)
        """
        mlora_data = self._build_mlora_data(token_ids, "train")
        mlora_data.return_hidden_states_ = True

        hidden = self.model.forward(mlora_data.model_data())  # (K, T, D)
        per_token_logprobs = self._chunked_logprobs(hidden, token_ids, chunk_size)

        return hidden, per_token_logprobs

    def compute_training_logprobs_single(
        self,
        k: int,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradients for a SINGLE adapter k.

        Memory cost is 1/K of compute_training_logprobs — only one adapter's
        computation graph is live at a time. Call backward() before the next
        adapter to keep peak memory at O(T × L × D) instead of O(K × T × L × D).

        Returns:
            hidden: (1, T, dim) hidden states with grad attached
            per_token_logprobs: (T-1,) log-probs for adapter k
        """
        T = len(token_ids)
        data_config = MLoRADataConfig(
            adapter_name=f"ensemble_{k}",
            adapter_type="lora",
            start_idx=0,
            end_idx=1,
            expand_fn=self._expand_fn,
            loss_fn=self._noop_loss,
            task_name=f"train_single_{k}",
        )
        mlora_data = MLoRAData(
            batch_tokens=[list(token_ids)],
            batch_mask=[[True] * T],
            data_config=[data_config],
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.return_hidden_states_ = True

        hidden = self.model.forward(mlora_data.model_data())  # (1, T, D)
        per_token_logprobs = self._chunked_logprobs(hidden, token_ids, chunk_size)  # (1, T-1)

        return hidden, per_token_logprobs[0]  # (T-1,)

    # ── Text generation ───────────────────────────────────────────────────

    def _get_n_layers(self) -> int:
        """Count decoder layers in the model."""
        return sum(1 for m in self.model.sequential() if m.name() == "Decoder")

    def _make_gen_data(
        self,
        token_ids: List[int],
        adapter_name: str,
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
        kv_quantize: bool = False,
    ) -> MLoRAData:
        """Build MLoRAData for single-adapter generation."""
        data_config = MLoRADataConfig(
            adapter_name=adapter_name,
            adapter_type="lora",
            start_idx=0,
            end_idx=1,
            expand_fn=self._expand_fn,
            loss_fn=self._noop_loss,
            task_name="generation",
        )
        mlora_data = MLoRAData(
            batch_tokens=[token_ids],
            batch_mask=[[True] * len(token_ids)],
            data_config=[data_config],
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        mlora_data.kv_cache_quantize_ = kv_quantize
        return mlora_data

    def _make_gen_data_batch(
        self,
        token_ids_batch: List[List[int]],
        adapter_name: str,
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
        kv_quantize: bool = False,
    ) -> MLoRAData:
        """Build MLoRAData for batched generation (N sequences, same adapter)."""
        N = len(token_ids_batch)
        data_config = MLoRADataConfig(
            adapter_name=adapter_name,
            adapter_type="lora",
            start_idx=0,
            end_idx=N,
            expand_fn=self._expand_fn,
            loss_fn=self._noop_loss,
            task_name="generation_batch",
        )
        mlora_data = MLoRAData(
            batch_tokens=token_ids_batch,
            batch_mask=[[True] * len(toks) for toks in token_ids_batch],
            data_config=[data_config],
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        mlora_data.kv_cache_quantize_ = kv_quantize
        return mlora_data

    def _make_gen_data_multi_adapter(
        self,
        token_ids_batch: List[List[int]],
        adapter_configs: List[Tuple[str, int, int]],
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
        kv_quantize: bool = False,
        kv_cache_preallocated: bool = False,
    ) -> MLoRAData:
        """Build MLoRAData for multi-adapter generation (N sequences, multiple adapters).

        Each adapter_config is (adapter_name, start_idx, end_idx) specifying which
        contiguous batch range uses which adapter.
        """
        data_configs = [
            MLoRADataConfig(
                adapter_name=name,
                adapter_type="lora",
                start_idx=start,
                end_idx=end,
                expand_fn=self._expand_fn,
                loss_fn=self._noop_loss,
                task_name=f"gen_multi_{name}",
            )
            for name, start, end in adapter_configs
        ]
        mlora_data = MLoRAData(
            batch_tokens=token_ids_batch,
            batch_mask=[[True] * len(toks) for toks in token_ids_batch],
            data_config=data_configs,
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        mlora_data.kv_cache_quantize_ = kv_quantize
        mlora_data.kv_cache_preallocated_ = kv_cache_preallocated
        return mlora_data

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
        kv_quantize: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Autoregressive generation with KV caching.

        Prefills the KV cache with the prompt in one pass, then decodes
        one token at a time reusing cached K/V — O(P + M) instead of O((P+M)²).

        Args:
            prompt_tokens: Tokenized prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            adapter_idx: Which ensemble member to use for generation.
            kv_quantize: If True, store KV cache in int8 (halves memory bandwidth).

        Returns:
            generated_tokens: Full sequence (prompt + generated).
            logprobs: Log-probabilities of each generated token.
        """
        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

        # ── Prefill: process entire prompt, populate KV cache ──
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name,
            kv_cache=kv_cache, cache_position=0, kv_quantize=kv_quantize,
        )
        logits = self.model.forward(mlora_data.model_data())  # (1, P, V)
        next_logits = logits[0, -1, :]  # (V,)

        tokens = list(prompt_tokens)
        gen_logprobs: List[float] = []
        cur_pos = len(prompt_tokens)

        # ── Decode: one token at a time, reusing KV cache ──
        for _ in range(max_tokens):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            log_prob = F.log_softmax(next_logits, dim=-1)[next_token].item()
            gen_logprobs.append(log_prob)
            tokens.append(next_token)

            if next_token == eos_token_id:
                break

            # Forward pass with just the new token + KV cache
            mlora_data = self._make_gen_data(
                [next_token], adapter_name,
                kv_cache=kv_cache, cache_position=cur_pos, kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (1, 1, V)
            next_logits = logits[0, -1, :]
            cur_pos += 1

        return tokens, gen_logprobs

    @torch.no_grad()
    def generate_two_phase(
        self,
        prompt_tokens: List[int],
        phase1_max_tokens: int,
        context_window: int,
        context_buffer: int,
        prefill_tokens: List[int],
        temperature: float = 1.0,
        eos_token_id: int = 2,
        stop_token_ids: Optional[List[int]] = None,
        adapter_idx: int = 0,
        kv_quantize: bool = False,
    ) -> Tuple[List[int], List[float], Optional[List[float]]]:
        """
        Two-phase autoregressive generation matching TwoPhaseTokenCompleter.

        Phase 1 (thinking): decode up to phase1_max_tokens - prompt_len tokens.
        If Phase 1 exhausts budget without stop → inject teacher-forced prefill →
        Phase 2 (code): decode with remaining context budget.

        Args:
            phase1_max_tokens: Total budget for prompt + Phase 1 output (e.g. 26000).
            context_window: Total context window size (e.g. 32768).
            context_buffer: Safety margin tokens (e.g. 50).
            prefill_tokens: Teacher-forced transition tokens (already tokenized).
            stop_token_ids: Additional stop tokens (e.g. [im_end_id]).

        Returns:
            full_tokens: prompt + phase1 + (prefill + phase2 if triggered).
            logprobs: Per-generated-token logprobs (0.0 for prefill tokens).
            custom_mask: (T-1)-length mask or None if single-phase completed.
                0.0 for prompt + prefill positions, 1.0 for trained positions.
        """
        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
        prompt_len = len(prompt_tokens)
        phase1_max_output = phase1_max_tokens - prompt_len
        if phase1_max_output <= 0:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds phase1_max_tokens {phase1_max_tokens}."
            )

        all_stop_ids = set()
        if stop_token_ids:
            all_stop_ids.update(stop_token_ids)

        # ── Prefill prompt into KV cache ──
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name,
            kv_cache=kv_cache, cache_position=0, kv_quantize=kv_quantize,
        )
        logits = self.model.forward(mlora_data.model_data())  # (1, P, V)
        next_logits = logits[0, -1, :]
        del logits

        tokens = list(prompt_tokens)
        gen_logprobs: List[float] = []
        cur_pos = prompt_len
        hit_stop = False

        # ── Phase 1: Thinking tokens ──
        for step in range(phase1_max_output):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            log_prob = F.log_softmax(next_logits, dim=-1)[next_token].item()
            gen_logprobs.append(log_prob)
            tokens.append(next_token)

            if next_token == eos_token_id or next_token in all_stop_ids:
                hit_stop = True
                break

            if step < phase1_max_output - 1:
                mlora_data = self._make_gen_data(
                    [next_token], adapter_name,
                    kv_cache=kv_cache, cache_position=cur_pos, kv_quantize=kv_quantize,
                )
                logits = self.model.forward(mlora_data.model_data())
                next_logits = logits[0, -1, :]
                del logits
                cur_pos += 1

        phase1_gen_count = len(gen_logprobs)

        # If Phase 1 completed naturally (hit stop or didn't exhaust budget),
        # return as single-phase — no prefill needed.
        if hit_stop or phase1_gen_count < phase1_max_output:
            return tokens, gen_logprobs, None

        # ── Phase boundary: inject teacher-forced prefill ──
        # Feed prefill tokens one at a time through KV cache (teacher forcing).
        cur_pos = prompt_len + phase1_gen_count
        # Need a forward pass for the last Phase 1 token to get KV cache updated
        # (the loop above skipped the last forward when step == phase1_max_output - 1)
        last_phase1_token = tokens[-1]
        mlora_data = self._make_gen_data(
            [last_phase1_token], adapter_name,
            kv_cache=kv_cache, cache_position=cur_pos - 1, kv_quantize=kv_quantize,
        )
        logits = self.model.forward(mlora_data.model_data())
        del logits
        # cur_pos is already correct (prompt_len + phase1_gen_count)

        for prefill_tok in prefill_tokens:
            mlora_data = self._make_gen_data(
                [prefill_tok], adapter_name,
                kv_cache=kv_cache, cache_position=cur_pos, kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())
            next_logits = logits[0, -1, :]
            del logits
            tokens.append(prefill_tok)
            gen_logprobs.append(0.0)  # Teacher-forced: no real logprob
            cur_pos += 1

        prefill_len = len(prefill_tokens)

        # ── Phase 2: Code tokens ──
        phase2_budget = (
            context_window - prompt_len - phase1_gen_count - prefill_len - context_buffer
        )
        if phase2_budget <= 0:
            # No room for Phase 2 — build mask and return
            total = len(tokens) - 1
            mask = [0.0] * (prompt_len - 1)  # prompt pad
            mask += [1.0] * phase1_gen_count  # Phase 1 (trained)
            mask += [0.0] * prefill_len       # prefill (not trained)
            return tokens, gen_logprobs, mask

        phase2_gen_count = 0
        for _ in range(phase2_budget):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            log_prob = F.log_softmax(next_logits, dim=-1)[next_token].item()
            gen_logprobs.append(log_prob)
            tokens.append(next_token)
            phase2_gen_count += 1

            if next_token == eos_token_id or next_token in all_stop_ids:
                break

            mlora_data = self._make_gen_data(
                [next_token], adapter_name,
                kv_cache=kv_cache, cache_position=cur_pos, kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())
            next_logits = logits[0, -1, :]
            del logits
            cur_pos += 1

        # ── Build mask (T-1 length) ──
        # prompt_pad (0) | phase1 (1) | prefill (0) | phase2 (1)
        mask = [0.0] * (prompt_len - 1)
        mask += [1.0] * phase1_gen_count
        mask += [0.0] * prefill_len
        mask += [1.0] * phase2_gen_count
        assert len(mask) == len(tokens) - 1, (
            f"Mask length {len(mask)} != tokens-1 {len(tokens)-1}: "
            f"prompt={prompt_len}, p1={phase1_gen_count}, prefill={prefill_len}, p2={phase2_gen_count}"
        )

        return tokens, gen_logprobs, mask

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_tokens: List[int],
        num_sequences: int,
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
        kv_quantize: bool = False,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batched autoregressive generation: N sequences from the same prompt
        in parallel via batched forward passes with shared KV cache prefix.

        ~N× faster than calling generate() N times sequentially because
        the GPU processes all N next-tokens in a single forward pass.

        Args:
            prompt_tokens: Tokenized prompt (shared by all sequences).
            num_sequences: Number of sequences to generate (N).
            max_tokens: Maximum new tokens per sequence.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            adapter_idx: Which ensemble member to use.
            kv_quantize: If True, store KV cache in int8 (halves bandwidth).

        Returns:
            List of (full_tokens, logprobs) tuples, one per sequence.
        """
        if num_sequences == 1:
            result = self.generate(
                prompt_tokens, max_tokens, temperature, eos_token_id, adapter_idx,
                kv_quantize=kv_quantize,
            )
            return [result]

        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        N = num_sequences

        # ── Prefill: process prompt once, then expand KV cache to batch N ──
        # Prefill without quantization for accuracy, quantize after expansion
        kv_cache_single: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name,
            kv_cache=kv_cache_single, cache_position=0,
        )
        logits = self.model.forward(mlora_data.model_data())  # (1, P, V)
        next_logits = logits[0, -1, :]  # (V,)

        # Expand KV cache from batch=1 to batch=N by repeating
        kv_cache_batch: List = []
        for layer_cache in kv_cache_single:
            if layer_cache is not None:
                k, v = layer_cache  # (1, n_kv_head, seq_len, head_dim)
                k_expanded = k.expand(N, -1, -1, -1).contiguous()
                v_expanded = v.expand(N, -1, -1, -1).contiguous()
                if kv_quantize:
                    # Quantize to int8 with per-channel scales
                    k_absmax = k_expanded.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    v_absmax = v_expanded.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    k_scale = k_absmax / 127.0
                    v_scale = v_absmax / 127.0
                    kv_cache_batch.append((
                        (k_expanded / k_scale).round().to(torch.int8),
                        (v_expanded / v_scale).round().to(torch.int8),
                        k_scale, v_scale,
                    ))
                else:
                    kv_cache_batch.append((k_expanded, v_expanded))
            else:
                kv_cache_batch.append(None)

        # Initialize per-sequence state
        all_tokens: List[List[int]] = [list(prompt_tokens) for _ in range(N)]
        all_logprobs: List[List[float]] = [[] for _ in range(N)]
        finished = [False] * N
        cur_pos = len(prompt_tokens)

        # Sample first token for all N sequences from shared prefill logits
        log_probs_all = F.log_softmax(next_logits, dim=-1)  # (V,)
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            first_tokens = torch.multinomial(probs.unsqueeze(0).expand(N, -1), num_samples=1).squeeze(-1)  # (N,)
        else:
            first_tokens = next_logits.argmax().unsqueeze(0).expand(N)

        next_token_ids: List[List[int]] = []
        for i in range(N):
            tok = first_tokens[i].item()
            lp = log_probs_all[tok].item()
            all_tokens[i].append(tok)
            all_logprobs[i].append(lp)
            if tok == eos_token_id:
                finished[i] = True
            next_token_ids.append([tok])

        # ── Batched decode loop ──
        for step in range(1, max_tokens):
            if all(finished):
                break

            # Build batch of next tokens (finished sequences get a dummy token)
            batch_tokens = []
            for i in range(N):
                if finished[i]:
                    batch_tokens.append([eos_token_id])  # dummy, output ignored
                else:
                    batch_tokens.append([all_tokens[i][-1]])

            mlora_data = self._make_gen_data_batch(
                batch_tokens, adapter_name,
                kv_cache=kv_cache_batch, cache_position=cur_pos,
                kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (N, 1, V)
            cur_pos += 1

            # Sample next token for each active sequence
            for i in range(N):
                if finished[i]:
                    continue

                seq_logits = logits[i, -1, :]  # (V,)
                log_probs_i = F.log_softmax(seq_logits, dim=-1)

                if temperature > 0:
                    probs_i = F.softmax(seq_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs_i, num_samples=1).item()
                else:
                    next_tok = seq_logits.argmax().item()

                all_tokens[i].append(next_tok)
                all_logprobs[i].append(log_probs_i[next_tok].item())

                if next_tok == eos_token_id:
                    finished[i] = True

        return list(zip(all_tokens, all_logprobs))

    # ── Multi-adapter batched generation ─────────────────────────────────

    @torch.no_grad()
    def generate_batch_multi_adapter(
        self,
        prompt_tokens: List[int],
        adapter_assignments: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        kv_quantize: bool = False,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batched autoregressive generation where different sequences use
        different adapters, all in a single forward pass per decode step.

        Leverages mLoRA's native multi-adapter routing: each batch element
        is assigned to an adapter via MLoRADataConfig, and the LoRA forward
        applies the correct weights per batch slice.

        ~K× faster than calling generate_batch() once per adapter because:
          - One prefill pass (K adapters batched) instead of K separate prefills
          - One decode loop with N elements instead of K loops with N/K elements

        Args:
            prompt_tokens: Tokenized prompt (shared by all sequences).
            adapter_assignments: adapter_idx for each of N sequences.
                E.g. [0, 1, 2, 3, 4, 0, 1, 2] for group_size=8, K=5.
            max_tokens: Maximum new tokens per sequence.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            kv_quantize: If True, store KV cache in int8.

        Returns:
            List of (full_tokens, logprobs) tuples in the ORIGINAL order
            of adapter_assignments (not grouped by adapter).
        """
        N = len(adapter_assignments)
        logger.info(f"  generate_batch_multi_adapter: N={N} max_tokens={max_tokens} kv_q={kv_quantize}")

        # ── Fast path: single adapter → delegate to existing method ──
        unique_adapters = set(adapter_assignments)
        if len(unique_adapters) == 1:
            return self.generate_batch(
                prompt_tokens, N, max_tokens, temperature,
                eos_token_id, adapter_idx=adapter_assignments[0],
                kv_quantize=kv_quantize,
            )
        if N == 1:
            result = self.generate(
                prompt_tokens, max_tokens, temperature,
                eos_token_id, adapter_idx=adapter_assignments[0],
                kv_quantize=kv_quantize,
            )
            return [result]

        n_layers = self._get_n_layers()

        # ── Phase 1: Group sequences by adapter for contiguous batch ranges ──
        # mLoRA requires data[start_idx:end_idx] to be contiguous per adapter.
        adapter_groups: Dict[int, List[int]] = {}
        for i, a in enumerate(adapter_assignments):
            adapter_groups.setdefault(a, []).append(i)
        sorted_adapters = sorted(adapter_groups.keys())

        batch_to_orig: List[int] = []     # batch_pos → original_pos
        adapter_configs: List[Tuple[str, int, int]] = []  # (name, start, end)
        batch_pos = 0
        for k in sorted_adapters:
            positions = adapter_groups[k]
            n_seqs = len(positions)
            batch_to_orig.extend(positions)
            adapter_configs.append((f"ensemble_{k}", batch_pos, batch_pos + n_seqs))
            batch_pos += n_seqs

        # ── Phase 2: Batched multi-adapter prefill ──
        # One forward pass with num_unique batch elements, each routing to its adapter.
        num_unique = len(sorted_adapters)
        prefill_kv_cache: List = [None] * n_layers
        prefill_configs = [
            (f"ensemble_{k}", idx, idx + 1)
            for idx, k in enumerate(sorted_adapters)
        ]
        prefill_data = self._make_gen_data_multi_adapter(
            token_ids_batch=[list(prompt_tokens)] * num_unique,
            adapter_configs=prefill_configs,
            kv_cache=prefill_kv_cache,
            cache_position=0,
            kv_quantize=False,  # don't quantize yet; quantize after expansion
        )
        logits = self.model.forward(prefill_data.model_data())  # (num_unique, P, V)

        # Extract per-adapter last-token logits
        adapter_prefill_logits: Dict[int, torch.Tensor] = {}
        for idx, k in enumerate(sorted_adapters):
            adapter_prefill_logits[k] = logits[idx, -1, :]  # (V,)
        del logits

        # ── Phase 3: Expand KV cache from num_unique → N sequences ──
        # When quantizing, quantize the K unique caches FIRST (K × fp16 → K × int8),
        # then expand the int8+scale tensors to N. Peak memory: K × fp16 + N × int8
        # instead of N × fp16 + N × int8.
        kv_cache_batch: List = []
        for layer_cache in prefill_kv_cache:
            if layer_cache is None:
                kv_cache_batch.append(None)
                continue
            cached_k, cached_v = layer_cache  # (num_unique, n_kv_heads, P, head_dim)

            if kv_quantize:
                # Quantize K unique rows in fp16 first
                k_absmax = cached_k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                v_absmax = cached_v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                k_scale = k_absmax / 127.0
                v_scale = v_absmax / 127.0
                k_q = (cached_k / k_scale).round().to(torch.int8)  # (num_unique, ...)
                v_q = (cached_v / v_scale).round().to(torch.int8)
                del cached_k, cached_v  # free fp16 immediately

                # Expand int8 + scales from num_unique → N
                expanded_kq, expanded_vq = [], []
                expanded_ks, expanded_vs = [], []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_kq.append(k_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vq.append(v_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_ks.append(k_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vs.append(v_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())

                kv_cache_batch.append((
                    torch.cat(expanded_kq, dim=0),
                    torch.cat(expanded_vq, dim=0),
                    torch.cat(expanded_ks, dim=0),
                    torch.cat(expanded_vs, dim=0),
                ))
            else:
                expanded_k_parts = []
                expanded_v_parts = []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_k_parts.append(
                        cached_k[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                    expanded_v_parts.append(
                        cached_v[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                kv_cache_batch.append((
                    torch.cat(expanded_k_parts, dim=0),
                    torch.cat(expanded_v_parts, dim=0),
                ))
        del prefill_kv_cache

        # ── Phase 4: Sample first token for all N sequences ──
        all_tokens: List[List[int]] = [list(prompt_tokens) for _ in range(N)]
        all_logprobs: List[List[float]] = [[] for _ in range(N)]
        finished = [False] * N
        cur_pos = len(prompt_tokens)

        for name, start, end in adapter_configs:
            k = int(name.split("_")[1])
            next_logits_k = adapter_prefill_logits[k]
            log_probs_k = F.log_softmax(next_logits_k, dim=-1)
            n_seqs = end - start

            if temperature > 0:
                probs_k = F.softmax(next_logits_k / temperature, dim=-1)
                first_tokens_k = torch.multinomial(
                    probs_k.unsqueeze(0).expand(n_seqs, -1), num_samples=1
                ).squeeze(-1)
            else:
                first_tokens_k = next_logits_k.argmax().unsqueeze(0).expand(n_seqs)

            for j in range(n_seqs):
                batch_pos = start + j
                tok = first_tokens_k[j].item()
                lp = log_probs_k[tok].item()
                all_tokens[batch_pos].append(tok)
                all_logprobs[batch_pos].append(lp)
                if tok == eos_token_id:
                    finished[batch_pos] = True
        del adapter_prefill_logits

        # ── Phase 5: Decode loop (multi-adapter forward per step) ──
        for step in range(1, max_tokens):
            if all(finished):
                break

            if step % 2048 == 0:
                n_active = sum(1 for f in finished if not f)
                logger.info(f"    [gen] step={step}/{max_tokens} active={n_active}/{N}")

            batch_tokens = []
            for i in range(N):
                if finished[i]:
                    batch_tokens.append([eos_token_id])
                else:
                    batch_tokens.append([all_tokens[i][-1]])

            mlora_data = self._make_gen_data_multi_adapter(
                token_ids_batch=batch_tokens,
                adapter_configs=adapter_configs,
                kv_cache=kv_cache_batch,
                cache_position=cur_pos,
                kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (N, 1, V)
            cur_pos += 1

            for i in range(N):
                if finished[i]:
                    continue
                seq_logits = logits[i, -1, :]
                log_probs_i = F.log_softmax(seq_logits, dim=-1)
                if temperature > 0:
                    probs_i = F.softmax(seq_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs_i, num_samples=1).item()
                else:
                    next_tok = seq_logits.argmax().item()
                all_tokens[i].append(next_tok)
                all_logprobs[i].append(log_probs_i[next_tok].item())
                if next_tok == eos_token_id:
                    finished[i] = True

        # ── Phase 6: Reorder results from batch order → original order ──
        results_batch_order = list(zip(all_tokens, all_logprobs))
        results: List[Optional[Tuple[List[int], List[float]]]] = [None] * N
        for batch_pos, orig_pos in enumerate(batch_to_orig):
            results[orig_pos] = results_batch_order[batch_pos]
        return results

    @torch.no_grad()
    def generate_batch_multi_adapter_two_phase(
        self,
        prompt_tokens: List[int],
        adapter_assignments: List[int],
        phase1_max_tokens: int,
        context_window: int,
        context_buffer: int,
        prefill_tokens: List[int],
        temperature: float = 1.0,
        eos_token_id: int = 2,
        stop_token_ids: Optional[List[int]] = None,
        kv_quantize: bool = False,
        # ── Streaming MI early stop parameters ──
        streaming_mi_enabled: bool = False,
        mi_window_size: int = 4096,
        mi_check_interval: int = 2048,
        mi_min_gen: int = 4096,
        mi_threshold: float = 0.0,
        mi_wrap_budget: int = 6000,
    ) -> List[Tuple[List[int], List[float], Optional[List[float]], Optional[Dict[str, Any]]]]:
        """
        Batched multi-adapter two-phase generation with optional streaming MI early stop.

        Like generate_batch_multi_adapter but with Phase 1/Phase 2 boundary:
        sequences that exhaust Phase 1 budget get teacher-forced prefill tokens
        injected before continuing to Phase 2.

        When streaming_mi_enabled=True, periodically checks ensemble MI on a
        window of recent tokens. Low-MI sequences get early-stopped with
        teacher-forced prefill + wrap budget, saving tokens for high-MI rollouts.

        Returns:
            List of (full_tokens, logprobs, custom_mask_or_None, mi_meta) per
            sequence, in the ORIGINAL order of adapter_assignments.
            mi_meta is a dict with keys: mi_stopped, mi_stop_step, mi_values,
            mi_check_steps — or None if streaming MI was disabled.
        """
        N = len(adapter_assignments)
        prompt_len = len(prompt_tokens)
        logger.info(
            f"  generate_batch_multi_adapter_two_phase: N={N} "
            f"phase1_max={phase1_max_tokens} streaming_mi={streaming_mi_enabled} kv_q={kv_quantize}"
        )
        phase1_max_output = phase1_max_tokens - prompt_len
        if phase1_max_output <= 0:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds phase1_max_tokens {phase1_max_tokens}."
            )

        all_stop_ids = set()
        if stop_token_ids:
            all_stop_ids.update(stop_token_ids)

        # ── Fast path: single sequence → delegate (only when no streaming MI) ──
        if N == 1 and not streaming_mi_enabled:
            tokens, lps, mask = self.generate_two_phase(
                prompt_tokens, phase1_max_tokens, context_window, context_buffer,
                prefill_tokens, temperature, eos_token_id, stop_token_ids,
                adapter_idx=adapter_assignments[0], kv_quantize=kv_quantize,
            )
            return [(tokens, lps, mask, None)]

        n_layers = self._get_n_layers()

        # ── Phase 1: Group sequences by adapter for contiguous batch ranges ──
        adapter_groups: Dict[int, List[int]] = {}
        for i, a in enumerate(adapter_assignments):
            adapter_groups.setdefault(a, []).append(i)
        sorted_adapters = sorted(adapter_groups.keys())

        batch_to_orig: List[int] = []
        adapter_configs: List[Tuple[str, int, int]] = []
        batch_pos = 0
        for k in sorted_adapters:
            positions = adapter_groups[k]
            n_seqs = len(positions)
            batch_to_orig.extend(positions)
            adapter_configs.append((f"ensemble_{k}", batch_pos, batch_pos + n_seqs))
            batch_pos += n_seqs

        # ── Phase 2: Batched multi-adapter prefill ──
        num_unique = len(sorted_adapters)
        prefill_kv_cache: List = [None] * n_layers
        prefill_configs = [
            (f"ensemble_{k}", idx, idx + 1)
            for idx, k in enumerate(sorted_adapters)
        ]
        prefill_data = self._make_gen_data_multi_adapter(
            token_ids_batch=[list(prompt_tokens)] * num_unique,
            adapter_configs=prefill_configs,
            kv_cache=prefill_kv_cache,
            cache_position=0,
            kv_quantize=False,
        )
        logits = self.model.forward(prefill_data.model_data())

        adapter_prefill_logits: Dict[int, torch.Tensor] = {}
        for idx, k in enumerate(sorted_adapters):
            adapter_prefill_logits[k] = logits[idx, -1, :]
        del logits

        # ── Phase 3: Expand KV cache from num_unique → N sequences ──
        kv_cache_batch: List = []
        for layer_cache in prefill_kv_cache:
            if layer_cache is None:
                kv_cache_batch.append(None)
                continue
            cached_k, cached_v = layer_cache

            if kv_quantize:
                k_absmax = cached_k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                v_absmax = cached_v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                k_scale = k_absmax / 127.0
                v_scale = v_absmax / 127.0
                k_q = (cached_k / k_scale).round().to(torch.int8)
                v_q = (cached_v / v_scale).round().to(torch.int8)
                del cached_k, cached_v

                expanded_kq, expanded_vq = [], []
                expanded_ks, expanded_vs = [], []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_kq.append(k_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vq.append(v_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_ks.append(k_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vs.append(v_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                kv_cache_batch.append((
                    torch.cat(expanded_kq, dim=0),
                    torch.cat(expanded_vq, dim=0),
                    torch.cat(expanded_ks, dim=0),
                    torch.cat(expanded_vs, dim=0),
                ))
            else:
                expanded_k_parts, expanded_v_parts = [], []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_k_parts.append(
                        cached_k[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                    expanded_v_parts.append(
                        cached_v[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                kv_cache_batch.append((
                    torch.cat(expanded_k_parts, dim=0),
                    torch.cat(expanded_v_parts, dim=0),
                ))
        del prefill_kv_cache

        # ── Phase 3b: Convert to pre-allocated KV cache for O(1) decode steps ──
        # Instead of torch.cat each step (O(n) copy → O(n²) total), pre-allocate
        # the full buffer and use index-copy (O(1) per step → O(n) total).
        max_cache_len = context_window
        for layer_idx in range(len(kv_cache_batch)):
            entry = kv_cache_batch[layer_idx]
            if entry is None:
                continue
            if kv_quantize:
                k_q, v_q, k_s, v_s = entry
                dev = k_q.device
                _, n_h, _, h_d = k_q.shape
                k_buf = torch.zeros(N, n_h, max_cache_len, h_d, dtype=torch.int8, device=dev)
                v_buf = torch.zeros(N, n_h, max_cache_len, h_d, dtype=torch.int8, device=dev)
                ks_buf = torch.zeros(N, n_h, max_cache_len, 1, dtype=k_s.dtype, device=dev)
                vs_buf = torch.zeros(N, n_h, max_cache_len, 1, dtype=v_s.dtype, device=dev)
                k_buf[:, :, :prompt_len, :] = k_q
                v_buf[:, :, :prompt_len, :] = v_q
                ks_buf[:, :, :prompt_len, :] = k_s
                vs_buf[:, :, :prompt_len, :] = v_s
                kv_cache_batch[layer_idx] = (k_buf, v_buf, ks_buf, vs_buf)
                del k_q, v_q, k_s, v_s
            else:
                k, v = entry
                dev = k.device
                _, n_h, _, h_d = k.shape
                k_buf = torch.zeros(N, n_h, max_cache_len, h_d, dtype=k.dtype, device=dev)
                v_buf = torch.zeros(N, n_h, max_cache_len, h_d, dtype=v.dtype, device=dev)
                k_buf[:, :, :prompt_len, :] = k
                v_buf[:, :, :prompt_len, :] = v
                kv_cache_batch[layer_idx] = (k_buf, v_buf)
                del k, v

        # ── Phase 4: Sample first token for all N sequences ──
        all_tokens: List[List[int]] = [list(prompt_tokens) for _ in range(N)]
        all_logprobs: List[List[float]] = [[] for _ in range(N)]
        finished = [False] * N
        cur_pos = prompt_len

        # ── Streaming MI early stop state ──
        mi_stopped = [False] * N
        mi_stop_step = [-1] * N
        mi_values_all: List[List[float]] = [[] for _ in range(N)]
        mi_check_steps_all: List[List[int]] = [[] for _ in range(N)]
        mi_prefill_cursor = [0] * N   # position in prefill_tokens for MI-stopped seqs
        mi_wrap_remaining = [0] * N   # remaining wrap budget after prefill

        for name, start, end in adapter_configs:
            k = int(name.split("_")[1])
            next_logits_k = adapter_prefill_logits[k]
            log_probs_k = F.log_softmax(next_logits_k, dim=-1)
            n_seqs = end - start

            if temperature > 0:
                probs_k = F.softmax(next_logits_k / temperature, dim=-1)
                first_tokens_k = torch.multinomial(
                    probs_k.unsqueeze(0).expand(n_seqs, -1), num_samples=1
                ).squeeze(-1)
            else:
                first_tokens_k = next_logits_k.argmax().unsqueeze(0).expand(n_seqs)

            for j in range(n_seqs):
                bp = start + j
                tok = first_tokens_k[j].item()
                lp = log_probs_k[tok].item()
                all_tokens[bp].append(tok)
                all_logprobs[bp].append(lp)
                if tok == eos_token_id or tok in all_stop_ids:
                    finished[bp] = True
        del adapter_prefill_logits

        # ── Phase 5a: Phase 1 decode loop (thinking tokens) ──
        for step in range(1, phase1_max_output):
            if all(finished):
                break

            if step % 2048 == 0:
                n_active = sum(1 for f in finished if not f)
                n_mi_stopped = sum(mi_stopped)
                logger.info(
                    f"    [gen/p1] step={step}/{phase1_max_output} "
                    f"active={n_active}/{N} mi_stopped={n_mi_stopped}"
                )

            # ── Streaming MI checkpoint ──
            if (streaming_mi_enabled
                    and step >= mi_min_gen
                    and (step - mi_min_gen) % mi_check_interval == 0):
                check_indices = [
                    i for i in range(N)
                    if not finished[i] and not mi_stopped[i]
                ]
                if check_indices:
                    windows = []
                    for i in check_indices:
                        w_start = max(0, len(all_tokens[i]) - mi_window_size)
                        windows.append(all_tokens[i][w_start:])
                    window_mis = self.compute_window_mi_batch(windows)
                    for j, i in enumerate(check_indices):
                        mi_values_all[i].append(window_mis[j])
                        mi_check_steps_all[i].append(step)
                        if window_mis[j] < mi_threshold:
                            mi_stopped[i] = True
                            mi_stop_step[i] = step
                            mi_prefill_cursor[i] = 0
                            mi_wrap_remaining[i] = mi_wrap_budget
                            logger.debug(
                                f"  MI early stop: seq {i} at step {step}, "
                                f"MI={window_mis[j]:.6f} < threshold={mi_threshold:.6f}"
                            )

            batch_tokens = []
            for i in range(N):
                if finished[i]:
                    batch_tokens.append([eos_token_id])
                else:
                    batch_tokens.append([all_tokens[i][-1]])

            mlora_data = self._make_gen_data_multi_adapter(
                token_ids_batch=batch_tokens,
                adapter_configs=adapter_configs,
                kv_cache=kv_cache_batch,
                cache_position=cur_pos,
                kv_quantize=kv_quantize,
                kv_cache_preallocated=True,
            )
            logits = self.model.forward(mlora_data.model_data())
            cur_pos += 1

            for i in range(N):
                if finished[i]:
                    continue

                if mi_stopped[i]:
                    # ── MI-stopped: inject prefill tokens, then wrap decode ──
                    if mi_prefill_cursor[i] < len(prefill_tokens):
                        # Teacher-forced prefill token
                        all_tokens[i].append(prefill_tokens[mi_prefill_cursor[i]])
                        all_logprobs[i].append(0.0)
                        mi_prefill_cursor[i] += 1
                    elif mi_wrap_remaining[i] > 0:
                        # Free decode for wrap-up code
                        seq_logits = logits[i, -1, :]
                        log_probs_i = F.log_softmax(seq_logits, dim=-1)
                        if temperature > 0:
                            probs_i = F.softmax(seq_logits / temperature, dim=-1)
                            next_tok = torch.multinomial(probs_i, num_samples=1).item()
                        else:
                            next_tok = seq_logits.argmax().item()
                        all_tokens[i].append(next_tok)
                        all_logprobs[i].append(log_probs_i[next_tok].item())
                        mi_wrap_remaining[i] -= 1
                        if (next_tok == eos_token_id or next_tok in all_stop_ids
                                or mi_wrap_remaining[i] <= 0):
                            finished[i] = True
                    else:
                        finished[i] = True
                else:
                    # ── Normal Phase 1 decode ──
                    seq_logits = logits[i, -1, :]
                    log_probs_i = F.log_softmax(seq_logits, dim=-1)
                    if temperature > 0:
                        probs_i = F.softmax(seq_logits / temperature, dim=-1)
                        next_tok = torch.multinomial(probs_i, num_samples=1).item()
                    else:
                        next_tok = seq_logits.argmax().item()
                    all_tokens[i].append(next_tok)
                    all_logprobs[i].append(log_probs_i[next_tok].item())
                    if next_tok == eos_token_id or next_tok in all_stop_ids:
                        finished[i] = True

        # Track which sequences completed in Phase 1 vs need Phase 2
        # MI-stopped sequences already handled prefill+wrap inside Phase 5a
        phase1_finished = list(finished)  # snapshot
        phase1_gen_counts = [len(all_logprobs[i]) for i in range(N)]
        needs_phase2 = [
            not phase1_finished[i] and not mi_stopped[i]
            for i in range(N)
        ]

        if any(needs_phase2):
            # ── Phase 5b: Teacher-forced prefill injection ──
            # Feed prefill tokens one at a time through the batch.
            # Finished/phase1-completed sequences get dummy tokens.
            for prefill_tok in prefill_tokens:
                batch_tokens = []
                for i in range(N):
                    if finished[i] or not needs_phase2[i]:
                        batch_tokens.append([eos_token_id])
                    else:
                        batch_tokens.append([all_tokens[i][-1]] if len(all_tokens[i]) > prompt_len else [eos_token_id])

                mlora_data = self._make_gen_data_multi_adapter(
                    token_ids_batch=batch_tokens,
                    adapter_configs=adapter_configs,
                    kv_cache=kv_cache_batch,
                    cache_position=cur_pos,
                    kv_quantize=kv_quantize,
                    kv_cache_preallocated=True,
                )
                logits = self.model.forward(mlora_data.model_data())
                cur_pos += 1

                for i in range(N):
                    if needs_phase2[i] and not finished[i]:
                        all_tokens[i].append(prefill_tok)
                        all_logprobs[i].append(0.0)  # teacher-forced
            del logits

            # ── Phase 5c: Phase 2 decode loop (code tokens) ──
            # Budget: remaining context after prompt + phase1 + prefill
            phase2_budget = (
                context_window - prompt_len
                - max(phase1_gen_counts)  # conservative: use max phase1 count
                - len(prefill_tokens)
                - context_buffer
            )

            for step in range(max(0, phase2_budget)):
                if all(finished):
                    break

                batch_tokens = []
                for i in range(N):
                    if finished[i]:
                        batch_tokens.append([eos_token_id])
                    else:
                        batch_tokens.append([all_tokens[i][-1]])

                mlora_data = self._make_gen_data_multi_adapter(
                    token_ids_batch=batch_tokens,
                    adapter_configs=adapter_configs,
                    kv_cache=kv_cache_batch,
                    cache_position=cur_pos,
                    kv_quantize=kv_quantize,
                    kv_cache_preallocated=True,
                )
                logits = self.model.forward(mlora_data.model_data())
                cur_pos += 1

                for i in range(N):
                    if finished[i]:
                        continue
                    seq_logits = logits[i, -1, :]
                    log_probs_i = F.log_softmax(seq_logits, dim=-1)
                    if temperature > 0:
                        probs_i = F.softmax(seq_logits / temperature, dim=-1)
                        next_tok = torch.multinomial(probs_i, num_samples=1).item()
                    else:
                        next_tok = seq_logits.argmax().item()
                    all_tokens[i].append(next_tok)
                    all_logprobs[i].append(log_probs_i[next_tok].item())
                    if next_tok == eos_token_id or next_tok in all_stop_ids:
                        finished[i] = True

        # ── Phase 6: Build masks, MI metadata, and reorder results ──
        prefill_len = len(prefill_tokens)
        results_batch_order: List[Tuple[List[int], List[float], Optional[List[float]], Optional[Dict[str, Any]]]] = []
        for i in range(N):
            # Build MI metadata for all sequences (even if MI was disabled)
            mi_meta: Optional[Dict[str, Any]] = None
            if streaming_mi_enabled:
                mi_meta = {
                    "mi_stopped": mi_stopped[i],
                    "mi_stop_step": mi_stop_step[i],
                    "mi_values": mi_values_all[i],
                    "mi_check_steps": mi_check_steps_all[i],
                }

            if mi_stopped[i]:
                # MI early stopped: prompt(0) | thinking(1) | prefill(0) | wrap(1)
                p1_count = mi_stop_step[i]  # thinking tokens before MI stop
                p_len = mi_prefill_cursor[i]  # should == len(prefill_tokens)
                wrap_count = len(all_logprobs[i]) - p1_count - p_len
                mask = [0.0] * (prompt_len - 1)
                mask += [1.0] * p1_count
                mask += [0.0] * p_len
                mask += [1.0] * max(0, wrap_count)
                expected = len(all_tokens[i]) - 1
                if len(mask) < expected:
                    mask += [1.0] * (expected - len(mask))
                elif len(mask) > expected:
                    mask = mask[:expected]
                results_batch_order.append((all_tokens[i], all_logprobs[i], mask, mi_meta))
            elif needs_phase2[i]:
                # Budget-stopped: prompt(0) | phase1(1) | prefill(0) | phase2(1)
                p1_count = phase1_gen_counts[i]
                p2_count = len(all_logprobs[i]) - p1_count - prefill_len
                mask = [0.0] * (prompt_len - 1)
                mask += [1.0] * p1_count
                mask += [0.0] * prefill_len
                mask += [1.0] * max(0, p2_count)
                expected = len(all_tokens[i]) - 1
                if len(mask) < expected:
                    mask += [1.0] * (expected - len(mask))
                elif len(mask) > expected:
                    mask = mask[:expected]
                results_batch_order.append((all_tokens[i], all_logprobs[i], mask, mi_meta))
            else:
                # Single-phase completed (hit stop/EOS): no custom mask
                results_batch_order.append((all_tokens[i], all_logprobs[i], None, mi_meta))

        results: List[Optional[Tuple[List[int], List[float], Optional[List[float]], Optional[Dict[str, Any]]]]] = [None] * N
        for batch_pos_idx, orig_pos in enumerate(batch_to_orig):
            results[orig_pos] = results_batch_order[batch_pos_idx]
        return results

    # ── Batched ensemble scoring ─────────────────────────────────────────

    def compute_ensemble_logprobs_batch(
        self,
        token_ids_list: List[List[int]],
        chunk_size: int = 256,
        max_score_batch: int = 2,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Score multiple sequences through all K adapters, sub-batching to
        avoid OOM on long sequences.

        Args:
            token_ids_list: List of token sequences to score.
            chunk_size: Tokens per chunk for LM head.
            max_score_batch: Max sequences per forward pass (each becomes K
                batch elements). Prevents OOM for long sequences.

        Returns:
            List of (logprobs, per_token_mi) tuples, one per input sequence.
            logprobs: (K, T_i-1), per_token_mi: (T_i-1,).
        """
        if len(token_ids_list) == 1:
            return [self.compute_ensemble_logprobs(token_ids_list[0], chunk_size)]

        # Group by length for efficient batching (avoids excessive padding)
        length_groups: Dict[int, List[int]] = {}
        for idx, tids in enumerate(token_ids_list):
            length_groups.setdefault(len(tids), []).append(idx)

        results: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(token_ids_list)

        self.model.seq_module_.eval()
        try:
            for length, indices in length_groups.items():
                # Sub-batch to avoid OOM: process max_score_batch sequences per forward pass
                for sub_start in range(0, len(indices), max_score_batch):
                    sub_indices = indices[sub_start:sub_start + max_score_batch]
                    n_sub = len(sub_indices)

                    batch_tokens = []
                    batch_masks = []
                    data_configs = []

                    for seq_pos, orig_idx in enumerate(sub_indices):
                        tids = token_ids_list[orig_idx]
                        for k in range(self.K):
                            batch_idx = seq_pos * self.K + k
                            batch_tokens.append(list(tids))
                            batch_masks.append([True] * length)
                            data_configs.append(MLoRADataConfig(
                                adapter_name=f"ensemble_{k}",
                                adapter_type="lora",
                                start_idx=batch_idx,
                                end_idx=batch_idx + 1,
                                expand_fn=self._expand_fn,
                                loss_fn=self._noop_loss,
                                task_name=f"batch_score_{orig_idx}_{k}",
                            ))

                    mlora_data = MLoRAData(
                        batch_tokens=batch_tokens,
                        batch_mask=batch_masks,
                        data_config=data_configs,
                    )
                    mlora_data.use_flash_causal_ = True
                    mlora_data.return_hidden_states_ = True

                    with torch.no_grad():
                        hidden = self.model.forward(mlora_data.model_data())
                        # hidden: (n_sub*K, T, D)

                    # Extract per-sequence, per-adapter logprobs + true MI
                    lm_head = self._get_lm_head()
                    for seq_pos, orig_idx in enumerate(sub_indices):
                        tids = token_ids_list[orig_idx]
                        T = len(tids)
                        target = torch.tensor(tids[1:], dtype=torch.long, device=hidden.device)

                        # Gather hidden states for this sequence's K adapters
                        adapter_hidden = hidden[seq_pos * self.K:(seq_pos + 1) * self.K]  # (K, T, D)

                        # Chunked logprob + MI extraction
                        logprob_chunks = []
                        mi_chunks = []
                        for start in range(0, T - 1, chunk_size):
                            end = min(start + chunk_size, T - 1)
                            chunk_logits = lm_head(adapter_hidden[:, start:end, :]).float()
                            chunk_lp = F.log_softmax(chunk_logits, dim=-1)

                            # Gather logprobs at generated tokens
                            chunk_target = target[start:end].unsqueeze(0).unsqueeze(-1).expand(self.K, -1, 1)
                            logprob_chunks.append(chunk_lp.gather(2, chunk_target).squeeze(-1))

                            # True MI from full distribution
                            chunk_probs = chunk_lp.exp()
                            H_members = -(chunk_probs * chunk_lp).sum(dim=-1)  # (K, chunk)
                            log_mix = torch.logsumexp(chunk_lp, dim=0) - math.log(self.K)
                            mix_probs = log_mix.exp()
                            H_mixture = -(mix_probs * log_mix).sum(dim=-1)  # (chunk,)
                            mi_chunks.append(H_mixture - H_members.mean(dim=0))

                            del chunk_logits, chunk_lp, chunk_probs, H_members, log_mix, mix_probs, H_mixture

                        results[orig_idx] = (
                            torch.cat(logprob_chunks, dim=1),  # (K, T-1)
                            torch.cat(mi_chunks, dim=0),        # (T-1,)
                        )

                    del hidden
                    torch.cuda.empty_cache()
        finally:
            self.model.seq_module_.train()

        return results

    # ── Windowed MI scoring (for streaming early stop) ──────────────────

    @torch.no_grad()
    def compute_window_mi_batch(
        self,
        token_windows: List[List[int]],
        chunk_size: int = 256,
    ) -> List[float]:
        """
        Compute mean true MI for each token window.

        Used during streaming early-stop: at periodic checkpoints in the
        decode loop, the last W tokens of each active sequence are scored
        through all K adapters. Low MI = in-distribution → early stop.

        Processes one window at a time to keep VRAM predictable:
          Hidden: K × W × D × 2B ≈ 160MB
          Chunked LM head peak: K × 256 × V × 4B ≈ 780MB
          Total temporary: ~1GB

        Args:
            token_windows: List of token sequences (one per active sequence).
                Each is the last W tokens of the sequence generated so far.
            chunk_size: Tokens per chunk for LM head (lower = less memory).

        Returns:
            List of scalar mean MI values, one per input window.
        """
        results: List[float] = []

        self.model.seq_module_.eval()
        try:
            for window_tokens in token_windows:
                if len(window_tokens) < 2:
                    results.append(0.0)
                    continue

                mlora_data = self._build_mlora_data(window_tokens, "mi_window")
                mlora_data.return_hidden_states_ = True

                with torch.no_grad():
                    hidden = self.model.forward(mlora_data.model_data())  # (K, W, D)
                _, per_token_mi = self._chunked_logprobs(
                    hidden, window_tokens, chunk_size, compute_mi=True
                )
                results.append(float(compute_true_mi(per_token_mi).item()))
                del hidden, per_token_mi
        finally:
            self.model.seq_module_.train()

        return results

    # ── Optimizer management ──────────────────────────────────────────────

    def step_all_optimizers(self):
        """Step all K adapter optimizers and zero gradients."""
        for ctx in self.contexts:
            ctx.step()

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save(self, path: str, step: int):
        """Save all adapter weights."""
        for k, ctx in enumerate(self.contexts):
            save_dir = os.path.join(path, f"ensemble_{k}", f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(ctx.weight_dict(), os.path.join(save_dir, "adapter.pt"))

    def load(self, path: str, step: int):
        """Load adapter weights from checkpoint."""
        for k, ctx in enumerate(self.contexts):
            fpath = os.path.join(path, f"ensemble_{k}", f"step_{step}", "adapter.pt")
            if os.path.exists(fpath):
                ctx.recover_weight(torch.load(fpath, weights_only=True))
                logger.info(f"Loaded ensemble member {k} from {fpath}")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _expand_fn(
        batch_tokens: List[Tokens], align_len: Optional[int] = None
    ) -> Tuple[List[Tokens], List[Masks]]:
        """Pad tokens to uniform length."""
        if align_len is None:
            align_len = max(len(t) for t in batch_tokens)
        padded_tokens, padded_masks = [], []
        for tokens in batch_tokens:
            pad_len = align_len - len(tokens)
            padded_tokens.append(tokens + [0] * pad_len)
            padded_masks.append([True] * len(tokens) + [False] * pad_len)
        return padded_tokens, padded_masks

    @staticmethod
    def _noop_loss(
        input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None
