"""
Nuclear norm diversity regularization for LoRA ensemble.

Encourages ensemble members to span different directions in parameter space
by maximizing the nuclear norm of the stacked lora_A weight matrix.

Math (from UP-RLHF utils/nnm_utils.py — nuclear_norm_maximization):
    For each (layer, module) pair:
        W = cat([lora_A_k for k in 0..K-1], dim=0)   shape: (K*rank, in_dim)
        reg = -nuclear_norm(W)                          default
        reg = -nuclear_norm(W) / fro_norm(W) * sqrt(max_dim)   use_fro_norm=True
    nnm_loss = mean(reg over all layers)

When adapters are diverse, W has high rank → high nuclear norm → low nnm_loss.
Minimising nnm_loss (adding nnm_coef * nnm_loss to the total loss) pushes the
ensemble toward diversity.

Key difference from UP-RLHF: weight extraction goes through
    ensemble.contexts[k].adapter_model_[linear_name].lora_a_
instead of model.named_parameters(), because mLoRA stores adapter weights
inside TrainLoRAContext objects, not as named parameters on the model.
"""

import math
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from .ensemble import LoRAEnsemble


def get_ensemble_lora_a_stacked(
    ensemble: "LoRAEnsemble",
) -> Dict[str, torch.Tensor]:
    """
    Build a dict mapping each linear layer name to a stacked lora_A matrix.

    For each linear tracked by mLoRA (e.g. 'layers.0.self_attn.q_proj'),
    collects lora_A from all K adapters and concatenates along dim=0:
        stacked shape: (K * rank, in_dim)

    Returns:
        dict mapping linear_name → stacked tensor of shape (K * rank, in_dim)
    """
    per_layer: Dict[str, List[torch.Tensor]] = {}

    for ctx in ensemble.contexts:
        for linear_name, lora_module in ctx.adapter_model_.items():
            if linear_name not in per_layer:
                per_layer[linear_name] = []
            per_layer[linear_name].append(lora_module.lora_a_)

    # Only include layers where all K adapters contributed
    return {
        name: torch.cat(tensors, dim=0)
        for name, tensors in per_layer.items()
        if len(tensors) == ensemble.K
    }


def compute_nuclear_norm_diversity_loss(
    ensemble: "LoRAEnsemble",
    use_fro_norm: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute the nuclear norm diversity loss for the ensemble.

    Adapted from UP-RLHF utils/nnm_utils.py: nuclear_norm_maximization().

    Args:
        ensemble: LoRAEnsemble with K adapters.
        use_fro_norm: if True, normalise nuclear norm by Frobenius norm
                      (scale-invariant; matches UP-RLHF peft/utils_NNM.py variant).

    Returns:
        (nnm_loss, mean_nuclear_norm, mean_fro_norm)
        nnm_loss is a scalar tensor with gradients flowing to all K lora_A tensors.
        Add `nnm_coef * nnm_loss` to the total loss (it is already negative).
    """
    weight_dict = get_ensemble_lora_a_stacked(ensemble)

    if not weight_dict:
        dummy = torch.zeros(1, requires_grad=True)
        return dummy, 0.0, 0.0

    reg_loss: torch.Tensor = torch.zeros(1, device=next(iter(weight_dict.values())).device)
    nuc_total = 0.0
    fro_total = 0.0
    n = len(weight_dict)

    for weight in weight_dict.values():
        # Cast to float32 for numerical stability (same as UP-RLHF)
        w = weight.float()
        w_nuc = torch.norm(w, p="nuc")
        w_fro = torch.norm(w, p="fro")

        if use_fro_norm:
            max_dim = max(w.shape)
            reg = -w_nuc / (w_fro.detach() + 1e-8) * math.sqrt(max_dim)
        else:
            reg = -w_nuc

        reg_loss = reg_loss + reg
        nuc_total += w_nuc.detach().item()
        fro_total += w_fro.detach().item()

    nnm_loss = reg_loss / n
    return nnm_loss, nuc_total / n, fro_total / n
