# UG-TTT: Epistemic Uncertainty for Test-Time Discovery

Code for the NeurIPS 2026 submission *Epistemic Uncertainty for Test-Time Discovery*.

UG-TTT augments test-time RL discovery with a small ensemble of LoRA adapters over a frozen base model. Token-level mutual information across the ensemble produces an epistemic-uncertainty signal that is added as an exploration bonus to the policy advantage; a nuclear-norm regularizer on the stacked adapter matrices prevents the ensemble from collapsing to a shared weight configuration. The result is sustained higher solution-family diversity throughout training and improved maximum reward on three of four scientific-discovery benchmarks (AC1, AC2, CP26, Erdős minimum-overlap).

## Repository layout

```
ug_ttt/                 Core training package
  rl/mlora_train.py       Main entry point — multi-LoRA RL trainer
  rl/ensemble.py          K-adapter ensemble forward + per-adapter optimizer
  rl/uncertainty.py       Token-level MI / RMI / variance metrics
  rl/nuclear_norm.py      Nuclear-norm diversity regularizer on stacked LoRAs
  rl/{rollouts,metrics,data_processing,problem_env,types}.py
  recipes/ttt/            Per-task environments and state classes
    env_cp.py, env_ac.py, env_erdos.py, env_ttt.py
    sampler.py            PUCT / greedy state samplers
    state.py              State (de)serialization
mLoRA/                  Vendored multi-LoRA serving runtime (patched for K-adapter sampling)
tasks/                  Verifiers and prompts for each benchmark
  alphaevolve_ac/         AC1 (autocorrelation inequality)
  alphaevolve_ac2/        AC2
  alphaevolve_cp/         CP26 (circle packing)
  erdos_min_overlap/      Erdős minimum-overlap
utils/                  CPU scheduling helpers
scripts/run.sh          Launcher
analyze_*.py            Figure scripts for the paper
results/                Run artifacts referenced by the analyze_* scripts
assets/                 Figures shipped with the paper
```

## Setup

Requires a CUDA GPU and Python ≥ 3.10.

```bash
pip install -r requirements/requirements-math.txt
```

The mLoRA runtime is vendored under `mLoRA/`; add it to `PYTHONPATH` (the launcher does this).

## Running

The four benchmarks share one launcher:

```bash
scripts/run.sh cp     --problem_idx 26      # CP26
scripts/run.sh ac1                          # AC1
scripts/run.sh ac2                          # AC2
scripts/run.sh erdos  --problem_idx 200     # Erdős
```

Default flags reproduce the UG-TTT configuration: 5 LoRA adapters, MI exploration bonus (`rmi_coef=0.1`), nuclear-norm regularizer (`nnm_coef=0.075`), Qwen3-8B base, fp16, 10 epochs.

To run the baseline used in the paper (single adapter, no exploration bonus, no diversity regularizer):

```bash
scripts/run.sh cp --problem_idx 26 \
    --num_ensemble_members 1 --rmi_coef 0 --nnm_coef 0
```

Logs go to `logs/<env>_rmi_nnm/`; W&B run name is set automatically.

## Reproducing paper figures

After the runs finish (artifacts land in `results/`):

```bash
python analyze_figure1.py            # Fig. 1: per-epoch entropy + final-epoch gains
python analyze_cp26_diversity.py
python analyze_ac2_diversity.py
python analyze_erdos_diversity.py
```

## Citation

```bibtex
@inproceedings{ug_ttt_2026,
  title={Epistemic Uncertainty for Test-Time Discovery},
  author={Riaz, Kainat},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

The training code in `ug_ttt/`, `tasks/`, `utils/`, and `scripts/` is released under the Apache 2.0 license. The vendored `mLoRA/` runtime retains its upstream Apache 2.0 license — see `mLoRA/LICENSE`.
