# UG-TTT: Epistemic Uncertainty for Test-Time Discovery

Code release for the NeurIPS 2026 submission *Epistemic Uncertainty for Test-Time Discovery*.

UG-TTT augments test-time RL discovery with a small ensemble of LoRA adapters over a frozen base model. Token-level mutual information across the ensemble produces an epistemic-uncertainty signal that is added as an exploration bonus to the policy advantage, and a nuclear-norm regularizer on the stacked adapter matrices prevents the ensemble from collapsing to a shared weight configuration. The result is sustained higher solution-family diversity throughout training and improved maximum reward on three of four scientific-discovery benchmarks (AC1, AC2, CP26, Erdős minimum-overlap).

The paper PDF is included in this repository as [`UG_TTT.pdf`](UG_TTT.pdf).

## Setup

Requires a CUDA GPU and Python 3.10 or newer.

```bash
pip install -r requirements/requirements-math.txt
```

The mLoRA runtime is vendored under `mLoRA/`; the launcher adds it to `PYTHONPATH` automatically.

## Running the benchmarks

All four benchmarks share a single launcher:

```bash
scripts/run.sh cp     --problem_idx 26      # CP26 (circle packing)
scripts/run.sh ac1                          # AC1  (autocorrelation inequality)
scripts/run.sh ac2                          # AC2
scripts/run.sh erdos  --problem_idx 200     # Erdős minimum-overlap
```

The default flags reproduce the UG-TTT configuration reported in the paper: 5 LoRA adapters, MI exploration bonus (`rmi_coef=0.1`), nuclear-norm regularizer (`nnm_coef=0.075`), Qwen3-8B base, fp16, 6 epochs.

To run the single-adapter baseline (no exploration bonus, no diversity regularizer):

```bash
scripts/run.sh cp --problem_idx 26 \
    --num_ensemble_members 1 --rmi_coef 0 --nnm_coef 0
```

Training logs are written to `logs/<env>_rmi_nnm/` and the W&B run name is set automatically.

## Reproducing the paper figures

Once the runs above finish and artifacts land in `results/`:

```bash
python analyze_figure1.py            # Fig. 1: per-epoch entropy + final-epoch gains
python analyze_cp26_diversity.py
python analyze_ac2_diversity.py
python analyze_erdos_diversity.py
```

## Citation

```bibtex
@inproceedings{riaz2026ugttt,
  title     = {Epistemic Uncertainty for Test-Time Discovery},
  author    = {Riaz, Kainat and Mohsin, Muhammad Ahmed and Bilal, Ahsan and Umer, Muhammad and Mohsin, Ayesha and Riaz, Aqib and Subhan, Ali and Cioffi, John M.},
  booktitle = {Submitted to the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://kainatriaz98.github.io/uncertainty-guided-ttt/}
}
```

## License

The training code under `ug_ttt/`, `tasks/`, `utils/`, and `scripts/` is released under the Apache 2.0 license. The vendored `mLoRA/` runtime retains its upstream Apache 2.0 license; see `mLoRA/LICENSE`.
