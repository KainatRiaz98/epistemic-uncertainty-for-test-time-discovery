"""
Figure 1 analyzer: builds the numerical labels tying MI to discovery across
AC1, CP26, Erdos for {baseline, method}.

Outputs a JSON dump + printable summary covering every panel:
  A. per-epoch mean uncertainty_true_mi                (method vs baseline)
  B. per-rollout MI conditional on template family     (novel vs collapsed)
  C. family composition at the final logged epoch      (stacked-bar inputs)
  D. discovery metrics: steps-to-threshold, correct@ep5, new-best count
  E. top-k exemplars for the "receipts" panel
"""
import json, os, re, math, collections, statistics

ROOT = '/home/kainat/neurips2026/discover'

EXPERIMENTS = {
    'AC1': {
        'baseline': ('ac1_baseline', 1),
        'method':   ('ac1_rmi_nnm_streaming', 5),
    },
    'CP26': {
        'baseline': ('cp26_rmi_baseline', 5),
        'method':   ('cp26_rmi_nnm', 5),
    },
    'Erdos': {
        'baseline': ('erdos_baseline', 1),
        'method':   ('erdos_rmi_nnm_no_streaming_mi', 5),
    },
}

# ---------------------------------------------------------------- code helpers
def extract_code(gen_text):
    blocks = re.findall(r'```python\n(.*?)```', gen_text, flags=re.DOTALL)
    return blocks[-1] if blocks else ''

# ------------------------------------------------------ AC1 family taggers ---
# AC1 problem: non-negative step-function sequence of length 1000 minimising
#   2 n * max(conv(a,a)) / (sum(a))**2
# Known hot templates:
#   - genetic algorithm / mutate+crossover                           (baseline mode)
#   - target_sum = sqrt(2 n) analytical Cauchy-Schwarz trick         (method discovery)
#   - cvxpy linear program / semidefinite style                      (method)
#   - direct scipy.optimize.minimize over the 1000-D sequence        (method)
def ac1_tag_family(code):
    c = code.lower()
    if re.search(r'np\.sqrt\s*\(\s*2\s*\*?\s*n\s*\)', code) or 'sqrt(2*n)' in c or 'sqrt(2n)' in c:
        return 'sqrt_2n_analytic'
    if 'cvxpy' in c:
        return 'cvxpy_lp'
    if 'differential_evolution' in c:
        return 'diff_evo'
    if 'basinhopping' in c:
        return 'basinhopping'
    if 'genetic' in c or ('population' in c and ('mutate' in c or 'crossover' in c)):
        return 'genetic'
    if 'mutate' in c and 'crossover' in c:
        return 'genetic'
    if 'scipy.optimize.minimize' in c or ('minimize(' in c and 'scipy' in c) or 'l-bfgs-b' in c or 'slsqp' in c:
        return 'scipy_min'
    if 'simulated_annealing' in c or 'dual_annealing' in c:
        return 'sim_anneal'
    if 'gradient' in c and 'descent' in c:
        return 'grad_descent'
    return 'other'

AC1_NOVEL = {'sqrt_2n_analytic', 'cvxpy_lp', 'scipy_min', 'sim_anneal', 'basinhopping'}
AC1_COLLAPSED = {'genetic', 'grad_descent', 'diff_evo', 'other'}

# ------------------------------------------------------ CP26 family taggers --
# CP26: 26 unit-sum circles in a rectangle, maximise sum of radii.
# Key axes: init layout (grid / non_uniform_rows / hex), optimiser family,
# and structural choice {variable_radii vs equal_radii}.
def cp26_tag_family(code):
    c = code.lower()
    # variable radii is the *conceptual* template that unlocks the correct formulation
    if ('variable' in c and 'radi' in c) or re.search(r'radii\s*=\s*x\[', c):
        return 'variable_radii'
    if re.search(r'num_circles_per_row|circles_per_row|per_row\s*=\s*\[', code):
        return 'non_uniform_rows'
    if re.search(r'\[\s*5\s*,\s*5\s*,\s*5\s*,\s*5\s*,\s*6\s*\]', code):
        return 'non_uniform_rows'
    if 'hexagonal' in c or 'hex_grid' in c or 'hex grid' in c:
        return 'hex_grid'
    if 'concentric' in c or 'ring' in c:
        return 'concentric'
    if re.search(r'np\.full\(\s*n\s*,', code) or re.search(r'np\.ones\(\s*n\s*\)\s*\*', code):
        return 'equal_radii'
    if re.search(r'5\s*x\s*5|5x5|5\*5', c) or '5, 5, 5, 5, 6' in c:
        return 'uniform_rows_5x5'
    if 'grid' in c:
        return 'grid'
    if 'force' in c and ('direct' in c or 'repuls' in c):
        return 'force_directed'
    return 'other'

CP26_NOVEL = {'variable_radii', 'non_uniform_rows', 'hex_grid', 'concentric', 'force_directed'}
CP26_COLLAPSED = {'equal_radii', 'uniform_rows_5x5', 'grid', 'other'}

# ------------------------------------------------------ Erdos family taggers -
# Erdos: maximise correlation / step-function-like functional.  Method discovers
# fourier and bump_triangle inits that the 1-LoRA baseline never samples.
def erdos_tag_family(code):
    c = code.lower()
    if 'fourier' in c or 'harmonic' in c or re.search(r'np\.(sin|cos)\s*\(', code):
        return 'fourier'
    if 'bump' in c or 'triangular' in c or 'triangle' in c or 'tent' in c:
        return 'bump_triangle'
    if 'gaussian' in c or 'np.exp(-' in c:
        return 'gaussian_bump'
    if 'step' in c and ('function' in c or 'characteristic' in c):
        return 'step_function'
    if 'piecewise' in c:
        return 'piecewise'
    if 'initial_h_values' in c or 'warm_start' in c or 'warm-start' in c:
        return 'warm_start'
    if 'np.random.uniform' in c or 'np.random.rand' in c:
        return 'random_init'
    if re.search(r'np\.full\(|np\.ones\(|np\.zeros\(', code):
        return 'constant_init'
    if 'linspace' in c:
        return 'linspace_based'
    return 'other'

ERDOS_NOVEL = {'fourier', 'bump_triangle', 'gaussian_bump', 'piecewise', 'warm_start'}
ERDOS_COLLAPSED = {'step_function', 'random_init', 'constant_init', 'linspace_based', 'other'}

TAGGERS = {
    'AC1':   (ac1_tag_family,   AC1_NOVEL,   AC1_COLLAPSED),
    'CP26':  (cp26_tag_family,  CP26_NOVEL,  CP26_COLLAPSED),
    'Erdos': (erdos_tag_family, ERDOS_NOVEL, ERDOS_COLLAPSED),
}

# ---------------------------------------------------------- loader / crunch --
def load_rollouts(path):
    rows = []
    with open(os.path.join(path, 'trajectories.jsonl')) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def shannon(counter):
    tot = sum(counter.values())
    if tot == 0: return 0.0
    H = 0.0
    for v in counter.values():
        if v == 0: continue
        p = v / tot
        H -= p * math.log2(p)
    return H

def summarise_run(bench, variant, folder, n_lora):
    tag_fn, NOVEL, COLLAPSED = TAGGERS[bench]
    rows = load_rollouts(os.path.join(ROOT, folder))

    per_ep_mi           = collections.defaultdict(list)
    per_ep_mi_valid     = collections.defaultdict(list)    # MI conditioned on correctness=1
    per_ep_family       = collections.defaultdict(collections.Counter)
    per_ep_correct      = collections.defaultdict(int)
    per_ep_total        = collections.defaultdict(int)
    per_ep_max_reward   = collections.defaultdict(float)
    per_ep_mean_reward  = collections.defaultdict(list)
    mi_by_family_bucket = {'novel': [], 'collapsed': []}
    cumulative_best     = -1.0
    per_ep_newbest      = collections.defaultdict(int)
    steps_seen_reward   = []  # (global_step_proxy, reward) for steps-to-threshold
    top_rollouts        = []

    # order rows so step index is sensible; trajectories are written in order
    for idx, d in enumerate(rows):
        ep  = d['epoch']
        adp = d['adapter_idx']
        r   = d.get('reward_exec', 0.0)
        mi  = d.get('uncertainty_true_mi', 0.0)
        correct = d.get('correctness', 0) > 0

        per_ep_total[ep]       += 1
        per_ep_mean_reward[ep].append(r)
        per_ep_mi[ep].append(mi)
        if r > per_ep_max_reward[ep]:
            per_ep_max_reward[ep] = r
        if r > cumulative_best:
            cumulative_best = r
            per_ep_newbest[ep] += 1
        steps_seen_reward.append((idx, r))

        if not correct:
            continue
        per_ep_correct[ep] += 1
        per_ep_mi_valid[ep].append(mi)

        code   = extract_code(d['generated_text'])
        family = tag_fn(code)
        per_ep_family[ep][family] += 1
        if family in NOVEL:
            mi_by_family_bucket['novel'].append(mi)
        elif family in COLLAPSED:
            mi_by_family_bucket['collapsed'].append(mi)

        top_rollouts.append({
            'reward': r, 'epoch': ep, 'adapter': adp,
            'family': family, 'mi': mi, 'idx': idx,
        })

    top_rollouts.sort(key=lambda x: x['reward'], reverse=True)

    # steps-to-reach-threshold (AC1 uses 0.63 and 0.64)
    thresholds = {'AC1': [0.62, 0.63, 0.64], 'CP26': [2.60, 2.62, 2.63], 'Erdos': [2.60, 2.61, 2.615]}
    hits = {}
    for thr in thresholds[bench]:
        hit_idx = None
        for i, r in steps_seen_reward:
            if r >= thr:
                hit_idx = i
                break
        hits[thr] = hit_idx

    return {
        'bench': bench, 'variant': variant, 'folder': folder, 'n_lora': n_lora,
        'n_rows': len(rows),
        'per_ep_mean_mi':      {ep: float(statistics.mean(v))  for ep, v in per_ep_mi.items()},
        'per_ep_median_mi':    {ep: float(statistics.median(v)) for ep, v in per_ep_mi.items()},
        'per_ep_mean_mi_correct': {ep: float(statistics.mean(v)) if v else None for ep, v in per_ep_mi_valid.items()},
        'per_ep_family':       {ep: dict(c) for ep, c in per_ep_family.items()},
        'per_ep_family_entropy': {ep: shannon(c) for ep, c in per_ep_family.items()},
        'per_ep_correct':      dict(per_ep_correct),
        'per_ep_total':        dict(per_ep_total),
        'per_ep_max_reward':   dict(per_ep_max_reward),
        'per_ep_mean_reward':  {ep: float(statistics.mean(v)) for ep, v in per_ep_mean_reward.items()},
        'per_ep_newbest':      dict(per_ep_newbest),
        'cumulative_best':     cumulative_best,
        'total_newbests':      sum(per_ep_newbest.values()),
        'mi_novel_mean':       (float(statistics.mean(mi_by_family_bucket['novel']))     if mi_by_family_bucket['novel']     else None),
        'mi_novel_median':     (float(statistics.median(mi_by_family_bucket['novel']))   if mi_by_family_bucket['novel']     else None),
        'mi_novel_n':          len(mi_by_family_bucket['novel']),
        'mi_collapsed_mean':   (float(statistics.mean(mi_by_family_bucket['collapsed'])) if mi_by_family_bucket['collapsed'] else None),
        'mi_collapsed_median': (float(statistics.median(mi_by_family_bucket['collapsed'])) if mi_by_family_bucket['collapsed'] else None),
        'mi_collapsed_n':      len(mi_by_family_bucket['collapsed']),
        'hits_to_threshold':   hits,
        'top10':               top_rollouts[:10],
        'top30_family_counts': dict(collections.Counter(r['family'] for r in top_rollouts[:30])),
    }

def main():
    out = {}
    for bench, variants in EXPERIMENTS.items():
        out[bench] = {}
        for variant, (folder, n_lora) in variants.items():
            out[bench][variant] = summarise_run(bench, variant, folder, n_lora)
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(ROOT, 'figure1_stats.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

if __name__ == '__main__':
    main()
