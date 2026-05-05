"""
AC2 diversity portfolio analyzer
Compares ac2_baseline (5 LoRA + RMI, no NNM) vs
         ac2_rmi_nnm_streaming/non_streaming (5 LoRA + RMI + NNM, no streaming MI).

Follows the same two-axis approach as analyze_cp26_diversity.py:
  (1) OUTER SEARCH strategy family  (GA / SA / DE / CMA / gradient / PSO / ...)
  (2) PARAMETRIZATION template      (continuous-clip / binary-sign / sparse /
                                     gaussian-bump / symmetric / fourier /
                                     analytic-sequence / ...)

We report per-epoch:
  - counts and Shannon entropy over search families
  - counts and Shannon entropy over parametrization templates
  - mode-survival table (which families are still alive at each epoch)
  - adapter-level spread for the final epoch and for the top-30 rollouts
Meant to ground the FIGURE_IDEAS_AC2_DIVERSITY.txt narrative.
"""
import json, re, math, collections, os

BASE = '/home/kainat/neurips2026/discover/ac2_baseline'
METH = '/home/kainat/neurips2026/discover/ac2_rmi_nnm_streaming/non_streaming'


def extract_code(g):
    blocks = re.findall(r'```python\n(.*?)```', g, flags=re.DOTALL)
    return blocks[-1] if blocks else ''


def shannon(c):
    tot = sum(c.values())
    if tot == 0:
        return 0.0
    H = 0.0
    for v in c.values():
        if v > 0:
            p = v / tot
            H -= p * math.log2(p)
    return H


def tag_search(code):
    """Outer search / optimizer family. Priority-ordered so the *dominant*
    loop wins when several libraries are imported but only one is actually
    driving the generated solution."""
    c = code.lower()
    # explicit scipy wrappers first (they usually gate the whole solve)
    if 'dual_annealing' in c or 'simulated_annealing' in c or ('annealing' in c and 'schedule' in c):
        return 'sim_anneal'
    if 'differential_evolution' in c:
        return 'diff_evo'
    if 'basinhopping' in c:
        return 'basinhopping'
    if 'shgo' in c:
        return 'shgo'
    if 'cma' in c and ('cma.' in c or 'cmaes' in c or 'cma_es' in c or 'import cma' in c):
        return 'cma_es'
    if 'pyswarm' in c or ('particle' in c and 'swarm' in c) or re.search(r'\bpso\b', c):
        return 'particle_swarm'
    # genetic = tournament/crossover/mutation/elitism loop
    if ('genetic' in c
        or 'tournament' in c
        or ('crossover' in c and 'mutation' in c)
        or ('elite' in c and 'mutation' in c)):
        return 'genetic'
    # scipy minimize with a local method
    if ('slsqp' in c or 'l-bfgs' in c or 'l_bfgs' in c or 'lbfgsb' in c
        or 'nelder-mead' in c or 'nelder_mead' in c or 'trust-constr' in c
        or 'trust_constr' in c or 'cobyla' in c or 'powell' in c):
        return 'gradient_based'
    if 'adam' in c or ('gradient' in c and ('descent' in c or 'step' in c)):
        return 'gradient_based'
    if 'scipy.optimize.minimize' in c or ('minimize(' in c and 'scipy' in c):
        return 'gradient_based'
    if 'hill_climb' in c or 'hillclimb' in c or ('greedy' in c and 'flip' in c):
        return 'greedy_local'
    if 'random_search' in c or ('random' in c and 'restart' in c):
        return 'random_search'
    return 'other_search'


def tag_template(code):
    """Parametrization / structural template for the step function.
    Exactly one label per rollout — priority-ordered so the most distinctive
    structural choice wins."""
    c = code.lower()
    # analytic / number-theoretic sequences
    if ('rudin' in c or 'shapiro' in c or 'legendre' in c or 'barker' in c
        or 'chirp' in c or 'frank' in c):
        return 'analytic_seq'
    # fourier / harmonic basis
    if ('fourier' in c
        or ('np.sin' in c and 'np.cos' in c)
        or (re.search(r'np\.(sin|cos)\s*\(', code) and 'harmonic' in c)):
        return 'fourier_harmonic'
    # gaussian bumps
    if 'gaussian' in c or re.search(r'np\.exp\s*\(\s*-', code):
        return 'gaussian_bump'
    # binary / sign / ternary
    if ('np.sign' in c or 'bernoulli' in c or 'binary' in c
        or 'ternary' in c or re.search(r'\bchoice\s*\(\s*\[\s*-?1\s*,', code)):
        return 'binary_sign'
    # sparse / mostly-zeros init
    if 'sparse' in c or re.search(r'np\.zeros\s*\(', code):
        return 'sparse_zero'
    # explicit enforced symmetry (mirror / reverse / palindrome)
    if 'symmetric' in c or ('mirror' in c and 'array' in c) or 'palindrome' in c:
        return 'symmetric'
    # piecewise / step-block construction
    if 'piecewise' in c or ('block' in c and 'height' in c):
        return 'piecewise_block'
    # continuous-clipped float heights (the "vanilla" float vector)
    if 'np.clip' in c or 'clip(' in c:
        return 'continuous_clipped'
    # uniform-random float init with no extra structure
    if 'np.random.uniform' in c or 'random.uniform' in c:
        return 'random_uniform'
    return 'other_template'


def analyze(path, label):
    per_ep_search = collections.defaultdict(collections.Counter)
    per_ep_template = collections.defaultdict(collections.Counter)
    per_ep_combo = collections.defaultdict(collections.Counter)
    per_ep_search_correct = collections.defaultdict(collections.Counter)
    per_ep_template_correct = collections.defaultdict(collections.Counter)
    per_ep_correct = collections.defaultdict(int)
    per_ep_total = collections.defaultdict(int)
    per_ep_best = collections.defaultdict(float)
    per_ep_adp_search = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    top30 = []
    rows = 0
    with open(os.path.join(path, 'trajectories.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            rows += 1
            ep = d['epoch']
            adp = d['adapter_idx']
            r = d.get('reward_exec', 0.0)
            per_ep_total[ep] += 1
            if r > per_ep_best[ep]:
                per_ep_best[ep] = r
            code = extract_code(d['generated_text'])
            s = tag_search(code)
            t = tag_template(code)
            per_ep_search[ep][s] += 1
            per_ep_template[ep][t] += 1
            per_ep_combo[ep][(s, t)] += 1
            per_ep_adp_search[ep][adp][s] += 1
            if d.get('correctness', 0) > 0 and r >= 0.5:
                per_ep_correct[ep] += 1
                per_ep_search_correct[ep][s] += 1
                per_ep_template_correct[ep][t] += 1
            top30.append((r, ep, adp, s, t))
    top30.sort(reverse=True)
    top30 = top30[:30]
    all_search = collections.Counter()
    all_template = collections.Counter()
    for ep in per_ep_search:
        all_search.update(per_ep_search[ep])
        all_template.update(per_ep_template[ep])
    return dict(label=label, rows=rows,
                per_ep_search=per_ep_search, per_ep_template=per_ep_template,
                per_ep_combo=per_ep_combo,
                per_ep_search_correct=per_ep_search_correct,
                per_ep_template_correct=per_ep_template_correct,
                per_ep_correct=per_ep_correct, per_ep_total=per_ep_total,
                per_ep_best=per_ep_best,
                per_ep_adp_search=per_ep_adp_search,
                top30=top30, all_search=all_search, all_template=all_template)


def fmt_counter(c):
    return ' '.join(f'{k}:{v}' for k, v in c.most_common())


def print_res(res):
    print('\n' + '=' * 100)
    print(f"{res['label']}   (total rows: {res['rows']})")
    print('=' * 100)
    print("\nALL ROLLOUTS — per-epoch OUTER SEARCH family distribution")
    print("  ep  n_correct/total  best     search-family counts                                     H(bits)  modes")
    for ep in sorted(res['per_ep_search'].keys()):
        c = res['per_ep_search'][ep]
        print(f"  ep{ep} {res['per_ep_correct'][ep]:3d}/{res['per_ep_total'][ep]:3d}   "
              f"best={res['per_ep_best'][ep]:.4f}  {fmt_counter(c):70s}  "
              f"H={shannon(c):.3f}  modes={len(c)}")
    print(f"  ALL n={sum(res['all_search'].values())}  modes={len(res['all_search'])}  "
          f"H={shannon(res['all_search']):.3f}  dist={fmt_counter(res['all_search'])}")

    print("\nCORRECT-ish ROLLOUTS (r>=0.5) — per-epoch OUTER SEARCH family")
    for ep in sorted(res['per_ep_search_correct'].keys()):
        c = res['per_ep_search_correct'][ep]
        print(f"  ep{ep} n={sum(c.values()):3d}  {fmt_counter(c):70s}  "
              f"H={shannon(c):.3f}  modes={len(c)}")

    print("\nCORRECT-ish ROLLOUTS — per-epoch PARAMETRIZATION template family")
    for ep in sorted(res['per_ep_template_correct'].keys()):
        c = res['per_ep_template_correct'][ep]
        print(f"  ep{ep} n={sum(c.values()):3d}  {fmt_counter(c):70s}  "
              f"H={shannon(c):.3f}  modes={len(c)}")

    print("\nPER-ADAPTER SEARCH distribution (all rollouts, final epoch)")
    last_ep = max(res['per_ep_adp_search'].keys())
    for a, c in sorted(res['per_ep_adp_search'][last_ep].items()):
        print(f"  adapter{a}: {fmt_counter(c):60s}  H={shannon(c):.3f}")

    print("\nTOP-30 ROLLOUTS by reward_exec")
    for r, ep, adp, s, t in res['top30']:
        print(f"  r={r:.4f} ep{ep} adp{adp} search={s:15s} template={t}")
    adp = collections.Counter(x[2] for x in res['top30'])
    srch = collections.Counter(x[3] for x in res['top30'])
    tmpl = collections.Counter(x[4] for x in res['top30'])
    print(f"  TOP30 adapter spread : {dict(adp)}")
    print(f"  TOP30 search spread  : {dict(srch)}")
    print(f"  TOP30 template spread: {dict(tmpl)}")


def mode_survival(base, meth, which='per_ep_search_correct'):
    fams = sorted(set().union(base['all_search'].keys(), meth['all_search'].keys())
                  if 'search' in which
                  else set().union(base['all_template'].keys(), meth['all_template'].keys()))
    print(f"\nMODE-SURVIVAL TABLE — {which}")
    print(f"  family              | baseline epochs present   | method epochs present")
    for f in fams:
        b_eps = sorted(ep for ep, c in base[which].items() if c.get(f, 0) > 0)
        m_eps = sorted(ep for ep, c in meth[which].items() if c.get(f, 0) > 0)
        print(f"  {f:20s}| {str(b_eps):25s} | {m_eps}")


if __name__ == '__main__':
    base = analyze(BASE, 'BASELINE  (ac2_baseline, 5 LoRA + RMI, NO NNM)')
    meth = analyze(METH, 'METHOD    (ac2_rmi_nnm_streaming/non_streaming, 5 LoRA + RMI + NNM)')
    print_res(base)
    print_res(meth)
    mode_survival(base, meth, 'per_ep_search_correct')
    mode_survival(base, meth, 'per_ep_template_correct')
