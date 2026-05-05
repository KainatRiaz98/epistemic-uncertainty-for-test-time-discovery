"""
CP26 diversity portfolio analyzer
Compares cp26_rmi_baseline vs cp26_rmi_nnm with per-epoch approach distributions.
Classifies by both INITIALIZATION STRATEGY and OPTIMIZATION METHOD,
since both runs collapse to scipy_opt as outer optimizer.
"""
import json, re, math, collections, os, sys

BASE = '/home/kainat/neurips2026/discover/cp26_rmi_baseline'
METH = '/home/kainat/neurips2026/discover/cp26_rmi_nnm'


def extract_code(gen_text):
    blocks = re.findall(r'```python\n(.*?)```', gen_text, flags=re.DOTALL)
    return blocks[-1] if blocks else ''


def tag_init(code):
    """Initialization / geometric layout family."""
    c = code.lower()
    # priority ordered
    if 'hexagonal' in c or 'hex_grid' in c or 'hex grid' in c:
        return 'hex_grid'
    if 'force' in c and ('direct' in c or 'repuls' in c):
        return 'force_directed'
    if re.search(r'num_circles_per_row|circles_per_row|per_row\s*=\s*\[', code):
        return 'non_uniform_rows'
    # rows = [5,5,5,5,6] pattern detection
    if re.search(r'\[\s*5\s*,\s*5\s*,\s*5\s*,\s*5\s*,\s*6\s*\]', code):
        return 'non_uniform_rows'
    if 'lattice' in c or 'triangul' in c:
        return 'lattice'
    if re.search(r'5\s*x\s*5|5x5|5\*5', c) or '5, 5' in c:
        return 'uniform_rows_5x5'
    if 'grid' in c:
        return 'grid'
    if 'random' in c and 'uniform' in c:
        return 'random_init'
    if 'concentric' in c or 'ring' in c or 'nested' in c:
        return 'concentric'
    if 'boundary' in c or 'corner' in c:
        return 'boundary_fit'
    return 'other_init'


def tag_opt(code):
    c = code.lower()
    # priority ordered
    if 'differential_evolution' in c:
        return 'diff_evo'
    if 'basinhopping' in c:
        return 'basinhopping'
    if 'cvxpy' in c:
        return 'cvxpy'
    if 'simulated_annealing' in c or 'dual_annealing' in c:
        return 'sim_anneal'
    if 'scipy.optimize.minimize' in c or ('minimize(' in c and 'scipy' in c) or 'slsqp' in c or 'trust-constr' in c or 'trust_constr' in c:
        return 'scipy_min'
    if 'genetic' in c or 'ga_' in c or 'population' in c:
        return 'genetic'
    if 'gradient' in c and 'descent' in c:
        return 'grad_descent'
    if 'projected' in c:
        return 'projected_grad'
    if 'shgo' in c:
        return 'shgo'
    return 'other_opt'


def structural_flags(code):
    c = code.lower()
    flags = []
    # non-uniform rows: list like [5,5,5,5,6] or [4,5,4,5,4,4] or 26-specific
    if re.search(r'\[\s*\d+\s*(?:,\s*\d+\s*){3,}\]', code):
        # find such lists, check if sum==26 or contains unequal ints
        for m in re.finditer(r'\[((?:\s*\d+\s*,){2,}\s*\d+\s*)\]', code):
            nums = [int(x) for x in m.group(1).split(',') if x.strip().isdigit()]
            if len(nums) >= 3 and sum(nums) == 26 and len(set(nums)) > 1:
                flags.append('non_uniform_rows')
                break
    if 'variable' in c and ('radi' in c):
        flags.append('variable_radii')
    # detect whether all radii equal
    if re.search(r'np\.full\(\s*n\s*,', code) or re.search(r'np\.ones\(\s*n\s*\)\s*\*', code):
        flags.append('equal_radii')
    if 'sqrt' in c:
        flags.append('sqrt_analytic')
    if 'basinhopping' in c:
        flags.append('basinhopping_refine')
    if 'multi' in c and ('start' in c or 'stage' in c):
        flags.append('multistart')
    return flags


def shannon(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    H = 0.0
    for v in counter.values():
        if v == 0:
            continue
        p = v / tot
        H -= p * math.log2(p)
    return H


def analyze(path, label):
    per_epoch_init = collections.defaultdict(collections.Counter)
    per_epoch_opt = collections.defaultdict(collections.Counter)
    per_epoch_combo = collections.defaultdict(collections.Counter)
    per_epoch_correct = collections.defaultdict(int)
    per_epoch_total = collections.defaultdict(int)
    per_epoch_flags = collections.defaultdict(collections.Counter)
    per_epoch_adapter_init = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    top30 = []  # (reward, epoch, adapter, init, opt)
    rows = 0
    traj_path = os.path.join(path, 'trajectories.jsonl')
    with open(traj_path) as f:
        for line in f:
            d = json.loads(line)
            rows += 1
            ep = d['epoch']
            adp = d['adapter_idx']
            correct = d.get('correctness', 0) > 0
            per_epoch_total[ep] += 1
            if not correct:
                continue
            per_epoch_correct[ep] += 1
            code = extract_code(d['generated_text'])
            init = tag_init(code)
            opt = tag_opt(code)
            per_epoch_init[ep][init] += 1
            per_epoch_opt[ep][opt] += 1
            per_epoch_combo[ep][(init, opt)] += 1
            per_epoch_adapter_init[ep][adp][init] += 1
            for fl in structural_flags(code):
                per_epoch_flags[ep][fl] += 1
            top30.append((d.get('reward_exec', 0), ep, adp, init, opt))
    top30.sort(reverse=True)
    top30 = top30[:30]
    # aggregate all-epoch entropy
    all_init = collections.Counter()
    all_opt = collections.Counter()
    all_combo = collections.Counter()
    for ep in per_epoch_init:
        all_init.update(per_epoch_init[ep])
        all_opt.update(per_epoch_opt[ep])
        all_combo.update(per_epoch_combo[ep])
    return {
        'label': label, 'rows': rows,
        'per_epoch_init': dict(per_epoch_init),
        'per_epoch_opt': dict(per_epoch_opt),
        'per_epoch_combo': dict(per_epoch_combo),
        'per_epoch_correct': dict(per_epoch_correct),
        'per_epoch_total': dict(per_epoch_total),
        'per_epoch_flags': dict(per_epoch_flags),
        'per_epoch_adapter_init': {ep: {a: dict(c) for a, c in ad.items()} for ep, ad in per_epoch_adapter_init.items()},
        'top30': top30,
        'all_init': all_init, 'all_opt': all_opt, 'all_combo': all_combo,
    }


def print_table(res):
    print(f"\n=== {res['label']} === (total rows: {res['rows']})")
    print("epoch  correct/total  | INIT distribution                      H_init  | OPT distribution                 H_opt  | COMBO modes H_combo")
    for ep in sorted(res['per_epoch_init'].keys()):
        init_c = res['per_epoch_init'][ep]
        opt_c = res['per_epoch_opt'][ep]
        combo_c = res['per_epoch_combo'][ep]
        c = res['per_epoch_correct'][ep]
        t = res['per_epoch_total'][ep]
        init_s = ' '.join(f"{k}{v}" for k, v in init_c.most_common())
        opt_s = ' '.join(f"{k}{v}" for k, v in opt_c.most_common())
        print(f"ep{ep}  {c}/{t}  | {init_s:60s} H={shannon(init_c):.3f} | {opt_s:40s} H={shannon(opt_c):.3f} | modes={len(combo_c)} H_combo={shannon(combo_c):.3f}")
    print(f"ALL    n={sum(res['all_init'].values())} | init modes={len(res['all_init'])} H_init={shannon(res['all_init']):.3f} | opt modes={len(res['all_opt'])} H_opt={shannon(res['all_opt']):.3f} | combo modes={len(res['all_combo'])} H_combo={shannon(res['all_combo']):.3f}")
    print("\nSTRUCTURAL FLAGS PER EPOCH:")
    for ep in sorted(res['per_epoch_flags'].keys()):
        flags = res['per_epoch_flags'][ep]
        denom = res['per_epoch_correct'][ep]
        s = ' '.join(f"{k}={v}/{denom}({v/max(denom,1)*100:.0f}%)" for k, v in flags.most_common())
        print(f"  ep{ep}: {s}")
    print("\nTOP-30 ROLLOUTS by reward_exec:")
    for r, ep, adp, init, opt in res['top30']:
        print(f"  r={r:.4f} ep{ep} adp{adp}  init={init:20s} opt={opt}")
    # top-30 adapter spread
    adp_count = collections.Counter(x[2] for x in res['top30'])
    init_count = collections.Counter(x[3] for x in res['top30'])
    opt_count = collections.Counter(x[4] for x in res['top30'])
    print(f"  TOP30 adapter spread: {dict(adp_count)}")
    print(f"  TOP30 init spread   : {dict(init_count)}")
    print(f"  TOP30 opt spread    : {dict(opt_count)}")
    print("\nPER-ADAPTER INIT (final epoch):")
    last_ep = max(res['per_epoch_adapter_init'].keys())
    for a, c in sorted(res['per_epoch_adapter_init'][last_ep].items()):
        cc = collections.Counter(c)
        print(f"  adapter{a}: {dict(cc)}  H={shannon(cc):.3f}")


if __name__ == '__main__':
    base = analyze(BASE, 'BASELINE (cp26_rmi_baseline, 5 LoRA + RMI)')
    meth = analyze(METH, 'METHOD   (cp26_rmi_nnm,      5 LoRA + RMI + NNM)')
    print_table(base)
    print_table(meth)
