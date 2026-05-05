"""
Erdos diversity portfolio analyzer
erdos_baseline (1 LoRA, no RMI/NNM, no streaming) vs
erdos_rmi_nnm_no_streaming_mi (5 LoRA + RMI + NNM + NO streaming)

Same axis-hunting as CP26: coarse approach family collapses to scipy_opt, so the
real portfolio lives at a finer level. We key on:
  (1) INIT family: analytic/step / uniform / random / gaussian / fourier / warm_start / bump
  (2) OPT / SOLVER: basinhopping / diff_evo / slsqp / l_bfgs_b / nelder_mead
                    / trust_constr / shgo / dual_annealing / correlate_trick / custom
  (3) RESOLUTION bucket: lo(<100) / mid(100..199) / hi(200..499) / ultra(>=500)
"""
import json, re, math, collections, os

BASE = '/home/kainat/neurips2026/discover/erdos_baseline'
METH = '/home/kainat/neurips2026/discover/erdos_rmi_nnm_no_streaming_mi'

def extract_code(g):
    blocks = re.findall(r'```python\n(.*?)```', g, flags=re.DOTALL)
    return blocks[-1] if blocks else ''

def shannon(c):
    tot = sum(c.values())
    if tot == 0: return 0.0
    H = 0.0
    for v in c.values():
        if v > 0:
            p = v/tot
            H -= p*math.log2(p)
    return H

def tag_init(code):
    c = code.lower()
    # priority: analytic/theoretical > specific shapes > random
    if 'step' in c and ('function' in c or 'h[' in c or 'characteristic' in c):
        return 'step_function'
    if 'piecewise' in c and ('constant' in c or 'linear' in c):
        return 'piecewise'
    if 'gaussian' in c or 'np.exp(-' in c:
        return 'gaussian_bump'
    if 'fourier' in c or 'harmonic' in c or re.search(r'np\.(sin|cos)\s*\(', code):
        return 'fourier'
    if 'bump' in c or 'triangular' in c or 'triangle' in c or 'tent' in c:
        return 'bump_triangle'
    if 'initial_h_values' in c or 'warm_start' in c or 'warm-start' in c:
        return 'warm_start'
    if 'np.random.uniform' in c or 'np.random.rand' in c:
        return 'random_init'
    if re.search(r'np\.full\(|np\.ones\(|np\.zeros\(', code):
        return 'constant_init'
    if 'linspace' in c and ('interp' in c or 'meshgrid' not in c):
        return 'linspace_based'
    return 'other_init'

def tag_opt(code):
    c = code.lower()
    if 'basinhopping' in c:
        return 'basinhopping'
    if 'differential_evolution' in c:
        return 'diff_evo'
    if 'dual_annealing' in c:
        return 'dual_annealing'
    if 'shgo' in c:
        return 'shgo'
    if 'slsqp' in c:
        return 'slsqp'
    if 'l-bfgs-b' in c or 'l_bfgs' in c or 'lbfgsb' in c:
        return 'l_bfgs_b'
    if 'nelder-mead' in c or 'nelder_mead' in c:
        return 'nelder_mead'
    if 'trust-constr' in c or 'trust_constr' in c:
        return 'trust_constr'
    if 'cobyla' in c:
        return 'cobyla'
    if 'powell' in c:
        return 'powell'
    if 'cvxpy' in c:
        return 'cvxpy'
    if 'scipy.optimize.minimize' in c or ('minimize(' in c and 'scipy' in c):
        return 'scipy_min_other'
    if 'genetic' in c or 'ga_' in c:
        return 'genetic'
    if 'gradient' in c and 'descent' in c:
        return 'grad_descent'
    return 'other_opt'

def detect_correlate(code):
    c = code.lower()
    return 'correlate(' in c or 'signal.correlate' in c

def detect_fft(code):
    c = code.lower()
    return 'np.fft' in c or 'scipy.fft' in c or 'rfft' in c

def npoints_bucket(n):
    if n is None: return 'unknown'
    if n < 100: return 'lo(<100)'
    if n < 200: return 'mid(100-199)'
    if n < 500: return 'hi(200-499)'
    return 'ultra(>=500)'

def extract_npoints(code):
    # find assignments like n_points = 200  (take the first non-trivial one)
    for m in re.finditer(r'n_points\s*=\s*(\d+)', code):
        val = int(m.group(1))
        if val >= 16:
            return val
    for m in re.finditer(r'\bN\s*=\s*(\d+)', code):
        val = int(m.group(1))
        if val >= 16:
            return val
    return None

def analyze(path, label):
    per_ep_init = collections.defaultdict(collections.Counter)
    per_ep_opt = collections.defaultdict(collections.Counter)
    per_ep_res = collections.defaultdict(collections.Counter)
    per_ep_flags = collections.defaultdict(collections.Counter)
    per_ep_correct = collections.defaultdict(int)
    per_ep_total = collections.defaultdict(int)
    per_ep_best_r = collections.defaultdict(float)
    per_ep_npts_mean = collections.defaultdict(list)
    per_ep_newbest = collections.defaultdict(int)
    running_best = -1
    top30 = []
    rows = 0
    with open(os.path.join(path, 'trajectories.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            rows += 1
            ep = d['epoch']
            adp = d['adapter_idx']
            r = d.get('reward_exec', 0)
            per_ep_total[ep] += 1
            if r > running_best:
                running_best = r
                per_ep_newbest[ep] += 1
            if r > per_ep_best_r[ep]:
                per_ep_best_r[ep] = r
            if d.get('correctness', 0) <= 0:
                continue
            per_ep_correct[ep] += 1
            code = extract_code(d['generated_text'])
            init = tag_init(code)
            opt = tag_opt(code)
            npts = extract_npoints(code)
            per_ep_npts_mean[ep].append(npts if npts is not None else 0)
            per_ep_init[ep][init] += 1
            per_ep_opt[ep][opt] += 1
            per_ep_res[ep][npoints_bucket(npts)] += 1
            if detect_correlate(code):
                per_ep_flags[ep]['correlate_trick'] += 1
            if detect_fft(code):
                per_ep_flags[ep]['fft'] += 1
            if 'fourier' in code.lower() or re.search(r'np\.(sin|cos)\s*\(', code):
                per_ep_flags[ep]['fourier'] += 1
            if 'initial_h_values' in code:
                per_ep_flags[ep]['warm_start'] += 1
            top30.append((r, ep, adp, init, opt, npts))
    top30.sort(reverse=True)
    top30 = top30[:30]
    return {
        'label': label, 'rows': rows,
        'per_ep_init': per_ep_init, 'per_ep_opt': per_ep_opt,
        'per_ep_res': per_ep_res, 'per_ep_flags': per_ep_flags,
        'per_ep_correct': per_ep_correct, 'per_ep_total': per_ep_total,
        'per_ep_best': per_ep_best_r,
        'per_ep_npts_mean': {e: (sum(v)/len(v) if v else 0) for e, v in per_ep_npts_mean.items()},
        'per_ep_newbest': per_ep_newbest,
        'top30': top30,
    }

def print_res(res):
    print(f"\n==== {res['label']} ==== (rows={res['rows']})")
    print("\nPer-epoch INIT:")
    for ep in sorted(res['per_ep_init'].keys()):
        c = res['per_ep_init'][ep]
        print(f"  ep{ep} ({res['per_ep_correct'][ep]}/{res['per_ep_total'][ep]}): {dict(c)}  H={shannon(c):.3f}  modes={len(c)}")
    print("\nPer-epoch OPT:")
    for ep in sorted(res['per_ep_opt'].keys()):
        c = res['per_ep_opt'][ep]
        print(f"  ep{ep}: {dict(c)}  H={shannon(c):.3f}  modes={len(c)}")
    print("\nPer-epoch RESOLUTION bucket:")
    for ep in sorted(res['per_ep_res'].keys()):
        c = res['per_ep_res'][ep]
        print(f"  ep{ep}: {dict(c)}  mean_npts={res['per_ep_npts_mean'].get(ep,0):.0f}")
    print("\nPer-epoch FLAGS:")
    for ep in sorted(res['per_ep_flags'].keys()):
        c = res['per_ep_flags'][ep]
        d = res['per_ep_correct'][ep]
        s = ' '.join(f"{k}={v}/{d}({v/max(d,1)*100:.0f}%)" for k,v in c.most_common())
        print(f"  ep{ep}: {s}")
    print("\nPer-epoch best reward / new-bests:")
    for ep in sorted(res['per_ep_best'].keys()):
        print(f"  ep{ep}: best={res['per_ep_best'][ep]:.4f} new_bests={res['per_ep_newbest'].get(ep,0)}")
    print("\nTop-30 rollouts:")
    for r,ep,adp,init,opt,n in res['top30']:
        print(f"  r={r:.4f} ep{ep} adp{adp} init={init:15s} opt={opt:15s} npts={n}")
    adp_c = collections.Counter(x[2] for x in res['top30'])
    init_c = collections.Counter(x[3] for x in res['top30'])
    opt_c = collections.Counter(x[4] for x in res['top30'])
    print(f"  TOP30 adapters: {dict(adp_c)}")
    print(f"  TOP30 init    : {dict(init_c)}")
    print(f"  TOP30 opt     : {dict(opt_c)}")

base = analyze(BASE, 'BASELINE (erdos_baseline, 1 LoRA, no RMI/NNM)')
meth = analyze(METH, 'METHOD   (erdos_rmi_nnm_no_streaming_mi, 5 LoRA + RMI + NNM)')
print_res(base); print_res(meth)
