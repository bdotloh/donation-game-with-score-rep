"""Cross-check of the engine against an independent re-implementation (independent_model.py, written from the notes
and tests/reference/helper_funcs.jl without looking at engine/).  Fixed seeds, runs in well under a minute.
Uses tests/independent_model.py; not the engine's code."""
import csv
import os
import sys
import numpy as np
import pytest
from scipy.optimize import brentq, root
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
import engine as E                      # noqa: E402
import independent_model as im          # noqa: E402

TAUS = [1, 2, 3, 5, 8]
STEP1 = dict(max_iter=1, force_max_iter=True, stop_when_transient=False)
PAD = lambda a, L: np.hstack([a, np.zeros((a.shape[0], L - a.shape[1]))]) if a.shape[1] < L else a

# ------------------------------------------------------------------ random configurations
def rand_table(rng, C, p_float):
    ent = lambda: '%.12f' % rng.random() if rng.random() < p_float else rng.choice(['0', 'g'])
    return '|'.join(','.join(ent() for _ in range(C)) for _ in range(C))

def rand_barcode(rng, C):
    """valid barcode, flat order [need class][act][can class]; '-' only where the loser has class > 0"""
    parts = []
    for key in ('CN', 'NC'):
        s = ''
        for need in range(C):
            for act in range(2):
                for can in range(C):
                    opts = ['0', '+'] + (['-'] if (can if key == 'CN' else need) > 0 else [])
                    s += '0' if rng.random() < 0.3 else opts[rng.integers(len(opts))]
        parts.append(key + ':' + (s if set(s) != {'0'} else s[:-1] + '+'))
    return ' '.join(parts)

def rand_spec(rng, kind=None, n=None, tau=None):
    kind = kind or rng.choice(['binary', 'ternary', 'ternary', 'lifted'])
    tau = tau or int(rng.choice(TAUS))
    norm = (rand_barcode(rng, 2), None, False) if kind == 'binary' else \
           (rand_barcode(rng, 3), tau, False) if kind == 'ternary' else (rand_barcode(rng, 2), tau, True)
    n = n or int(rng.integers(1, 4))
    strats = []
    for _ in range(n):
        k = rng.choice(['bin', 'phi', 'defector', 'disc'] + (['ter', 'ter'] if norm[1] else []))
        p = rng.choice([0.0, 0.5, 1.0])
        strats.append(('bin', rand_table(rng, 2, p)) if k == 'bin' else ('ter', rand_table(rng, 3, p), norm[1]) if k == 'ter'
                      else ('phi', float(rng.uniform(0, 3))) if k == 'phi' else (str(k),))
    rho = rng.dirichlet(np.ones(n))
    if n > 1 and rng.random() < 0.2:
        rho[rng.integers(n)] = 0.0
        rho = rho / rho.sum() if rho.sum() > 0 else np.ones(n) / n
    eu = float(rng.uniform())
    return dict(norm=norm, strats=strats, rho=rho.tolist(), noise=(float(rng.choice([0.0, 0.1, 0.5, 1.0])), eu, 1 - eu))

def rand_w(rng, n, tau):
    L = int(rng.integers(1, 15))
    W = rng.random((n, L)) * (rng.random((n, L)) < 0.7)
    mode = rng.integers(3)
    if tau and mode == 1: W[:, :tau + 1] = 0.0                       # mass only in class 2
    if tau and mode == 2: W *= np.where(np.arange(L) > tau, 3.0, 0.3)  # plenty beyond tau
    for r in range(n):
        if W[r].sum() == 0: W[r, rng.integers(L)] = 1.0
    return W / W.sum(1, keepdims=True)

def mine(spec):
    s = []
    for x in spec['strats']:
        s.append(im.table_help(x[1]) if x[0] == 'bin' else im.table_help(x[1], x[2]) if x[0] == 'ter' else im.phi_disc(x[1])
                 if x[0] == 'phi' else im.table_help('0,0|0,0' if x[0] == 'defector' else '0,0|0,g'))
    code, tau, lifted = spec['norm']
    return s, im.parse_norm(im.lift_code(code) if lifted else code, tau)

def engine(spec):
    mk = {'bin': lambda x: E.BinaryStrategy.from_name(x[1]), 'ter': lambda x: E.TernaryStrategy.from_name(x[1], x[2]),
          'phi': lambda x: E.PhiDisc(x[1]), 'defector': lambda x: E.Defectors(), 'disc': lambda x: E.Discriminators()}
    code, tau, lifted = spec['norm']
    norm = E.ScoringNorm.from_barcode(code).to_ternary(tau) if lifted else E.ScoringNorm.from_barcode(code, tau)
    return E.Population([mk[x[0]](x) for x in spec['strats']], rho=spec['rho']), norm, E.Noise(*spec['noise'])

def solver(norm, noise, **kw):
    return E.StableScoreSolver(norm, noise, **kw)


# ------------------------------------------------------------------ the oracle itself against the user's Julia outputs
def julia_rows():
    for d in (os.path.join(HERE, 'reference'),):
        path = os.path.join(d, 'stable_cases.tsv')
        if os.path.exists(path):
            return list(csv.DictReader(open(path), delimiter='\t'))
    return []

def julia_like(strats, norm, noise, rho, w0, maxn, tol=1e-5):
    """helper_funcs.jl loop: `while counter < maxn && chg > tol`, chg = max |dw|; returns w and (alpha, beta, gamma) of the last step"""
    w = np.array(w0, float); n = len(strats); chg, count, trans = 1.0, 0, np.zeros((n, 3))
    K = im.kernel_fast(strats, norm, w.shape[1] + maxn + 1)
    while count < maxn and chg > tol:
        count += 1; L = w.shape[1]
        _, _, up, down = im.rates(K, noise, rho, w)
        trans = np.array([[up[i, 0], up[i, min(1, L - 1)], down[i, min(1, L - 1)]] for i in range(n)])
        dw = np.zeros((n, L + 1)); dw[:, :L] -= w * (up + down); dw[:, 1:] += w * up; dw[:, :L - 1] += (w * down)[:, 1:]
        w, chg = np.hstack([w, np.zeros((n, 1))]) + dw, np.abs(dw).max()
    return w, trans

@pytest.mark.skipif(not julia_rows(), reason='tests/reference/stable_cases.tsv not found')
def test_oracle_reproduces_julia_reference():
    fl = lambda t: np.array([float(x) for x in t.split()])
    rows = [r for k, r in enumerate(julia_rows()) if int(r['maxn']) <= 7 or (int(r['maxn']) <= 100 and k % 4 == 0)]
    assert len(rows) > 100
    for r in rows:
        norm = im.parse_norm(r['barcode'].replace('CmeetsN', 'CN').replace('NmeetsC', 'NC')); eps = float(r['noise'])
        names = [r['strategy_a']] + ([r['strategy_b']] if r['kind'] == 'two' else []); sfx = ['a', 'b'][:len(names)]
        rho = [float(r['rho_a']), 1 - float(r['rho_a'])] if r['kind'] == 'two' else [1.0]
        w0 = [fl(r['w_init_' + x]) for x in sfx]; L0 = max(len(x) for x in w0)
        w, trans = julia_like([im.table_help(x) for x in names], norm, (eps, 0.5, 0.5), rho,
                              np.array([np.pad(x, (0, L0 - len(x))) for x in w0]), int(r['maxn']))
        for i, x in enumerate(sfx):
            exp = fl(r['w_' + x]); L = max(len(exp), w.shape[1])
            assert np.abs(np.pad(w[i], (0, L - w.shape[1])) - np.pad(exp, (0, L - len(exp)))).max() < 1e-12
            if int(r['maxn']) > 0 and len(fl(r['trans_' + x])) == 3:
                assert np.abs(trans[i] - fl(r['trans_' + x])).max() < 1e-12

# ------------------------------------------------------------------ rates and chains
def test_rates_match_oracle():
    rng = np.random.default_rng(1)
    for _ in range(400):
        spec = rand_spec(rng); w = rand_w(rng, len(spec['strats']), spec['norm'][1]); L = w.shape[1]
        s, nm = mine(spec); pop, norm, noise = engine(spec)
        G, F, up, down = im.rates(im.kernel(s, nm, L), spec['noise'], spec['rho'], w)
        R = E.TransitionRates(norm, noise)(pop, w)
        for a, b in ((R.G[:, 1:L + 1], G), (R.F[:, :L], F), (R.g[:, 1:L + 1], up), (R.f[:, :L], down)):
            assert np.abs(a - b).max() < 1e-12

def test_chain_matches_oracle():
    rng = np.random.default_rng(2)
    for _ in range(60):
        spec = rand_spec(rng); w0 = rand_w(rng, len(spec['strats']), spec['norm'][1]); steps = 20
        s, nm = mine(spec); pop, norm, noise = engine(spec)
        res = solver(norm, noise, tol=1e-300, max_iter=steps, force_max_iter=True, stop_when_transient=False)(pop, w0)
        wm, wm_prev, _ = im.evolve(s, nm, spec['noise'], spec['rho'], w0, steps)
        L = max(wm.shape[1], res.w.shape[1])
        assert np.abs(PAD(res.w, L) - PAD(wm, L)).max() < 1e-12 and np.abs(PAD(res.w_prev, L) - PAD(wm_prev, L)).max() < 1e-12

@pytest.mark.parametrize('tau, steps', [(1, 30), (40, 12)])   # tau = 1; tau beyond every occupied score (class 2 empty)
def test_tau_edge_cases(tau, steps):
    rng = np.random.default_rng(3 + tau)
    for _ in range(25):
        spec = rand_spec(rng, kind='ternary', tau=tau); w0 = rand_w(rng, len(spec['strats']), tau)
        s, nm = mine(spec); pop, norm, noise = engine(spec)
        res = solver(norm, noise, tol=1e-300, max_iter=steps, force_max_iter=True, stop_when_transient=False)(pop, w0)
        wm = im.evolve(s, nm, spec['noise'], spec['rho'], w0, steps)[0]
        L = max(wm.shape[1], res.w.shape[1])
        assert np.abs(PAD(res.w, L) - PAD(wm, L)).max() < 1e-12

def test_rho_zero_strategy_does_not_affect_others():
    rng = np.random.default_rng(4); done = 0
    while done < 20:
        spec = rand_spec(rng, n=3)
        spec['rho'] = [0.6, 0.4, 0.0]
        sub = dict(spec, strats=spec['strats'][:2], rho=[0.6, 0.4]); w0 = rand_w(rng, 3, spec['norm'][1])
        out = []
        for sp, w in ((spec, w0), (sub, w0[:2])):
            pop, norm, noise = engine(sp)
            out.append(solver(norm, noise, tol=1e-300, max_iter=25, force_max_iter=True, stop_when_transient=False)(pop, w).w)
        L = max(o.shape[1] for o in out)
        assert np.abs(PAD(out[0], L)[:2] - PAD(out[1], L)).max() < 1e-13
        done += 1

def test_lifted_binary_equals_binary():
    rng = np.random.default_rng(5)
    for _ in range(25):
        code, tau = rand_barcode(rng, 2), int(rng.choice(TAUS)); strats = [E.BinaryStrategy.from_name(rand_table(rng, 2, 0.5)), E.PhiDisc(1.2)]
        rho, noise = [0.7, 0.3], E.Noise(0.2, 0.3, 0.7); w0 = rand_w(rng, 2, tau)
        kw = dict(tol=1e-300, max_iter=40, force_max_iter=True, stop_when_transient=False)
        ws = [solver(n, noise, **kw)(E.Population(strats, rho=rho), w0).w for n in
              (E.ScoringNorm.from_barcode(code), E.ScoringNorm.from_barcode(code).to_ternary(tau), E.ScoringNorm.from_barcode(im.lift_code(code), tau))]
        L = max(w.shape[1] for w in ws)
        assert all(np.abs(PAD(ws[0], L) - PAD(w, L)).max() < 1e-14 for w in ws[1:])

# ------------------------------------------------------------------ stationary distributions
def stable_oracle(spec, S=150):
    s, nm = mine(spec)
    w, ok, _ = im.stationary(s, nm, spec['noise'], spec['rho'], S=S)
    return (w if ok and w[:, -5:].sum() < 1e-15 else None), s, nm

def test_stationary_is_invariant_under_one_engine_step():
    rng = np.random.default_rng(6); n_checked = 0
    for _ in range(60):
        spec = rand_spec(rng)
        if spec['noise'][0] == 0: continue
        eu = 0.2 * rng.random(); spec['noise'] = (spec['noise'][0], eu, 1 - eu)    # eps_up << eps_down: stable regime
        w, s, nm = stable_oracle(spec, S=100)
        if w is None: continue
        pop, norm, noise = engine(spec)
        w1 = solver(norm, noise, **STEP1)(pop, w).w
        assert np.abs(PAD(w1, w.shape[1] + 1)[:, :w.shape[1]] - w).max() < 1e-12
        n_checked += 1
        if n_checked == 15: break
    assert n_checked >= 10

def test_engine_converges_to_oracle_stationary():
    rng = np.random.default_rng(7); n_checked = 0
    for _ in range(40):
        spec = rand_spec(rng, n=2); eu = 0.3 * rng.random(); spec['noise'] = (float(rng.choice([0.5, 1.0])), eu, 1 - eu)
        w, s, nm = stable_oracle(spec, S=80)
        if w is None: continue
        pop, norm, noise = engine(spec)
        res = solver(norm, noise, tol=1e-11, max_iter=3000)(pop, rand_w(rng, 2, spec['norm'][1]))
        assert res.converged and not res.drifting.any()
        L = max(w.shape[1], res.w.shape[1])
        assert np.abs(PAD(res.w, L) - PAD(w, L)).max() < 1e-8
        n_checked += 1
        if n_checked == 6: break
    assert n_checked >= 4

# ------------------------------------------------------------------ closed forms from the notes
def test_notes_page6_defectors_vs_discriminators():
    norm, noise = E.ScoringNorm.from_barcode('CN:00000-00 NC:00000000'), E.Noise(0.1, 0.4, 0.6)
    pop = E.Population([E.Defectors(), E.Discriminators()], rho=[0.5, 0.5])
    res = solver(norm, noise, tol=1e-12, max_iter=5000)(pop, np.full((2, 2), 0.5))   # relaxation time ~400 steps: ~1e-7 error left
    j = np.arange(res.w.shape[1])
    assert abs(res.w[0, 0] - 0.754095) < 1e-6 and abs(res.w[1, 0] - 1 / 3) < 1e-6
    assert np.allclose((res.w * j).sum(1), [0.326093, 2.0], atol=1e-5)
    s = [im.table_help('0,0|0,0'), im.table_help('0,0|0,g')]                           # oracle agrees
    w, ok, _ = im.stationary(s, im.parse_norm('CN:00000-00 NC:00000000'), (0.1, 0.4, 0.6), [0.5, 0.5], S=60)
    assert ok and np.allclose((w * np.arange(61)).sum(1), [0.326093, 2.0], atol=2e-6)

def phi_rates_notes(w0, phi, eps, eu, ed, L):
    """page 7: G_j = (1-w0)/(4 (j-1)^phi) (j > 1, G_1 = 0), F_j = (1-w0)/4 (1 - j^-phi); g, f with noise"""
    j = np.arange(1, L + 1.0)
    G = np.concatenate([[0.0], (1 - w0) / (4 * (j[1:] - 1) ** phi)])
    F = (1 - w0) / 4 * (1 - j[:L - 1] ** -phi)
    return (1 - eps) * G + eps * eu, (1 - eps) * F + eps * ed

def test_phi_disc_rates_match_page7():
    rng = np.random.default_rng(8); norm = E.ScoringNorm.from_barcode('CN:00000-0+ NC:00000000')
    for _ in range(200):
        phi, L = float(rng.uniform(0, 3)), int(rng.integers(2, 25)); eps, eu = float(rng.choice([0, 0.1, 0.5, 1])), float(rng.random())
        w = rng.random((1, L)); w /= w.sum()
        R = E.TransitionRates(norm, E.Noise(eps, eu, 1 - eu))(E.Population([E.PhiDisc(phi)], rho=[1.0]), w)
        g, f = phi_rates_notes(w[0, 0], phi, eps, eu, 1 - eu, L)
        assert np.abs(R.g[0, 1:L + 1] - g).max() < 1e-12 and np.abs(R.f[0, 1:L] - f).max() < 1e-12

def phi1_gamma(eps, eu, ed):
    """page 8, phi = 1: pi_0 from its normalisation equation (B depends on pi_0), pi_k from the gamma-function formula"""
    up, dn = eps * eu, eps * ed
    def terms(pi0):
        B = (1 - eps) * (1 - pi0) / 4; a, b, r = B / up, dn / (B + dn), up / (B + dn); k = np.arange(1, 3000)
        return np.exp(np.log(k) + (k - 1) * np.log(r) + gammaln(k + a) - gammaln(1 + a) + gammaln(1 + b) - gammaln(k + b)) * up / dn
    pi0 = brentq(lambda p: 1 / (1 + terms(p).sum()) - p, 1e-9, 1 - 1e-9, xtol=1e-15)
    pi = np.concatenate([[pi0], pi0 * terms(pi0)])
    return pi0, pi, (pi * np.arange(len(pi))).sum()

@pytest.mark.parametrize('eps, eu', [(0.1, 0.4), (0.1, 0.2), (0.3, 0.5), (0.5, 0.1), (1.0, 0.3)])
def test_phi1_gamma_solution(eps, eu):
    pi0, pi, mean = phi1_gamma(eps, eu, 1 - eu)
    if (eps, eu) == (0.1, 0.4):
        assert abs(pi0 - 0.2555) < 5e-5 and abs(mean - 1.889) < 5e-4          # numbers quoted in the notes
    pop, norm, noise = E.Population([E.PhiDisc(1.0)], rho=[1.0]), E.ScoringNorm.from_barcode('CN:00000-0+ NC:00000000'), E.Noise(eps, eu, 1 - eu)
    w1 = solver(norm, noise, **STEP1)(pop, pi[None, :300]).w                    # gamma solution is invariant under one engine step
    assert np.abs(w1[0, :300] - pi[:300]).max() < 1e-13
    res = solver(norm, noise, tol=1e-10, max_iter=4000)(pop, np.full((1, 2), 0.5))
    assert res.converged and np.abs(res.w[0, :60] - pi[:60]).max() < 1e-7

def test_phi0_is_discriminators():
    rng = np.random.default_rng(9)
    for _ in range(50):
        norm = E.ScoringNorm.from_barcode(rand_barcode(rng, 2)); noise = E.Noise(float(rng.choice([0, 0.2, 1])), 0.3, 0.7)
        w = rng.random((2, 9)); w /= w.sum(1, keepdims=True)
        R = [E.TransitionRates(norm, noise)(E.Population([a, E.Defectors()], rho=[0.4, 0.6]), w) for a in (E.PhiDisc(0), E.Discriminators())]
        assert all(np.abs(getattr(R[0], k) - getattr(R[1], k)).max() < 1e-15 for k in 'gfGF')

def class_rates(strat, norm, noise, tau, m):
    """single class-based strategy: up/down rate of a focal score in class c, for class masses m (representatives 0, 1, tau+1)"""
    eps, eu, ed = noise; reps = [0, 1] if tau is None else [0, 1, tau + 1]; u, d = [], []
    for j in reps:
        G, F = np.sum([np.array(im.pair([strat], norm, 0, j, 0, l)) * mc for mc, l in zip(m, reps)], axis=0)
        u.append((1 - eps) * G + eps * eu); d.append((1 - eps) * F + eps * ed)
    return np.array(u), np.array([0.0] + d[1:])

def segment_pi(u, d, tau):
    """three geometric segments: 0 | 1..tau (ratio u1/d1) | > tau (ratio u2/d2); returns pi(j) and the class masses"""
    r1, r2 = u[1] / d[1], u[2] / d[2]; a = u[0] / d[1]
    S1 = tau if abs(r1 - 1) < 1e-14 else (1 - r1 ** tau) / (1 - r1)
    top = a * r1 ** (tau - 1) * u[1] / d[2]
    pi0 = 1 / (1 + a * S1 + top / (1 - r2))
    pi = lambda j: pi0 * np.where(j == 0, 1.0, np.where(j <= tau, a * r1 ** np.maximum(j - 1, 0), top * r2 ** np.maximum(j - tau - 1, 0)))
    return pi, np.array([pi0, pi0 * a * S1, pi0 * top / (1 - r2)])

def test_ternary_class_based_single_strategy_three_segments():
    rng = np.random.default_rng(10); n_checked = 0
    for _ in range(60):
        tau = int(rng.choice(TAUS)); eu = float(rng.uniform(0.05, 0.5)); noise = (float(rng.choice([0.2, 0.5, 1.0])), eu, 1 - eu)
        spec = dict(norm=(rand_barcode(rng, 3), tau, False), strats=[('ter', rand_table(rng, 3, 0.5), tau)], rho=[1.0], noise=noise)
        s, nm = mine(spec)
        def resid(m):
            u, d = class_rates(s[0], nm, noise, tau, m)
            return m - segment_pi(u, d, tau)[1] if u[2] < d[2] and u[1] > 0 and u[0] >= 0 and d[1] > 0 else np.full(3, 9.0)
        for _ in range(30):                                                    # self-consistent class masses from random starts
            sol = root(resid, rng.dirichlet(np.ones(3)), tol=1e-14)
            if sol.success and np.abs(resid(sol.x)).max() < 1e-11 and sol.x.min() > -1e-12: break
        else: continue
        u, d = class_rates(s[0], nm, noise, tau, sol.x); pi = segment_pi(u, d, tau)[0](np.arange(600))
        if pi[-30:].sum() > 1e-15: continue
        pop, norm, nz = engine(spec)
        w1 = solver(norm, nz, **STEP1)(pop, pi[None, :]).w
        assert np.abs(w1[0, :600] - pi).max() < 1e-12
        n_checked += 1
    assert n_checked >= 10

# ------------------------------------------------------------------ eps = 0
def test_eps_zero_flags():
    norm = lambda c: E.ScoringNorm.from_barcode(c)
    w0 = np.array([[0.5, 0.3, 0.2]]); noise0 = E.Noise(0.0, 0.5, 0.5)
    # G > 0 = F: scores spread upward for ever
    up_only = solver(norm('CN:0000000+ NC:00000000'), noise0, tol=1e-10, max_iter=300)(E.Population([E.Discriminators()]), w0)
    assert up_only.drifting.all() and not up_only.converged
    s, nm = [im.table_help('0,0|0,g')], im.parse_norm('CN:0000000+ NC:00000000')
    _, _, up, down = im.rates(im.kernel_fast(s, nm, up_only.w.shape[1]), (0.0, 0.5, 0.5), [1.0], up_only.w)
    assert up[0, -1] > down[0, -1] == 0.0                                                 # my own analysis agrees
    # G = F = 0: nothing ever moves, every w is stationary, and the chain stops at once without drifting
    frozen = solver(norm('CN:00000-00 NC:00000000'), noise0, tol=1e-12, max_iter=50)(E.Population([E.Discriminators()]), w0)
    assert frozen.converged and not frozen.drifting.any() and np.allclose(frozen.w[0, :3], w0[0])
    # F > 0 = G: Defectors only go down; relaxation is algebraic (rate ~ 1 - w0), so check the path, not convergence
    s, nm = [im.table_help('0,0|0,0')], im.parse_norm('CN:00000-00 NC:00000000')
    losing = solver(norm('CN:00000-00 NC:00000000'), noise0, tol=1e-300, max_iter=400, force_max_iter=True, stop_when_transient=False)(E.Population([E.Defectors()]), w0)
    wm = im.evolve(s, nm, (0.0, 0.5, 0.5), [1.0], w0, 400)[0]
    assert not losing.drifting.any() and losing.w[0, 0] > 0.9 and np.abs(PAD(losing.w, wm.shape[1]) - PAD(wm, wm.shape[1])).max() < 1e-12
