"""Agent-based check of the whole score model, independent of the rate formulas.

A population of N = 1e5 agents is simulated directly from the verbal model (notes pp. 1-4, 7-8):
every step each agent draws a uniformly random partner, its own role and the partner's role (each Can with
probability 1/2, independently; an interaction happens only if exactly one is Can), the Can helps with the
probability given by its strategy, the focal agent's score changes by the scoring norm
(CN[c_need][act][c_can] if it is the Can, NC[c_need][act][c_can] if it is the Need), and with probability eps
the change is replaced by +1 (eps_up) or -1 (eps_down; nothing at score 0).  All agents update at once from
the start-of-step state.  The score histograms are compared with `StableScoreSolver(..., max_iter=t,
force_max_iter=True)` after the same number of steps.

The simulation below uses only numpy and none of the engine's rate code.  Tolerances come from sampling noise:
 * every bin with engine probability > 2e-3: |q - p| <= 6 sqrt(p (1 - p) / n_k)
 * total variation distance <= mean + 6 sd of the TV of a pure multinomial sample drawn from the engine's p
 * mean score and share at score 0: within 6 standard errors
All seeds are fixed.  Runtime about 1 minute.
"""
import numpy as np
import pytest

from engine import (BinaryStrategy, Noise, PhiDisc, Population, ScoringNorm, StableScoreSolver,
                    TernaryStrategy)

N = 100_000
ZMAX = 6.0
PMIN = 2e-3
CHAR = {"-": -1, "0": 0, "+": 1}


# ------------------------------------------------------------------ agent-based simulation
def _norm_arrays(code):
    cn, nc = code.split()
    cn, nc = cn[3:], nc[3:]
    K = 2 if len(cn) == 8 else 3
    return [np.array([CHAR[c] for c in s], dtype=np.int32).reshape(K, 2, K) for s in (cn, nc)], K


def _table(name, K):
    P = np.array([[1.0 if x == "g" else float(x) for x in row.split(",")] for row in name.split("|")])
    if P.shape[0] == 2 and K == 3:                      # binary strategy in a ternary population: lift
        m = [0, 1, 1]
        P = P[np.ix_(m, m)]
    assert P.shape == (K, K)
    return P


def simulate(strategies, rho, norm, tau, eps, eps_up, eps_down, w_init, times, seed, n=N):
    """strategies: ('table', 'a,b|c,d') or ('phi', phi).  Returns {t: [score counts of each strategy]}, counts."""
    rng = np.random.default_rng(seed)
    (CN, NC), K = _norm_arrays(norm)
    ns = len(strategies)
    P = np.zeros((ns, K, K))
    phi = np.zeros(ns)
    is_phi = np.zeros(ns, dtype=bool)
    for k, (kind, spec) in enumerate(strategies):
        if kind == "phi":
            is_phi[k], phi[k] = True, spec
        else:
            P[k] = _table(spec, K)
    counts = np.round(np.asarray(rho) * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    strat = np.repeat(np.arange(ns), counts)
    w_init = np.asarray(w_init, float)
    edges = lambda c: np.diff(np.concatenate([[0], np.round(np.cumsum(w_init / w_init.sum()) * c).astype(int)]))
    score = np.concatenate([np.repeat(np.arange(len(w_init)), edges(c)) for c in counts])
    perm = rng.permutation(n)
    strat, score = strat[perm], score[perm].astype(np.int32)
    idx = np.arange(n)
    cls = (lambda s: np.minimum(s, 1)) if K == 2 else (lambda s: (s > 0).astype(int) + (s > tau))
    out, t = {}, 0
    for T in sorted(times):
        while t < T:
            m = idx + rng.integers(1, n, size=n)        # a uniformly random other agent
            m[m >= n] -= n
            u_help, u_noise = rng.random(n), rng.random(n)
            own_can, par_can = rng.random(n) < 0.5, rng.random(n) < 0.5
            c = cls(score)
            sm, stm, cm = score[m], strat[m], c[m]
            d_st, d_s, d_c = (np.where(own_can, a, b) for a, b in ((strat, stm), (score, sm), (c, cm)))
            n_s, n_c = np.where(own_can, sm, score), np.where(own_can, cm, c)
            p = P[d_st, n_c, d_c]
            if is_phi.any():
                p = np.where(is_phi[d_st], np.where((d_s >= 1) & (n_s >= 1), np.maximum(d_s, 1) ** (-phi[d_st]), 0.0), p)
            act = (u_help < p).astype(int)
            dS = np.where(own_can, CN[n_c, act, d_c], NC[n_c, act, d_c])
            dS = np.where(own_can != par_can, dS, 0)
            up = u_noise < eps * eps_up
            dn = ~up & (u_noise < eps)
            dS = np.where(up, 1, np.where(dn, -1, dS))
            score = np.maximum(score + dS, 0).astype(np.int32)
            t += 1
        out[T] = [np.bincount(score[strat == k], minlength=int(score.max()) + 1) for k in range(ns)]
    return out, counts


# ------------------------------------------------------------------ engine side
def engine_w(strategies, rho, norm, tau, eps, eps_up, eps_down, w_init, t, **solver_kw):
    objs = []
    for kind, spec in strategies:
        if kind == "phi":
            objs.append(PhiDisc(spec))
        elif spec.count("|") == 1:
            objs.append(BinaryStrategy.from_name(spec))
        else:
            objs.append(TernaryStrategy.from_name(spec, tau))
    solver = StableScoreSolver(ScoringNorm.from_barcode(norm, tau=tau), Noise(eps, eps_up, eps_down), tol=1e-5,
                               max_iter=t, force_max_iter=True, **solver_kw)
    return np.asarray(solver(Population(objs, rho=list(rho)), np.tile(np.asarray(w_init, float), (len(objs), 1))).w)


# ------------------------------------------------------------------ yardsticks
_mc = np.random.default_rng(12345)


def _pad(a, L):
    out = np.zeros(L)
    out[:len(a)] = a
    return out


def deviation(counts, p, nk):
    """(worst z over bins with p > PMIN, TV, TV noise threshold, |z of the mean|, |z of the share at 0|)"""
    L = max(len(counts), len(p))
    q, p = _pad(counts, L) / nk, _pad(p, L)
    p = p / p.sum()
    se = np.sqrt(p * (1 - p) / nk)
    big = p > PMIN
    zmax = float(np.abs((q[big] - p[big]) / se[big]).max())
    tv = 0.5 * np.abs(q - p).sum()
    tvs = 0.5 * np.abs(_mc.multinomial(nk, p, size=300) / nk - p).sum(1)
    j = np.arange(L)
    mean_e = (p * j).sum()
    sd = np.sqrt((p * j ** 2).sum() - mean_e ** 2)
    z_mean = abs(((q * j).sum() - mean_e) / (sd / np.sqrt(nk)))
    z0 = abs((q[0] - p[0]) / np.sqrt(p[0] * (1 - p[0]) / nk)) if 0 < p[0] < 1 else 0.0
    return zmax, tv, tvs.mean() + 6 * tvs.std(), z_mean, z0


def check(sim_hist, counts, w, label=""):
    for k, nk in enumerate(counts):
        zmax, tv, tv_thr, z_mean, z0 = deviation(sim_hist[k], w[k], int(nk))
        msg = f"{label} strategy {k}: max z={zmax:.2f}, TV={tv:.4f} (limit {tv_thr:.4f}), z(mean)={z_mean:.2f}, z(share at 0)={z0:.2f}"
        assert zmax <= ZMAX and tv <= tv_thr and z_mean <= ZMAX and z0 <= ZMAX, msg


# ------------------------------------------------------------------ the scenarios
MONEY = "CN:00000-++ NC:00000+--"
MONEY_T5 = "CN:0000000--+++0--+++ NC:0000000++---0++---"      # the money norm lifted to tau = 5
NOISE_5050 = (0.1, 0.5, 0.5)
THIRDS = [1 / 3, 1 / 3, 1 / 3]

RUN_A = dict(strategies=[("table", "0,0|0,0"), ("table", "0,0|0,g"), ("table", "0,0,0|0,g,0|0,g,0")], rho=THIRDS,
             norm=MONEY_T5, tau=5, eps=0.1, eps_up=0.5, eps_down=0.5, w_init=[0.5, 0.5])
RUN_B = dict(strategies=[("table", "0,0|0,0"), ("table", "0,0|0,g"), ("phi", 0.17)], rho=THIRDS,
             norm=MONEY, tau=None, eps=0.1, eps_up=0.5, eps_down=0.5, w_init=[0.5, 0.5])
PHI1 = dict(strategies=[("phi", 1.0)], rho=[1.0], norm="CN:00000-0+ NC:00000000", tau=None, eps=0.1, eps_up=0.4,
            eps_down=0.6, w_init=[1.0])
DEF_DISC = dict(strategies=[("table", "0,0|0,0"), ("table", "0,0|0,g")], rho=[0.5, 0.5], norm="CN:00000-00 NC:00000000",
                tau=None, eps=0.1, eps_up=0.4, eps_down=0.6,
                w_init=[(1 / 3) * (2 / 3) ** j for j in range(40)])      # starts at the Discriminators' stationary law
# random genuinely ternary norms / probabilistic ternary strategies (drawn once with a fixed seed;
# in each the engine's chain has a stationary distribution)
RANDOM = {
    'tau1': dict(strategies=[('table', '0.576,0.126,g|0.882,0.067,g|0.438,0,0.465'), ('table', '0.603,0.745,g|0,0.757,g|0.839,0.362,0.424')],
             rho=[0.5489, 0.4511],
             norm='CN:+0+0--+0-+-00+-++0 NC:+0++00-00+-++00---',
             tau=1,
             eps=0.2,
             eps_up=0.2,
             eps_down=0.8,
             w_init=[0.5, 0.5]),
    'tau2': dict(strategies=[('table', 'g,0.158,g|0.591,g,g|0,0,0.563'), ('table', 'g,0,0.411|0,g,0.922|0.749,0.723,0.713')],
             rho=[0.4093, 0.5907],
             norm='CN:++-+-0000+0-+0-00- NC:0+++0+-+0+-++0-+--',
             tau=2,
             eps=0.05,
             eps_up=0.2,
             eps_down=0.8,
             w_init=[1.0]),
    'tau4': dict(strategies=[('table', '0,0,0|0,g,0.547|0.488,0.64,0'), ('table', '0,0.443,0.782|0.078,0.059,g|0,0.89,0.225'), ('table', '0,0.869,0.813|0,g,0.683|g,g,0')],
             rho=[0.2311, 0.6507, 0.1182],
             norm='CN:+0++-+000+++00-+0+ NC:+0000+++-0-+0-----',
             tau=4,
             eps=0.2,
             eps_up=0.2,
             eps_down=0.8,
             w_init=[1.0]),
}
MIXED = dict(strategies=[('phi', 0.5), ('table', '0.465,g,0|0.538,0,0|0.924,0,g')],
         rho=[0.4065, 0.5935],
         norm='CN:+0++0-+000+0+--00- NC:+++00+000-000-+000',
         tau=1,
         eps=0.2,
         eps_up=0.6,
         eps_down=0.4,
         w_init=[0.5, 0.5])


def run_and_check(sc, times, seed, label):
    hist, counts = simulate(**sc, times=times, seed=seed)
    for t in times:
        check(hist[t], counts, engine_w(**sc, t=t), f"{label} t={t}")
    return hist, counts


def test_run_A_money_ternary_tau5():
    """Defectors, AllD-Disc and a ternary strategy under the money norm lifted to tau = 5 (user's Run A)."""
    run_and_check(RUN_A, [1, 5, 20, 100], seed=11, label="Run A")


def test_run_B_money_phidisc():
    """Defectors, AllD-Disc and phi-Disc(0.17) under the money norm (user's Run B)."""
    run_and_check(RUN_B, [1, 5, 20, 100], seed=12, label="Run B")


def test_phi_disc_alone_matches_notes():
    """Notes p. 8: phi = 1, eps = .1, up = .4, down = .6, everyone starts at 0: pi_0 = 0.2555, E[score] = 1.889."""
    hist, counts = run_and_check(PHI1, [1, 20, 400], seed=13, label="phi-Disc(1)")
    q = hist[400][0] / N
    se0 = np.sqrt(0.2555 * (1 - 0.2555) / N)
    assert abs(q[0] - 0.2555) <= 6 * se0 + 6e-5
    mean = (q * np.arange(len(q))).sum()
    assert abs(mean - 1.889) <= 6 * 2.6 / np.sqrt(N) + 5e-4          # sd of the score is about 2.6


def test_defectors_vs_discriminators_matches_notes():
    """Notes p. 5-6: Defectors have mean score 0.326093 and pi_0 = 0.754095, Discriminators mean 2 and pi_0 = 1/3.
    (Started from the Discriminators' stationary law: their chain needs ~1000 steps to relax from score 0.)"""
    hist, counts = run_and_check(DEF_DISC, [1, 20, 400], seed=14, label="Defectors/Discriminators")
    for k, (pi0, mean, sd) in enumerate([(0.754095, 0.326093, 0.66), (1 / 3, 2.0, 2.45)]):
        q = hist[400][k] / counts[k]
        se0 = np.sqrt(pi0 * (1 - pi0) / counts[k])
        assert abs(q[0] - pi0) <= 6 * se0
        assert abs((q * np.arange(len(q))).sum() - mean) <= 6 * sd / np.sqrt(counts[k])


@pytest.mark.parametrize("key,seed", [("tau1", 21), ("tau2", 22), ("tau4", 23)])
def test_random_ternary_norm(key, seed):
    run_and_check(RANDOM[key], [1, 20, 300], seed=seed, label=f"random ternary {key}")


def test_mixed_phi_disc_and_ternary_strategy():
    run_and_check(MIXED, [1, 20, 300], seed=31, label="phi-Disc(0.5) + ternary")


def test_the_check_has_power():
    """The same yardsticks must reject an engine run whose noise rate is 20% off (sanity check on the test)."""
    hist, counts = simulate(**RUN_A, times=[100], seed=11)
    wrong = engine_w(**{**RUN_A, "eps": 0.12}, t=100)
    with pytest.raises(AssertionError):
        check(hist[100], counts, wrong, "wrong eps")


def _transposed(name):
    rows = [r.split(",") for r in name.split("|")]
    return "|".join(",".join(rows[c][r] for c in range(len(rows))) for r in range(len(rows)))


@pytest.mark.parametrize("mistake", ["rows and columns of every strategy table swapped", "one '+' in CN dropped"])
def test_the_check_detects_ternary_mistakes(mistake):
    """Same yardsticks, ternary case: an engine that read the tables or the norm wrongly must be rejected."""
    sc = RANDOM["tau1"]
    hist, counts = simulate(**sc, times=[300], seed=21)
    if mistake.startswith("rows"):
        wrong = {**sc, "strategies": [("table", _transposed(spec)) for _, spec in sc["strategies"]]}
    else:
        cn, nc = sc["norm"].split()
        i = cn.index("+", 3)
        wrong = {**sc, "norm": cn[:i] + "0" + cn[i + 1:] + " " + nc}
    with pytest.raises(AssertionError):
        check(hist[300], counts, engine_w(**wrong, t=300), mistake)


def test_lifted_binary_norm_gives_the_binary_distribution():
    """A binary norm lifted to ternary classes gives exactly the binary result (engine only)."""
    binary = ScoringNorm.from_barcode(MONEY)
    for tau in (1, 3):
        lifted = binary.to_ternary(tau)
        res = []
        for norm, strategies in ((binary, [BinaryStrategy.from_name("0,0|0,0"), BinaryStrategy.from_name("0,0|0,g")]),
                                 (lifted, [BinaryStrategy.from_name("0,0|0,0"), BinaryStrategy.from_name("0,0|0,g")])):
            solver = StableScoreSolver(norm, Noise(*NOISE_5050), max_iter=60, force_max_iter=True)
            res.append(np.asarray(solver(Population(strategies, rho=[0.5, 0.5]), np.tile([0.5, 0.5], (2, 1))).w))
        L = min(res[0].shape[1], res[1].shape[1])
        assert np.abs(res[0][:, :L] - res[1][:, :L]).max() < 1e-12
