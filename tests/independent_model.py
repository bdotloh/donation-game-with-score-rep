"""Independent oracle for the donation-game score chain, written from the notes (pp.1-8) and
tests/reference/helper_funcs.jl, NOT from the engine.  Works on raw integer scores throughout.
Conventions: help(sn, sc) = Pr(Can with score sc helps Need with score sn); norm = (CN, NC) with
CN(sn, act, sc) / NC(sn, act, sc) in {-1,0,+1} = score change of the Can / of the Need (act 1 = helped)."""
import numpy as np

def cls(s, tau=None):
    """score class: binary (tau None) 0 | >=1 ; ternary 0 | 1..tau | >tau"""
    return 0 if s == 0 else (1 if tau is None or s <= tau else 2)

def table_help(name, tau=None):
    """'a,b|c,d' or 'a,b,c|d,e,f|g,h,i' (rows = Need class, cols = Can class; 0, g or a number)"""
    val = lambda x: {'0': 0.0, 'g': 1.0}[x] if x in ('0', 'g') else float(x)
    P = [[val(x) for x in row.split(',')] for row in name.split('|')]
    return lambda sn, sc: P[cls(sn, tau)][cls(sc, tau)]

def phi_disc(phi):
    """score-0 Can never helps, never helps a score-0 Need, else helps with prob sc**-phi"""
    return lambda sn, sc: 0.0 if sn == 0 or sc == 0 else float(sc) ** (-phi)

def parse_norm(code, tau=None):
    """'CN:xxxxxxxx NC:xxxxxxxx' (8 chars, tau=None) or 18 chars each; flat index [need class][act][can class]"""
    parts = dict(p.split(':') for p in code.split())
    sym, C = {'-': -1, '0': 0, '+': 1}, (2 if tau is None else 3)
    tabs = [[sym[c] for c in parts[key]] for key in ('CN', 'NC')]
    assert all(len(t) == 2 * C * C for t in tabs), 'barcode length does not match the number of classes'
    return tuple((lambda t: lambda sn, act, sc: t[(cls(sn, tau) * 2 + act) * C + cls(sc, tau)])(t) for t in tabs)

def lift_code(code):
    """binary barcode -> ternary barcode whose class-1 and class-2 entries both copy binary class 1"""
    out = []
    for p in code.split():
        key, s = p.split(':')
        out.append(key + ':' + ''.join(s[(min(a, 1) * 2 + act) * 2 + min(c, 1)]
                                       for a in range(3) for act in range(2) for c in range(3)))
    return ' '.join(out)

def pair(strats, norm, i, j, k, l):
    """(Pr[dS=+1], Pr[dS=-1]) of focal (strategy i, score j) meeting (k, l): enumerate roles and help/decline"""
    CN, NC = norm
    up = dn = 0.0
    for h, d_of in ((strats[i](l, j), lambda a: CN(l, a, j)),     # focal = Can, other = Need  (prob 1/4)
                    (strats[k](j, l), lambda a: NC(j, a, l))):    # focal = Need, other = Can  (prob 1/4)
        for act, pr in ((1, h), (0, 1.0 - h)):
            d = d_of(act)
            up += 0.25 * pr * (d == 1)
            dn += 0.25 * pr * (d == -1)
    return up, dn

def kernel(strats, norm, L):
    """K[0/1][i, k, j, l] = pair() up/down probabilities for scores j, l < L (w-independent)"""
    n = len(strats)
    K = np.zeros((2, n, n, L, L))
    for i in range(n):
        for k in range(n):
            for j in range(L):
                for l in range(L):
                    K[:, i, k, j, l] = pair(strats, norm, i, j, k, l)
    return K

def kernel_fast(strats, norm, L):
    """Same array as kernel(), built from L x L tables of help / norm entries (used for large L; checked equal to kernel())"""
    CN, NC = norm
    idx = np.arange(L)
    H = [np.array([[s(sn, sc) for sc in idx] for sn in idx]) for s in strats]                 # H[i][need, can]
    cn = [np.array([[CN(sn, a, sc) for sc in idx] for sn in idx]) for a in (0, 1)]            # cn[act][need, can]
    nc = [np.array([[NC(sn, a, sc) for sc in idx] for sn in idx]) for a in (0, 1)]
    n = len(strats)
    K = np.zeros((2, n, n, L, L))
    for t, sign in enumerate((1, -1)):
        for i in range(n):
            donor = 0.25 * (H[i] * (cn[1] == sign) + (1 - H[i]) * (cn[0] == sign))       # focal = Can: [need l, can j]
            for k in range(n):
                recv = 0.25 * (H[k] * (nc[1] == sign) + (1 - H[k]) * (nc[0] == sign))    # focal = Need: [need j, can l]
                K[t, i, k] = donor.T + recv                                                # index [j, l]
    return K

def rates(K, noise, rho, w):
    """G, F, up, down, all (n, L): G[i,j] = Pr(dS=+1 | i,j) (Eq 1), F (Eq 2); up[i,j] = rate j -> j+1,
    down[i,j] = rate j -> j-1 (Eq 3-4; nothing goes down from 0)."""
    eps, eu, ed = noise
    n, L = w.shape
    G = np.einsum('ikjl,k,kl->ij', K[0][:, :, :L, :L], np.asarray(rho, float), w)
    F = np.einsum('ikjl,k,kl->ij', K[1][:, :, :L, :L], np.asarray(rho, float), w)
    up, down = (1 - eps) * G + eps * eu, (1 - eps) * F + eps * ed
    down[:, 0] = 0.0
    return G, F, up, down

def evolve(strats, norm, noise, rho, w0, steps, fast=False):
    """`steps` synchronous updates w <- w + dw (rates from the current w); support grows by one score per step.
    Returns (w, w_prev, rates_used_in_last_step)."""
    w = np.array(w0, float)
    K = (kernel_fast if fast else kernel)(strats, norm, w.shape[1] + steps)
    w_prev = last = None
    for _ in range(steps):
        n, L = w.shape
        G, F, up, down = rates(K, noise, rho, w)
        dw = np.zeros((n, L + 1))
        dw[:, :L] -= w * (up + down)
        dw[:, 1:] += w * up
        dw[:, :L - 1] += (w * down)[:, 1:]
        w_prev, last = w, (G, F, up, down)
        w = np.hstack([w, np.zeros((n, 1))]) + dw
    return w, w_prev, last

def detailed_balance(up, down):
    """Eq 5-6 on states 0..S (up[S] unused): pi_j = pi_0 prod_{m<=j} up[m-1]/down[m]; None if down[m]=0<up[m-1]"""
    up, down = np.asarray(up, float), np.asarray(down, float)
    if np.any((down[1:] == 0) & (up[:-1] > 0)):
        return None
    with np.errstate(divide='ignore'):
        logr = np.where(up[:-1] > 0, np.log(up[:-1]) - np.log(np.where(down[1:] > 0, down[1:], 1.0)), -np.inf)
    lp = np.concatenate([[0.0], np.cumsum(logr)])
    lp = np.where(np.isnan(lp), -np.inf, lp)
    p = np.exp(lp - lp.max())
    return p / p.sum()

def stationary(strats, norm, noise, rho, S=60, w_start=None, tol=1e-14, maxit=20000, damp=0.5):
    """Fixed point w = pi(rates(w)) on states 0..S (S+1 states), damped Picard iteration.  Returns (w (n,S+1), converged, n_iter).
    Check the mass of the top few states: if it is not ~0 the truncation S is too small or there is no stable distribution.
    Picard can fail (stay on a 2-cycle) for near-critical cases (up/down ratio of the top segment ~ 0.95-0.99)."""
    n = len(strats)
    K = kernel_fast(strats, norm, S + 1)
    w = np.array(w_start, float) if w_start is not None else np.tile(0.5 ** np.arange(S + 1) / 2, (n, 1))
    if w.shape[1] < S + 1:
        w = np.hstack([w, np.zeros((n, S + 1 - w.shape[1]))])
    for it in range(1, maxit + 1):
        _, _, up, down = rates(K, noise, rho, w)
        pis = [detailed_balance(up[i], down[i]) for i in range(n)]
        if any(p is None for p in pis):
            return w, False, it
        pi = np.array(pis)
        err = np.abs(pi - w).max()
        w = (1 - damp) * w + damp * pi
        if err < tol:
            return w, True, it
    return w, False, maxit
