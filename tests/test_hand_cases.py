"""Hand-computed tiny cases. Expected numbers are written as literal arithmetic from the notes
(role factor 1/4 each way; G_j = rate j-1 -> j; F_j = rate j -> j-1). Only the public engine API is used."""
import numpy as np
from engine import *

def rates(norm, strats, rho, w, eps=0.0, up=.5, down=.5):
    tr = TransitionRates(norm, Noise(eps, up, down))
    return tr(Population(strats, rho=rho), np.array(w, float))

def code(n, entries):  # entries: {(need,act,can): '+'/'-'}; 18-char string
    s = ['0'] * 18
    for (a, b, c), ch in entries.items(): s[a*6 + b*3 + c] = ch
    return ''.join(s)

def norm3(cn, nc, tau): return ScoringNorm.from_barcode(f"CN:{code(3,cn)} NC:{code(3,nc)}", tau)


def test_hand_computed_cases():
    results = []
    def check(name, got, exp):
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(exp, float), rtol=0, atol=1e-12, err_msg=name)
        results.append(name)
    D3 = lambda tau: TernaryStrategy.from_name("0,0,0|0,0,0|0,0,0", tau)
    w4 = [[.1, .2, .3, .4]]
    # A: tau=1. Defector Can with score 1 (class1) declines to Need class 2 (mass .7) -> loses. F_1=.25*.7
    r = rates(norm3({(2,0,1):'-'}, {}, 1), [D3(1)], [1.], w4); check("A CN idx13 tau=1", r.F[0,1:4], [.175, 0, 0])
    # A': NC: Need class 2 (score>=2) loses when Can class 1 (mass .2) declines. F_2=F_3=.25*.2
    r = rates(norm3({}, {(2,0,1):'-'}, 1), [D3(1)], [1.], w4); check("A' NC idx13 tau=1", r.F[0,1:4], [0, .05, .05])
    # B: tau=1 strategy helps need class1 (score 1, mass .2) when can class 2; CN[1][1][2]='+': can score>=2 gains
    st = TernaryStrategy.from_name("0,0,0|0,0,g|0,0,0", 1)
    r = rates(norm3({(1,1,2):'+'}, {}, 1), [st], [1.], w4); check("B tau=1", r.G[0,1:5], [0, 0, .05, .05])
    st2 = TernaryStrategy.from_name("0,0,0|0,0,g|0,0,0", 2)   # tau=2: need class1 = scores1,2 (mass .5), can class2 = score>=3 (j-1=3 -> G_4)
    r = rates(norm3({(1,1,2):'+'}, {}, 2), [st2], [1.], w4); check("B tau=2", r.G[0,1:5], [0, 0, 0, .125])
    # D: tau=2 uniform on 0..4; helps iff need class1 & can class1; CN[1][1][1] '+'
    wu = [[.2]*5]
    st = TernaryStrategy.from_name("0,0,0|0,g,0|0,0,0", 2)
    r = rates(norm3({(1,1,1):'+'}, {}, 2), [st], [1.], wu); check("D CN tau=2", r.G[0,1:6], [0, .1, .1, 0, 0])
    r = rates(norm3({}, {(1,1,1):'+'}, 2), [st], [1.], wu); check("D NC tau=2", r.G[0,1:6], [0, .1, .1, 0, 0])
    # E: help need class2 from can class1 ("0,0,0|0,0,0|0,g,0"): CN[2][1][1]='+' -> G_2,G_3 =.25*.4 ; NC[2][1][1]='+' -> G_4,G_5
    st = TernaryStrategy.from_name("0,0,0|0,0,0|0,g,0", 2)
    r = rates(norm3({(2,1,1):'+'}, {}, 2), [st], [1.], wu); check("E CN tau=2", r.G[0,1:6], [0, .1, .1, 0, 0])
    r = rates(norm3({}, {(2,1,1):'+'}, 2), [st], [1.], wu); check("E NC tau=2", r.G[0,1:6], [0, 0, 0, .1, .1])
    # E2: tau=1 same on w4: need class2 mass .7, can class1 = score1 (mass .2): CN: G_2 = .25*.7 ; NC: need class2 = scores>=2: G_3,G_4 = .25*.2
    st = TernaryStrategy.from_name("0,0,0|0,0,0|0,g,0", 1)
    r = rates(norm3({(2,1,1):'+'}, {}, 1), [st], [1.], w4); check("E2 CN tau=1", r.G[0,1:5], [0, .175, 0, 0])
    r = rates(norm3({}, {(2,1,1):'+'}, 1), [st], [1.], w4); check("E2 NC tau=1", r.G[0,1:5], [0, 0, .05, .05])
    # noise arithmetic: g = (1-eps)G + eps*up ; f = (1-eps)F + eps*down, nothing down from 0
    r = rates(norm3({(2,0,1):'-'}, {}, 1), [D3(1)], [1.], w4, eps=.2, up=.3, down=.7)
    check("noise f", r.f[0,:4], [0, .8*.175+.14, .14, .14]); check("noise g", r.g[0,1:4], [.06]*3)
    # C: phi-Disc, two strategies. phi-Disc rho .5 w0=.2 ; Defectors rho .5 w0=.6; need pos mass = .5*.8+.5*.4=.6
    cnphi = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
    w = np.zeros((2, 5)); w[0] = [.2, .1, .2, .3, .2]; w[1] = [.6, .4, 0, 0, 0]
    r = rates(cnphi, [PhiDisc(2), Defectors()], [.5, .5], w)
    j = np.arange(2, 6); check("C phi G", r.G[0,2:6], .25*.6/(j-1)**2.); check("C phi G1", r.G[0,1], 0.)
    j = np.arange(1, 5); check("C phi F", r.F[0,1:5], .25*.6*(1-j**-2.))
    check("C Defector F", r.F[1,1:5], [.15]*4); check("C Defector G", r.G[1,1:6], [0]*5)
    # C2: phi-Disc as Need never gains from help by itself; Need score irrelevant beyond positivity:
    w = np.zeros((1, 6)); w[0] = [.1, 0, 0, 0, 0, .9]
    r = rates(cnphi, [PhiDisc(1)], [1.], w); check("C2 need at score 5 only", r.G[0,1:7], [0] + [.25*.9/(j-1) for j in range(2,7)])
    assert len(results) == 18
