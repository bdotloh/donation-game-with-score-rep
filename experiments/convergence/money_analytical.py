import numpy as np
from scipy.optimize import fsolve


class CoupledChains:
    """Stationary distributions of two coupled birth-death chains, A and B.

    Chain A has up rate g and down rate f in every state.
    Chain B has up rate g from states 0, ..., tau-1, up rate eps * eps_up
    from states tau onwards, and down rate f.

    The rates are

        G = lam * [(1 - w_0^A)(1 - rho) + (1 - w_0^B) rho]
        F = lam * [1 - (1 - sum_{j=0}^{tau-1} w_j^B) rho]
        g = (1 - eps) G + eps * eps_up
        f = (1 - eps) F + eps * eps_down

    and g = g^A = g^B, f = f^A = f^B.

    With r = g / f and s = eps * eps_up / f, the stationary distributions are

        w_j^A = (1 - r) r^j
        w_j^B = w_0^B r^j                      for 0 <= j <= tau
        w_j^B = w_0^B r^tau s^(j - tau)        for j >= tau
        w_0^B = [(1 - r^tau) / (1 - r) + r^tau / (1 - s)]^(-1)

    The unknowns (w_0^A, w_0^B, f) are found by solving the three
    self-consistency equations (1)-(3) with scipy's fsolve.

    Conditions for stationary distributions (checked in __init__):
        1. eps > 0
        2. eps_down > eps_up
        3. rho (1 - eps) lam / (1 + tau (1 - eps eps_up / g))
               < (1 - eps_up / eps_down) [(1 - eps) lam + eps eps_down],
           with g = (1 - eps) lam eps_up / eps_down + eps eps_up.
    Condition 3 is equivalent to rho < rho_c (attribute rho_c). It is
    proved sufficient; its necessity was checked numerically only.

    When the conditions hold,
        (1 - rho) w_0^A + rho w_0^B = 1 - eps_up / eps_down.

    Parameters
    ----------
    lam : float
        lambda.
    eps : float
        epsilon.
    eps_up : float
        epsilon_up.
    eps_down : float
        epsilon_down.
    rho : float
        Fraction of type B, in [0, 1].
    tau : int
        Threshold state, tau >= 0. B's up rate is g from states
        0, ..., tau-1 and eps * eps_up from states tau onwards.

    Attributes
    ----------
    rho_c : float
        Largest rho for which condition 3 holds (set in __init__).
    w0A, w0B : float
        Stationary probability of state 0 for A and B (set by solve).
    g, f : float
        Up and down rates shared by A and B (set by solve).
    r : float
        Ratio g / f (set by solve).
    s : float
        Ratio eps * eps_up / f (set by solve).

    Example
    -------
    >>> c = CoupledChains(lam=.25, eps=.1, eps_up=.49, eps_down=.51,
    ...                   rho=.01, tau=20).solve()
    >>> round(c.w0A, 5), round(c.w0B, 5)
    (0.03892, 0.06836)
    """

    def __init__(self, lam, eps, eps_up, eps_down, rho, tau):
        assert eps > 0, f"need eps > 0, got eps={eps}"
        assert eps_down > eps_up, (
            f"need eps_down > eps_up for a stationary distribution, "
            f"got eps_up={eps_up}, eps_down={eps_down}")
        assert int(tau) >= 0, f"need tau >= 0, got tau={tau}"

        self.lam = lam
        self.eps = eps
        self.eps_up = eps_up
        self.eps_down = eps_down
        self.rho = rho
        self.tau = int(tau)
        self._solved = False

        # condition 3
        g = (1 - eps) * lam * eps_up / eps_down + eps * eps_up
        f_max = (1 - eps) * lam + eps * eps_down
        b = 1 + self.tau * (1 - eps * eps_up / g)
        self.rho_c = (1 - eps_up / eps_down) * f_max * b / ((1 - eps) * lam)
        assert rho < self.rho_c, (
            f"need rho < rho_c = {self.rho_c:.6g} for a stationary "
            f"distribution, got rho={rho}")

    def _residuals(self, x):
        """Residuals of equations (1)-(3) at x = (w0A, w0B, f)."""
        w0A, w0B, f = x
        lam, eps, rho, tau = self.lam, self.eps, self.rho, self.tau

        G = lam * ((1 - w0A) * (1 - rho) + (1 - w0B) * rho)
        g = (1 - eps) * G + eps * self.eps_up
        r = g / f
        s = eps * self.eps_up / f

        # 1 - sum_{j=0}^{tau-1} w_j^B
        tail_B = w0B * r ** tau / (1 - s)
        F = lam * (1 - tail_B * rho)

        return [
            w0A - (1 - r),
            w0B - 1 / ((1 - r ** tau) / (1 - r) + r ** tau / (1 - s)),
            f - ((1 - eps) * F + eps * self.eps_down),
        ]

    def solve(self, x0=(None, None, None)):
        """Solve for (w0A, w0B, f). Returns self.

        Parameters
        ----------
        x0 : tuple
            Initial guess (w0A, w0B, f). If w0A or w0B is None it is set
            to 1 - eps_up / eps_down. If f is None it is set to
            (1 - eps) * lam + eps * eps_down.

        Raises
        ------
        RuntimeError
            If fsolve does not converge, or if either ratio r, s is not
            in (0, 1).
        """
        w0A, w0B, f = x0
        q = 1 - self.eps_up / self.eps_down
        if w0A is None:
            w0A = q
        if w0B is None:
            w0B = q
        if f is None:
            f = (1 - self.eps) * self.lam + self.eps * self.eps_down

        sol, info, ier, msg = fsolve(self._residuals, [w0A, w0B, f],
                                     full_output=True)
        if ier != 1:
            raise RuntimeError(f"fsolve did not converge: {msg}")

        self.w0A, self.w0B, self.f = (float(v) for v in sol)
        G = self.lam * ((1 - self.w0A) * (1 - self.rho) + (1 - self.w0B) * self.rho)
        self.g = (1 - self.eps) * G + self.eps * self.eps_up
        self.r = self.g / self.f
        self.s = self.eps * self.eps_up / self.f

        if not (0 < self.r < 1 and 0 < self.s < 1):
            raise RuntimeError(f"No stationary distribution: r={self.r}, s={self.s}")

        self._solved = True
        return self

    def _check(self):
        if not self._solved:
            raise RuntimeError("Call solve() first.")

    def _w0B_norm(self):
        """w_0^B computed from r and s, so that wB sums exactly to 1."""
        k = self.tau
        return 1 / ((1 - self.r ** k) / (1 - self.r) + self.r ** k / (1 - self.s))

    def nA(self):
        """Smallest n such that the mass of A on states j >= n is below
        machine epsilon. That mass is r^n."""
        self._check()
        eps_mach = np.finfo(float).eps
        return int(np.ceil(np.log(eps_mach) / np.log(self.r)))

    def nB(self):
        """Smallest n such that the mass of B on states j >= n is below
        machine epsilon. For n >= tau that mass is
        w0B * r^tau * s^(n-tau) / (1 - s)."""
        self._check()
        eps_mach = np.finfo(float).eps
        k = self.tau
        w0B = self._w0B_norm()
        tail_k = w0B * self.r ** k / (1 - self.s)  # mass on j >= tau
        if tail_k < eps_mach:
            # cut inside 0..tau: mass on j >= n is
            # w0B * (r^n - r^tau) / (1 - r) + tail_k
            n = 0
            while w0B * (self.r ** n - self.r ** k) / (1 - self.r) + tail_k >= eps_mach:
                n += 1
            return n
        return k + int(np.ceil(np.log(eps_mach / tail_k) / np.log(self.s)))

    def wA(self, n=None):
        """Return w_j^A for j = 0, ..., n-1 as a numpy array.

        If n is None, n = nA(), so the returned mass sums to 1 within
        machine epsilon. The terms r^j are divided by their sum plus the
        analytic mass beyond n, so the distribution is normalised for the
        solved r.
        """
        self._check()
        if n is None:
            n = self.nA()
        j = np.arange(n)
        w = self.r ** j
        # divide by the total mass sum_{j>=0} r^j = 1 / (1 - r), computed as
        # the mass on 0..n-1 plus the analytic mass r^n / (1 - r) beyond
        return w / (w.sum() + self.r ** n / (1 - self.r))

    def wB(self, n=None):
        """Return w_j^B for j = 0, ..., n-1 as a numpy array.

        If n is None, n = nB(), so the returned mass sums to 1 within
        machine epsilon. The unnormalised terms are divided by their sum
        plus the analytic mass beyond n, so the distribution is normalised
        for the solved r and s.
        """
        self._check()
        if n is None:
            n = self.nB()
        j = np.arange(n)
        k = self.tau
        w = np.where(
            j <= k,
            self.r ** np.minimum(j, k),
            self.r ** k * self.s ** np.maximum(j - k, 0),
        )
        # analytic mass beyond n (same unnormalised scale, w_0 = 1)
        if n >= k:
            beyond = self.r ** k * self.s ** (n - k) / (1 - self.s)
        else:
            beyond = (self.r ** n - self.r ** k) / (1 - self.r) + self.r ** k / (1 - self.s)
        return w / (w.sum() + beyond)

    def __repr__(self):
        if not self._solved:
            return f"CoupledChains(unsolved, rho_c={self.rho_c:.6g})"
        return (f"CoupledChains(w0A={self.w0A:.6g}, w0B={self.w0B:.6g}, "
                f"g={self.g:.6g}, f={self.f:.6g}, r={self.r:.6g}, s={self.s:.6g}, "
                f"rho_c={self.rho_c:.6g})")


if __name__ == "__main__":
    c = CoupledChains(lam=.25, eps=.1, eps_up=.49, eps_down=.51,
                      rho=.01, tau=20).solve()
    print(c)
    eps_mach = np.finfo(float).eps
    for name, w in [("A", c.wA()), ("B", c.wB())]:
        err = abs(np.sum(w) - 1)
        print(f"{name}: {len(w)} states, |sum - 1| = {err:.3g} "
              f"({err / eps_mach:.2f} machine epsilon)")