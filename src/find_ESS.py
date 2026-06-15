import numpy as np
import compute_payoff
import compute_stable
from helpers import parse_bb_string, parse_strategy_string
class Find_ESS:
    def __init__(
        self, 
        bb, 
        w_init, 
        rhoA, 
        maxn, 
        maxw,
        benefit = 2, 
        cost=1, 
        n_rep_states=2, 
        rich_threshold=None,
        fit_threshold=1e-5
        ):
        """
        Find all the ESS strategies out of the supplied ESSstrategiesOptionsList, 
        under some particular Big Brother score-manager.
        ESS means resistant to invasion by any of the strategies in invaderOptionsList.
        For full generality, make both these the full sets.
        Maynard-Smith and Price criteria for ESS is that, for *every* B != A,
        EITHER 
        E(A,A) >  E(B,A)  ie. a rare B would be actively die out in a popn of A. 
        OR
        E(A,A) == E(B,A)  AND   E(A,B) > E(B,B), 
        i.e. B matches A in a popn of A, but A would invade pure B from rare.
        """
        if n_rep_states not in (2, 3):
            raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")
        if n_rep_states == 2 and (rich_threshold not in (None, 0)):
            raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
        if n_rep_states == 3 and (rich_threshold is None or rich_threshold <= 0):
            raise ValueError("For n_rep_states=3, rich_threshold must be a positive integer (>0).")
        
        self.bb = bb
        self.w_init = w_init
        self.rhoA = rhoA
        self.maxn = maxn
        self.maxw = maxw
        self.n_rep_states = n_rep_states
        self.rich_threshold = rich_threshold
        self.fit_threshold = fit_threshold

        self.benefit = benefit
        self.cost = cost
        self.All_ESS = []

    def __call__(self, ESSstrategiesOptionsList, invaderOptionsList):
        self.All_ESS = []
        for stratA_name in ESSstrategiesOptionsList:
            stratA = parse_strategy_string(stratA_name, n_rep_states=self.n_rep_states)
            res_stratA = compute_stable.with_1strategy(
                self.bb,
                stratA,
                self.w_init,
                self.maxn,
                self.maxw,
                self.n_rep_states,
                self.rich_threshold,
                )
            fit_stratA = compute_payoff.one_strategy(
                stratA,
                res_stratA['w_prev_state'],
                b=self.benefit,
                c = self.cost,
                role_factor=1
                )
            
            if fit_stratA < self.fit_threshold:
                continue
            isESS = True
            for stratB_name in invaderOptionsList:
                if stratA_name == stratB_name:
                    continue
                stratB = parse_strategy_string(stratB_name, n_rep_states=self.n_rep_states)
                resA,resB = compute_stable.with_2strategies(
                    bb=self.bb,
                    stratA=stratA,
                    stratB=stratB,
                    wA_init=self.w_init,
                    wB_init=self.w_init,
                    rhoA=self.rhoA,
                    n_rep_states=self.n_rep_states,
                    rich_threshold=self.rich_threshold,
                    maxn=self.maxn,
                    maxw=self.maxw
                    )
                fitA, fitB = compute_payoff.two_strategies(
                    stratA,
                    stratB,
                    resA['w_prev_state'],
                    resB['w_prev_state'],
                    self.rhoA,
                    b=self.benefit,
                    c=self.cost
                    )
                if fitA > fitB: 
                    continue    # A is safe from rare B
                elif fitA == fitB: 
                    # now A becomes invader, and B is dominant
                    resA_,resB_ = compute_stable.with_2strategies(
                    bb=self.bb,
                    stratA=stratA,
                    stratB=stratB,
                    wA_init=self.w_init,
                    wB_init=self.w_init,
                    rhoA=1-self.rhoA,
                    n_rep_states=self.n_rep_states,
                    rich_threshold=self.rich_threshold,
                    maxn=self.maxn,
                    maxw=self.maxw
                    )
                    fitA_, fitB_ = compute_payoff.two_strategies(
                        stratA,
                        stratB,
                        resA_['w_prev_state'],
                        resB_['w_prev_state'],
                        1-self.rhoA,
                        b=self.benefit,
                        c=self.cost
                        )
                    if fitA_ > fitB_:
                        continue # Although common A tolerates some B, common B would 
                                # lose to rare A, so A's still safe: move on to next B contender.
                    elif fitA_ == fitB_:
                        continue # something about ESS set. not implemented yet
                    else:
                        isESS = False
                        break
                else:
                    isESS = False
                    break
                
            if isESS:
                # Match Julia's final "boring ESS" check. Julia repeats the
                # one-strategy calculation without passing maxn, so it uses 100.
                final_res = compute_stable.with_1strategy(
                    self.bb,
                    stratA,
                    self.w_init,
                    maxn=100,
                    maxw=max(self.maxw, 100),
                    n_rep_states=self.n_rep_states,
                    rich_threshold=self.rich_threshold,
                )
                final_fit = compute_payoff.one_strategy(
                    stratA,
                    final_res['w_prev_state'],
                    b=self.benefit,
                    c=self.cost,
                )
                if final_fit >= 1e-5:
                    self.All_ESS.append(stratA_name)
            
        return self.All_ESS


if __name__ == "__main__":
    strategy_options = ["0,0|g,g", "0,0|0,0"]
    bb_string = "CmeetsN:00+00-++ NmeetsC:00000+--"

    bigb = parse_bb_string(bb_string)
    w_x = np.array([0.1, 0.0, 0.7, 0.0, 0.2])
    rhoA = 0.2

    find_ess = Find_ESS(
        bb=bigb,
        w_init=w_x,
        rhoA=rhoA,
        maxn=100,
        maxw=100,
    )
    ess = find_ess(strategy_options, strategy_options)
    print(ess)
