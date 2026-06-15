import csv
from pathlib import Path
import sys
import time

import numpy as np

src_path = Path(__file__).resolve().parents[0]/ "src"

# print(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(src_path))
from onestep import OneStep, OneStepSelf
from helpers import transform_distribution_into_states
import compute_payoff 



def _max_abs_change(w_curr, w_prev):
    if len(w_curr) != len(w_prev):
        w_prev = np.pad(w_prev, (0, len(w_curr) - len(w_prev)), constant_values=0.0)
    return np.max(np.abs(w_curr - w_prev))


def with_1strategy(
    bb,
    strat,
    w_init,
    maxn,
    maxw,
    n_rep_states=2,
    rich_threshold=None,
    tolerance=1e-5,
    force_until_maxn=False,
    return_prev_w=False,
    save_results_all_iterations=False,
    stop_when_transient=True,
    ):
    # Check that n_rep_states is valid
    if n_rep_states not in (2, 3):
        raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")
    if n_rep_states == 2 and (rich_threshold not in (None, 0)):
        raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
    if n_rep_states == 3 and (rich_threshold is None or rich_threshold <= 0):
        raise ValueError("For n_rep_states=3, rich_threshold must be a positive integer (>0).")
    
    onestep = OneStepSelf(
        bb,
        strat,
        maxw,
        n_rep_states,
        rich_threshold,
        save_results_all_iterations=save_results_all_iterations,
    )
    w_curr = w_init
    w_prev = w_curr
    len_w_init = len(w_init)
    for i in range(maxn):
        w_prev = w_curr
        w_curr = onestep(w_prev)
        chg = _max_abs_change(w_curr, w_prev)

        # All initial low-score bins are empty: the chain is transient.
        if stop_when_transient and np.sum(w_curr[:len_w_init]) == 0.0:
            break
        if chg <= tolerance and not force_until_maxn:
            break
    w_prev_states = transform_distribution_into_states(w_prev,rich_threshold=rich_threshold)
    results = dict()
    results['w'] = w_curr
    results['w_prev_state'] = w_prev_states
    results['trans'] = onestep.rates
    if save_results_all_iterations:
        results['trans_history'] = onestep.rates_history
        results['score_history'] = onestep.score_history
        results['rep_history'] = onestep.rep_history
    return results
  
def with_2strategies(
    bb,
    stratA,
    stratB,
    wA_init,
    wB_init,
    rhoA,
    maxn,
    maxw=100,
    n_rep_states=2,
    rich_threshold=None,
    tolerance=1e-5,
    force_until_maxn=False,
    save_results_all_iterations=False,
    stop_when_transient=True,
    ):
    # Check that n_rep_states is valid
    if n_rep_states not in (2, 3):
        raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")
    if n_rep_states == 2 and (rich_threshold not in (None, 0)):
        raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
    if n_rep_states == 3 and (rich_threshold is None or rich_threshold <= 0):
        raise ValueError("For n_rep_states=3, rich_threshold must be a positive integer (>0).")

    onestepA = OneStep(
        bb,
        stratA,
        stratB,
        rhoA,
        maxw,
        n_rep_states,
        rich_threshold,
        save_results_all_iterations=save_results_all_iterations,
    )
    onestepB = OneStep(
        bb,
        stratB,
        stratA,
        1-rhoA,
        maxw,
        n_rep_states,
        rich_threshold,
        save_results_all_iterations=save_results_all_iterations,
    )

    wA_curr = wA_init
    wB_curr = wB_init
    len_wA_init = len(wA_init)
    len_wB_init = len(wB_init)
    wA_prev = None
    wB_prev = None

    for i in range(maxn):
        wA_prev = wA_curr
        wB_prev = wB_curr

        wA_curr = onestepA(wA_prev, wB_prev)
        wB_curr = onestepB(wB_prev, wA_prev)
        max_chg = max(
            _max_abs_change(wA_curr, wA_prev),
            _max_abs_change(wB_curr, wB_prev),
        )

        # Both strategies have emptied their initial low-score bins.
        if stop_when_transient and (
            np.sum(wA_curr[:len_wA_init]) == 0.0
            and np.sum(wB_curr[:len_wB_init]) == 0.0
        ):
            break
        if max_chg <= tolerance and not force_until_maxn:
            break

    wA_states_final=transform_distribution_into_states(wA_prev,rich_threshold=rich_threshold)
    wB_states_final=transform_distribution_into_states(wB_prev,rich_threshold=rich_threshold)
    
    resultsA = dict()
    resultsA['w'] = wA_curr
    resultsA['w_prev_state'] = wA_states_final
    resultsA['trans']= onestepA.rates
    resultsB = dict()
    resultsB['w'] = wB_curr
    resultsB['w_prev_state'] = wB_states_final
    resultsB['trans']= onestepB.rates
    if save_results_all_iterations:
        resultsA['trans_history'] = onestepA.rates_history
        resultsA['score_history'] = onestepA.score_history
        resultsA['rep_history'] = onestepA.rep_history
        resultsB['trans_history'] = onestepB.rates_history
        resultsB['score_history'] = onestepB.score_history
        resultsB['rep_history'] = onestepB.rep_history

    return resultsA, resultsB

if __name__ == '__main__':
    from helpers import parse_strategy_string, parse_bb_string
    w_x = [0.1,0.7,0.2]
    w_y= [0.2,0.7,0.1]
    rho_x = 0.2
    rho_y = 1 - rho_x
    bb_string = "CmeetsN:00+00-++ NmeetsC:00000+--"
    strategy_x_string = "0,0|g,g"
    strategy_y_string = "0,0|0,0"

    bb = parse_bb_string(bb_string)
    strategy_x = parse_strategy_string(strategy_x_string)
    strategy_y = parse_strategy_string(strategy_y_string)
    resA,resB = with_2strategies(bb,strategy_x,strategy_y,w_x,w_y,rho_x,maxn=100)

    print("Results for A:")
    print(f"  w_state: {resA['w_prev_state']}")      # state_0 = 0.09318666725505377
    print(f"  trans:   {resA['trans']}")        # [0.0458596,0.0463559, 0.05]
    # print(f"  fit:     {resA['fit']}")      # 0.09318666725505377

    print("\nResults for B:")
    print(f"  w_state: {resB['w_prev_state']}")      # state_0 = 0.997659229264295
    print(f"  trans:   {resB['trans']}")        # [0.0,0.000496332,0.0958596]
    # print(f"  fit:     {resB['fit']}")      # 0.00023407707
