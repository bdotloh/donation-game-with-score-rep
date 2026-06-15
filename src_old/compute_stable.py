import csv
from pathlib import Path
import sys
import time

import numpy as np

src_path = Path(__file__).resolve().parents[0]/ "src"

# print(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(src_path))
from onestep import OneStep
from helpers import transform_distribution_into_states,parse_bb_string,parse_strategy_string
from compute_payoff import two_strategies


def with_1strategy(
    bb,
    strat,
    w_init,
    maxn,
    maxw,
    n_rep_states,
    rich_threshold
    ):
    # Check that n_rep_states is valid
    if n_rep_states not in (2, 3):
        raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")
    if n_rep_states == 2 and (rich_threshold not in (None, 0)):
        raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
    if n_rep_states == 3 and (rich_threshold is None or rich_threshold <= 0):
        raise ValueError("For n_rep_states=3, rich_threshold must be a positive integer (>0).")
    
    onestepA = OneStep(bb,strat,strat,rhoA,1-rhoA,maxw,n_rep_states,rich_threshold)
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
    rich_threshold=None
    ):
    # Check that n_rep_states is valid
    if n_rep_states not in (2, 3):
        raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")
    if n_rep_states == 2 and (rich_threshold not in (None, 0)):
        raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
    if n_rep_states == 3 and (rich_threshold is None or rich_threshold <= 0):
        raise ValueError("For n_rep_states=3, rich_threshold must be a positive integer (>0).")

    onestepA = OneStep(bb,stratA,stratB,rhoA,1-rhoA,maxw,n_rep_states,rich_threshold)
    onestepB = OneStep(bb,stratB,stratA,1-rhoA,rhoA,maxw,n_rep_states,rich_threshold)

    wA_curr = wA_init
    wB_curr = wB_init
    wA_prev = None
    wB_prev = None

    for i in range(maxn):
        wA_prev = wA_curr
        wB_prev = wB_curr

        wA_states_prev = transform_distribution_into_states(wA_prev, rich_threshold=rich_threshold)
        wB_states_prev = transform_distribution_into_states(wB_prev, rich_threshold=rich_threshold)

        wA_curr = onestepA(wA_prev, wA_states_prev, wB_states_prev)
        wB_curr = onestepB(wB_prev, wB_states_prev, wA_states_prev)

    wA_states_final=transform_distribution_into_states(wA_curr,rich_threshold=rich_threshold)
    wB_states_final=transform_distribution_into_states(wB_curr,rich_threshold=rich_threshold)
    payoffA,payoffB=two_strategies(stratA,stratB,wA_states_final,wB_states_final,rhoA)
    
    resultsA = dict()
    resultsA['w'] = wA_curr
    resultsA['w_state'] = wA_states_final
    resultsA['fit'] = payoffA
    resultsA['trans']= onestepA.rates
    resultsB = dict()
    resultsB['w'] = wB_curr
    resultsB['w_state'] = wB_states_final
    resultsB['fit'] = payoffB
    resultsB['trans']= onestepB.rates

    return resultsA, resultsB

if __name__ == '__main__':
    w_x = [0.1,0.7,0.2]
    w_y= [0.2,0.7,0.1]
    rho_x = 0.2
    rho_y = 1 - rho_x
    bb_string = "CmeetsN:00+00-++ NmeetsC:00-00+--"
    strategy_x_string = "0,0|g,g"
    strategy_y_string = "0,0|0,0"

    bb = parse_bb_string(bb_string)
    strategy_x = parse_strategy_string(strategy_x_string)
    strategy_y = parse_strategy_string(strategy_y_string)
    resA,resB = with_2strategies(bb,strategy_x,strategy_y,w_x,w_y,rho_x,maxn=100)

    print("Results for A:")
    print(f"  w_state: {resA['w_state']}")      # state_0 = 0.09318666725505377
    print(f"  trans:   {resA['trans']}")        # [0.0458596,0.0463559, 0.05]
    print(f"  fit:     {resA['fit']}")      # 0.09318666725505377

    print("\nResults for B:")
    print(f"  w_state: {resB['w_state']}")      # state_0 = 0.997659229264295
    print(f"  trans:   {resB['trans']}")        # [0.0,0.000496332,0.0958596]
    print(f"  fit:     {resB['fit']}")      # 0.00023407707


