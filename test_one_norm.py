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
import compute_stable  

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

def calc_stable_w_2strategies(bb,stratA,stratB,wA_init,wB_init,rhoA,maxn):
    onestepA = OneStep(bb,stratA,stratB,rhoA,1-rhoA)
    onestepB = OneStep(bb,stratB,stratA,1-rhoA,rhoA)

    wA_curr = wA_init
    wB_curr = wB_init
    wA_prev = None
    wB_prev = None

    for i in range(maxn):
        if wA_curr[-1] != 0.0:
            wA_curr = np.pad(wA_curr, (0, 1), constant_values=0.0)
        if wB_curr[-1] != 0.0:
            wB_curr = np.pad(wB_curr, (0, 1), constant_values=0.0)

        wA_prev = wA_curr
        wB_prev = wB_curr

        wA_states_prev = transform_distribution_into_states(wA_prev, k=0)
        wB_states_prev = transform_distribution_into_states(wB_prev, k=0)

        wA_curr = onestepA(wA_prev, wA_states_prev, wB_states_prev)
        wB_curr = onestepB(wB_prev, wB_states_prev, wA_states_prev)

    wA_states_final=transform_distribution_into_states(wA_curr,k=0)
    wB_states_final=transform_distribution_into_states(wB_curr,k=0)
    payoffA,payoffB=two_strategies(stratA,stratB,wA_states_final,wB_states_final,rho_x)
    
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

resA,resB = calc_stable_w_2strategies(bb,strategy_x,strategy_y,w_x,w_y,rho_x,maxn=100)

print("Results for A:")
print(f"  w_state: {resA['w_state']}")      # state_0 = 0.09318666725505377
print(f"  trans:   {resA['trans']}")        # [0.0458596,0.0463559, 0.05]
print(f"  fit:     {resA['fit']:.4f}")      # 0.09318666725505377

print("\nResults for B:")
print(f"  w_state: {resB['w_state']}")      # state_0 = 0.997659229264295
print(f"  trans:   {resB['trans']}")        # [0.0,0.000496332,0.0958596]
print(f"  fit:     {resB['fit']:.4f}")      # 0.00023407707


