from compute_score_transitions import ComputeTransitionRates
import makeQ
import numpy as np
from helpers import transform_distribution_into_states, Score2StateDistribution


class OneStepSelf:
    def __init__(
        self, 
        update_rule, 
        strat_focal, 
        maxw,
        n_rep_states=2,
        rich_threshold=None,
        save_results_all_iterations=False,
        ):
   
        self.update_rule = update_rule
        self.strat_focal = strat_focal
        self.n_rep_states = n_rep_states
        self.rich_threshold = rich_threshold
        self.maxw = maxw
        self.save_results_all_iterations = save_results_all_iterations
        self.rates_history = []
        self.score_history = []
        self.rep_history = []
        self.compute_rates = ComputeTransitionRates(self.update_rule,self.strat_focal,n_rep_states=n_rep_states)
    
    def __call__(self, w):
        n_w = len(w)
        if w[-1] != 0.0 and n_w - 1 < self.maxw:
            w = np.pad(w, (0, 1), constant_values=0.0)
            n_w +=1

        w_state = transform_distribution_into_states(w,self.rich_threshold)
        self.rates = self.compute_rates(w_state)
        
        if self.n_rep_states == 2:
            Q = makeQ.binary_rep_state(*self.rates,n_w)
        elif self.n_rep_states == 3:
            Q = makeQ.three_rep_state(*self.rates,n_w,rich_threshold=self.rich_threshold)
        w_focal_new = (np.eye(n_w) + Q).T @ w
        if self.save_results_all_iterations:
            self.rates_history.append(np.array(self.rates, copy=True))
            self.score_history.append(np.array(w_focal_new, copy=True))
            self.rep_history.append(
                transform_distribution_into_states(w_focal_new, self.rich_threshold)
            )
        return w_focal_new

class OneStep:
    def __init__(
        self, 
        update_rule, 
        strat_focal, 
        strat_opponent, 
        rho_focal, 
        maxw,
        n_rep_states=2,
        rich_threshold=None,
        save_results_all_iterations=False,
        ):

        self.update_rule = update_rule
        self.strat_focal = strat_focal
        self.strat_opponent = strat_opponent
        self.rho_focal = rho_focal
        self.n_rep_states = n_rep_states
        self.rich_threshold = rich_threshold
        self.maxw = maxw
        self.save_results_all_iterations = save_results_all_iterations
        self.rates_history = []
        self.score_history = []
        self.rep_history = []

        self.transform_score2state_dist = Score2StateDistribution(self.rich_threshold)
        self.compute_rates = ComputeTransitionRates(
            update_rule=self.update_rule,
            strat_focal=self.strat_focal,
            strat_other=self.strat_opponent,
            rho_focal=self.rho_focal,
            n_rep_states=self.n_rep_states
            )
    
    def __call__(
        self, 
        w_focal,
        w_opponent
        ):
        n_w = len(w_focal)
        if w_focal[-1] != 0.0 and n_w - 1 < self.maxw:
            w_focal = np.pad(w_focal, (0, 1), constant_values=0.0)
            n_w +=1

        w_state_focal = self.transform_score2state_dist(w_focal)
        w_state_opponent = self.transform_score2state_dist(w_opponent)

        self.rates = self.compute_rates(w_state_focal,w_state_opponent)
        
        if self.n_rep_states == 2:
            Q = makeQ.binary_rep_state(*self.rates,n_w)
        elif self.n_rep_states == 3:
            Q = makeQ.three_rep_state(*self.rates,n_w,rich_threshold=self.rich_threshold)
        w_focal_new = (np.eye(n_w) + Q).T @ w_focal

        if self.save_results_all_iterations:
            self.rates_history.append(np.array(self.rates, copy=True))
            self.score_history.append(np.array(w_focal_new, copy=True))
            self.rep_history.append(self.transform_score2state_dist(w_focal_new))

        return w_focal_new

if __name__ == '__main__':
    from helpers import *

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
    onestep_x = OneStepSelf(bb,strategy_x,maxw=10)
    w_x_new = onestep_x(w_x)
    print(w_x_new)
    print(onestep_x.rates)
    # onestep_x = OneStep(bb,strategy_x,strategy_y,rho_x,rho_y,2)
    # w_x_new = onestep_x(w_x,w_x_states,w_y_states)

    # onestep_y = OneStep(bb,strategy_y,strategy_x,rho_y,rho_x,2)
    # w_y_new = onestep_y(w_y,w_y_states,w_x_states)
    
    # print("bb_array:")
    # print(bb)
    # print("strategy_x:")
    # print(strategy_x)
    # print("strategy_y:")
    # print(strategy_y)
    
    # print("rates x:", onestep_x.rates)
    # print("rates y:", onestep_y.rates)
    # print('w_x:',w_x_new)
    # print('w_y:',w_y_new)


    # k = 1
    # expanded_bb_string = expand_binary_bb_string(bb_string)
    # bb_expanded = parse_bb_string(expanded_bb_string,n=3)
    # print('expanded bb string', expanded_bb_string)
    # print('expanded bb array')
    # print(bb_expanded)

    # expanded_strat_x_string = expand_binary_strategy_string(strategy_x_string)
    # expanded_strategy_x = parse_strategy_string(expanded_strat_x_string,n=3)

    # print("expanded strat x:")
    # print(expanded_strategy_x)
    
    # expanded_strat_y_string = expand_binary_strategy_string(strategy_y_string)
    # expanded_strategy_y = parse_strategy_string(expanded_strat_y_string,n=3)
    # print("expanded strat array:")
    # print(expanded_strategy_y)
   
    # w_x_states = transform_distribution_into_states(w_x,k=1)
    # w_y_states = transform_distribution_into_states(w_y,k=1)
    
    # onestep_x = OneStep(bb_expanded,expanded_strategy_x,expanded_strategy_y,rho_x,rho_y,n_rep_states=3,rich_threshold=k)
    # w_x_new = onestep_x(w_x,w_x_states,w_y_states)

    # onestep_y = OneStep(bb_expanded,expanded_strategy_y,expanded_strategy_x,rho_y,rho_x,n_rep_states=3,rich_threshold=k)
    # w_y_new = onestep_y(w_y,w_y_states,w_x_states)

    # print("rates x:", onestep_x.rates)
    # print("rates y:", onestep_y.rates)
    # print('w_x:',w_x_new)
    # print('w_y:',w_y_new)

    
