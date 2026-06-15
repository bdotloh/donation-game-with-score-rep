import numpy as np

class ComputeTransitionRates:
    """
    Compute parameters controlling reputation dynamics of focal player's strategy given update rule 
    and other player's strategy. 
    

    """
    def __init__(self,update_rule,strat_focal,strat_other=None,rho_focal=None,n_rep_states=2):
        self.update_rule = update_rule
        self.strat_focal = strat_focal
        self.strat_other = strat_other
        self.n_rep_states=n_rep_states
        self.rho_focal = rho_focal

        if self.strat_other is not None and self.rho_focal is None:
            raise ValueError("If strat_other is not None, rho_focal cannot be None.")
 
        if self.strat_other is None:
            self.type = 'focal-focal'
        else:
            self.type = 'focal-other'

    def __call__(self,repstatedist_focal,repstatedist_other=None):
        rates_self = self.rates_vs_opponent(self.strat_focal,self.strat_focal,repstatedist_focal)
        if self.type == 'focal-focal':
            rates = rates_self
        elif self.type == 'focal-other':
            if repstatedist_other is None:
                raise ValueError("repstatedist_other cannot be None for focal-other comparison.")       
            rates_other = self.rates_vs_opponent(self.strat_focal,self.strat_other,repstatedist_other)
            rates = (self.rho_focal*rates_self + (1-self.rho_focal)*rates_other) 
        
        return rates

    def realised_updates(self, focal_strategy, opponent_strategy):
        """
        given focal player's strategy, other player's strategy, and an update rule, 
        return two matrices, donor_delta and receiver_delta.

        donor_delta tells us what will actually happens to the focal player's reputation score 
        when he is a donor. this depends on the focal player's strategy. more specifically,

        donor_delta[f, i] = score update to a reputation f focal player when he is a donor and meets a receiver with reputation i 

        receiver_delta tells us what will actually happent to the focal player's reputation score 
        when he is a receiver. this depends on the other player player's strategy. more specifcally,

        receiver_delta[f, k] = score update to a reputation f focal player when he is a receiver and meets a donor with reputation k

        
        """
        reps = np.arange(self.n_rep_states)
        f_idx = reps[:, None]   # focal reputation, shape (n_rep_states, 1)
        opp_idx = reps[None, :] # opponent reputation, shape (1, n_rep_states)

        # focal player as donor case:
        # this is a vectorised way of doing   
        # donor_delta[f, i] = update_rule[0, i, focal_strategy[i, f], f]
        # for all f and i 
        donor_delta = self.update_rule[
            0,
            opp_idx,
            focal_strategy.T,
            f_idx,
        ]

        # focal player as receiver case:
        # this is a vectorised way of doing 
        # receiver_delta[f, k] = update_rule[1, f, opponent_strategy[f, k], k]
        # for all f and i 
        receiver_delta = self.update_rule[
            1,
            f_idx,
            opponent_strategy,
            opp_idx,
        ]

        return donor_delta, receiver_delta

    def rates_vs_opponent(
        self,
        focal_strategy, 
        opponent_strategy, 
        repstatedist_opponent
        ):
        """
        returns a vector of transition probabilities, ordered as follows.
        call the vector 'x'.
        if self.n_rep_states = 2,
            x[0] := prob. moving out of 0 state (alpha)
            x[1] := prob. moving out from state i to state i + 1 (i > 0) (beta)
            x[2] := prob. moving out from state i to state i - 1 (i > 0) (gamma)

        if self.n_rep_states = 3,
            x[0] := prob. moving out of 0 state (alpha)
            x[1] := prob. moving out from state i to state i + 1 (0 < i < k) (beta1)
            x[2] := prob. moving out from state i to state i - 1 (0 < i < k) (gamma1)
            x[3] := prob. moving out from state i to state i - 1 (i >= k) (beta2)
            x[4] := prob. moving out from state i to state i - 1 (i >= k) (gamma2)
        """
        donor_delta, receiver_delta = self.realised_updates(
            focal_strategy,
            opponent_strategy
        )

        delta = np.hstack((donor_delta, receiver_delta))
        weights = np.concatenate((repstatedist_opponent,repstatedist_opponent))

        if self.n_rep_states == 2:
            trans_occur = np.array(
                [
                    delta[0] == 1,  
                    delta[1] == 1,
                    delta[1] == -1
                    ]
            )
        
        elif self.n_rep_states == 3:
            trans_occur = np.array(
                [
                    delta[0] == 1,  
                    delta[1] == 1,
                    delta[1] == -1,
                    delta[2] == 1,
                    delta[2] == -1
                    ]
            )
        
        p=trans_occur @ weights

        return p/4

if __name__ == '__main__':
    from helpers import *

    w_x = [0.1,0.7,0.2]
    w_y= [0.2,0.7,0.1]
    w_x_states = Score2StateDistribution(rich_threshold=0)(w_x)
    w_y_states = Score2StateDistribution(rich_threshold=0)(w_y)

    rho_x = 0.2

    bb_string = "CmeetsN:00+00-++ NmeetsC:00-00+--"
    strategy_x_string = "0,0|g,g"
    strategy_y_string = "0,0|0,0"

    bb = parse_bb_string(bb_string)
    strategy_x = parse_strategy_string(strategy_x_string)
    strategy_y = parse_strategy_string(strategy_y_string)

    print("bb_array:")
    print(bb)
    print("strategy_x:")
    print(strategy_x)
    print("strategy_y:")
    print(strategy_y)

    computerates = ComputeTransitionRates(bb,strat_focal=strategy_x,strat_other=strategy_y,rho_focal=rho_x)
    rates_x = computerates(w_x_states,w_y_states)
    print("rates x:", rates_x)
    computerates = ComputeTransitionRates(bb,strat_focal=strategy_y,strat_other=strategy_x,rho_focal=1-rho_x)
    rates_y = computerates(w_y_states,w_x_states)
    print("rates y:", rates_y)

    expanded_bb_string = expand_binary_bb_string(bb_string)
    bb_expanded = parse_bb_string(expanded_bb_string,n_rep_states=3)
    print('expanded bb string', expanded_bb_string)
    print('expanded bb array')
    print(bb_expanded)

    expanded_strat_x_string = expand_binary_strategy_string(strategy_x_string)
    expanded_strategy_x = parse_strategy_string(expanded_strat_x_string,n_rep_states=3)

    print("expanded strat x:")
    print(expanded_strategy_x)
    
    expanded_strat_y_string = expand_binary_strategy_string(strategy_y_string)
    expanded_strategy_y = parse_strategy_string(expanded_strat_y_string,n_rep_states=3)
    print("expanded strat array:")
    print(expanded_strategy_y)
   
    w_x_states = Score2StateDistribution(rich_threshold=1)(w_x)
    w_y_states = Score2StateDistribution(rich_threshold=1)(w_y)

    computerates = ComputeTransitionRates(
        bb_expanded,
        strat_focal=expanded_strategy_x,
        strat_other=expanded_strategy_y,
        rho_focal=rho_x,
        n_rep_states=3
        )
    rates_x = computerates(w_x_states,w_y_states)
    print("rates x:", rates_x)
    computerates = ComputeTransitionRates(
        bb_expanded,
        strat_focal=expanded_strategy_y,
        strat_other=expanded_strategy_x,
        rho_focal=1-rho_x,
        n_rep_states=3
        )
    rates_y = computerates(w_y_states,w_x_states)
    print("rates y:", rates_y)
    