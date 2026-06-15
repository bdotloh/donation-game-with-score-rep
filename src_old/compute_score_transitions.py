import numpy as np


def realised_updates(focal_strategy, opponent_strategy, update_rule,n_rep_states=2):
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
    reps = np.arange(n_rep_states)
    f_idx = reps[:, None]   # focal reputation, shape (n_rep_states, 1)
    opp_idx = reps[None, :] # opponent reputation, shape (1, n_rep_states)

    # focal player as donor case:
    # this is a vectorised way of doing   
    # donor_delta[f, i] = update_rule[0, i, focal_strategy[i, f], f]
    # for all f and i 
    donor_delta = update_rule[
        0,
        opp_idx,
        focal_strategy.T,
        f_idx,
    ]

    # focal player as receiver case:
    # this is a vectorised way of doing 
    # receiver_delta[f, k] = update_rule[1, f, opponent_strategy[f, k], k]
    # for all f and i 
    receiver_delta = update_rule[
        1,
        f_idx,
        opponent_strategy,
        opp_idx,
    ]

    return donor_delta, receiver_delta


def rates_vs_opponent(
    focal_strategy, 
    opponent_strategy, 
    update_rule, 
    w_opponent
    ):
    """
    returns a vector of transition probabilities, ordered as follows.
    call the vector 'x'.
    x[0] := prob. moving out of 0 state (alpha)
    x[1] := prob. moving out from state i to state i + 1 (i > 0) (beta)
    x[2] := prob. moving out from state i to state i - 1 (i > 0) (gamma)

    """
    donor_delta, receiver_delta = realised_updates(
        focal_strategy,
        opponent_strategy,
        update_rule
    )

    delta = np.hstack((donor_delta, receiver_delta))
    weights = np.concatenate((w_opponent,w_opponent))

    trans_occur = np.array(
        [
            delta[0] == 1,  
            delta[1] == 1,
            delta[1] == -1
            ]
    )
    p=trans_occur @ weights


    return p/4

def rates_vs_opponent_3_states(
    focal_strategy, 
    opponent_strategy, 
    update_rule, 
    w_opponent
    ):
    """
    returns a vector of transition probabilities, ordered as follows.
    call the vector 'x'.
    x[0] := prob. moving out of 0 state (alpha)
    x[1] := prob. moving out from state i to state i + 1 (0 < i < k) (beta1)
    x[2] := prob. moving out from state i to state i - 1 (0 < i < k) (gamma1)
    x[3] := prob. moving out from state i to state i - 1 (i >= k) (beta2)
    x[4] := prob. moving out from state i to state i - 1 (i >= k) (gamma2)

    """
    donor_delta, receiver_delta = realised_updates(
        focal_strategy,
        opponent_strategy,
        update_rule,
        n_rep_states=3
    )

    delta = np.hstack((donor_delta, receiver_delta))
    weights = np.concatenate((w_opponent,w_opponent))

    trans_occur = np.array(
        [
            delta[0] == 1,  
            delta[1] == 1,
            delta[1] == -1,
            delta[2] == 1,
            delta[2] == -1
            ]
    )
    
    p = trans_occur @ weights

    return p/4


def rates_for_focal(
    focal_strategy,
    other_strategy,
    update_rule,
    w_self,
    w_other,
    rho_self,
    rho_other,
    n_rep_states=2
):
    if n_rep_states == 2:
        rates_self = rates_vs_opponent(
            focal_strategy,
            focal_strategy,
            update_rule,
            w_self
        )

        rates_other = rates_vs_opponent(
            focal_strategy,
            other_strategy,
            update_rule,
            w_other
        )
    
    elif n_rep_states == 3:
        rates_self = rates_vs_opponent_3_states(
            focal_strategy,
            focal_strategy,
            update_rule,
            w_self
        )

        rates_other = rates_vs_opponent_3_states(
            focal_strategy,
            other_strategy,
            update_rule,
            w_other
        )

    return (rho_self*rates_self + rho_other*rates_other) 

if __name__ == '__main__':
    from helpers import *

    w_x = [0.1,0.7,0.2]
    w_y= [0.2,0.7,0.1]
    w_x_states = transform_distribution_into_states(w_x,rich_threshold=0)
    w_y_states = transform_distribution_into_states(w_y,rich_threshold=0)

    rho_x = 0.2
    rho_y = 1 - rho_x
    

    bb_string = "CmeetsN:00+00-++ NmeetsC:00-00+--"
    strategy_x_string = "0,0|g,g"
    strategy_y_string = "0,0|0,0"

    bb = parse_bb_string(bb_string)
    strategy_x = parse_strategy_string(strategy_x_string)
    strategy_y = parse_strategy_string(strategy_y_string)

    rates_x = rates_for_focal(
        focal_strategy=strategy_x,
        other_strategy=strategy_y,
        update_rule=bb,
        w_self=w_x_states,
        w_other=w_y_states,
        rho_self=rho_x,
        rho_other=rho_y,
    )

    rates_y = rates_for_focal(
        focal_strategy=strategy_y,
        other_strategy=strategy_x,
        update_rule=bb,
        w_self=w_y_states,
        w_other=w_x_states,
        rho_self=rho_y,
        rho_other=rho_x,
    )

    print("bb_array:")
    print(bb)
    print("strategy_x:")
    print(strategy_x)
    print("strategy_y:")
    print(strategy_y)
    
    print("rates x:", rates_x)
    print("rates y:", rates_y)


    # expanded_bb_string = expand_binary_bb_string(bb_string)
    # bb_expanded = parse_bb_string(expanded_bb_string,n_rep_states=3)
    # print('expanded bb string', expanded_bb_string)
    # print('expanded bb array')
    # print(bb_expanded)

    # expanded_strat_x_string = expand_binary_strategy_string(strategy_x_string)
    # expanded_strategy_x = parse_strategy_string(expanded_strat_x_string,n_rep_states=3)

    # print("expanded strat x:")
    # print(expanded_strategy_x)
    
    # expanded_strat_y_string = expand_binary_strategy_string(strategy_y_string)
    # expanded_strategy_y = parse_strategy_string(expanded_strat_y_string,n_rep_states=3)
    # print("expanded strat array:")
    # print(expanded_strategy_y)
   
    # w_x_states = transform_distribution_into_states(w_x,rich_threshold=1)
    # w_y_states = transform_distribution_into_states(w_y,rich_threshold=1)

    # rates_x = rates_for_focal(
    #     focal_strategy=expanded_strategy_x,
    #     other_strategy=expanded_strategy_y,
    #     update_rule=bb_expanded,
    #     w_self=w_x_states,
    #     w_other=w_y_states,
    #     rho_self=rho_x,
    #     rho_other=rho_y,
    #     n_rep_states=3
    # )

    # rates_y = rates_for_focal(
    #     focal_strategy=expanded_strategy_y,
    #     other_strategy=expanded_strategy_x,
    #     update_rule=bb_expanded,
    #     w_self=w_y_states,
    #     w_other=w_x_states,
    #     rho_self=rho_y,
    #     rho_other=rho_x,
    #     n_rep_states=3
    # )
    # print("rates x:", rates_x)
    # print("rates y:", rates_y)


    