import numpy as np

def help_prob(strategy, donor_rep_dist, receiver_rep_dist):
    f"""
    Computes probability that donor helps receiver.

    strategy[i, j] = action when:
        receiver has reputation i
        donor has reputation j

    i = receiver reputation
    j = donor reputation
    
    average of the strategy’s four possible actions,
    weighted by how likely each donor-receiver reputation pair is.
    """

    return receiver_rep_dist @ strategy @ donor_rep_dist

def one_strategy(strategy, w_x, b=2, c=1, role_factor=1/4):
    """
    Computes expected payoffs for strategy x and strategy y
    in the donation game with reputation-dependent strategies.
    """

    h = help_prob(strategy, w_x, w_x)

    # expected payoffs
    payoff = (b - c) * h 

    return role_factor*payoff

def two_strategies(strategy_x, strategy_y, w_x, w_y, rho_x, b=2, c=1, role_factor=1/4):
    """
    Computes expected payoffs for strategy x and strategy y
    in the donation game with reputation-dependent strategies.
    """

    h_xx = help_prob(strategy_x, w_x, w_x)
    h_xy = help_prob(strategy_x, w_x, w_y)

    h_yx = help_prob(strategy_y, w_y, w_x)
    h_yy = help_prob(strategy_y, w_y, w_y)

    # expected payoffs
    payoff_x = (rho_x * (b - c) * h_xx + (1 - rho_x) * (b * h_yx - c * h_xy))
    payoff_y = (rho_x * (b * h_xy - c * h_yx) + (1 - rho_x) * (b - c) * h_yy)

    return role_factor*payoff_x, role_factor*payoff_y


if __name__ == "__main__":
    rho_x = .2
    w0_x = .1
    w0_y = .9

    strategy_x = np.array([0,0,1,1]).reshape((2,2))
    strategy_y = np.array([0,0,0,0]).reshape((2,2))

    w_x = np.array([w0_x,1-w0_x])
    w_y = np.array([w0_y,1-w0_y])

    h_xx = help_prob(strategy_x, w_x, w_x)  # x donor meets x receiver
    h_xy = help_prob(strategy_x, w_x, w_y)  # x donor meets y receiver

    h_yx = help_prob(strategy_y, w_y, w_x)  # y donor meets x receiver
    h_yy = help_prob(strategy_y, w_y, w_y)  # y donor meets y receiver

    payoff_x = one_strategy(strategy_x,w_x)
    payoff_y = one_strategy(strategy_y,w_y)
    # assert(payoff_x-.25<.0001)  # numbers from og calc_stable_w_1strategies julia code
    # assert(payoff_y-.13<.0001)  # numbers from og calc_stable_w_1strategies julia code
    print('strategy x')
    print(strategy_x)
    print(f'payoff if entire population plays x: {payoff_x:.3f}')
    

    print('strategy y')
    print(strategy_y)
    print(f'payoff if entire population plays y: {payoff_y:.3f}')

    print(f'payoff if {rho_x} of the population plays x and the rest plays y')

    payoff_x, payoff_y = two_strategies(strategy_x,strategy_y,w_x,w_y,rho_x)
    # assert(payoff_x-.082<.0001) # numbers from og calc_stable_w_2strategies julia code
    # assert(payoff_y-.175<.0001) # # numbers from og calc_stable_w_2strategies julia code
    print(f'\t payoff(x) = {payoff_x:.3f}')
    print(f'\t payoff(y) = {payoff_y:.3f}')
