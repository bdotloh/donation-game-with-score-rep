import numpy as np

def binary_rep_state(alpha,beta,gamma,n_w):
    f"""
    make Q matrix for "binary" reputation system
    n_w is the length of the score distribution vector
    shape(Q) = n_w * n_w
    Qij := probabiltiy flow rate from state i to state j
    rows are zero-sum
    """
    Q = np.zeros((n_w,n_w))
    Q[0,0] = -alpha
    Q[0,1] = alpha
    for i in range(1,n_w-1):
        Q[i,i-1] = gamma
        Q[i,i+1] = beta
        Q[i,i] = -(gamma+beta)
    
    Q[-1,-1] = -gamma
    Q[-1,-2] = gamma 
    return Q

def three_rep_state(alpha,beta1,gamma1,beta2,gamma2,n_rep_states,rich_threshold):
    f"""
    make Q matrix for "binary" reputation system
    n_rep_states is the length of the score distribution vector
    shape(Q) = n_w * n_w
    Qij := probabiltiy flow rate from state i to state j
    rows are zero-sum
    """
    Q = np.zeros((n_rep_states,n_rep_states))
    Q[0,0] = -alpha
    Q[0,1] = alpha
    for i in range(1,rich_threshold):
        Q[i,i-1] = gamma1
        Q[i,i+1] = beta1
        Q[i,i] = -(gamma1+beta1)
    for i in range(rich_threshold,n_rep_states-1):
        Q[i,i-1] = gamma2
        Q[i,i+1] = beta2
        Q[i,i] = -(gamma2+beta2)
    
    Q[-1,-1] = -gamma2
    Q[-1,-2] = gamma2 
    return Q

if __name__ == '__main__':
    alpha = .1
    beta = .2
    gamma = .3
    beta2 = .4
    gamma2 = .5
    k=1
    n_w = 10
    
    print('Q for 2 rep state')
    rates = np.array([alpha,beta,gamma])
    Q2 = binary_rep_state(*rates,n_w)
    print(Q2)

    print('Q for 3 rep state')
    rates = np.array([alpha,beta,gamma,beta2,gamma2])
    Q3 = three_rep_state(*rates,n_w,rich_threshold=k)
    print(Q3)
