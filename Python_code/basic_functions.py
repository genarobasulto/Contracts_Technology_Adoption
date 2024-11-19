import numpy as np
from itertools import combinations

def vimt(q, N, th): 
    """
    Utility function. 
    """
    return q*(th - 0.5*q)*N
def vimt_p(q, N, th): 
    """
    Derivative of the utility
    """
    return th - q
def vimt_pi(c,N, th):
    """
    Inverse of the derivative of the utility
    """
    return th - c 
def contracts(N, th, c, P, t_i = 1): 
    """
    This function returns the optimal contract (q,p) for one time.
    """
    q_opt = max(0, vimt_pi(c, N, th))
    if q_opt>0:
        p_opt = max((vimt(q_opt, N, th) - P*t_i)/q_opt, 0)
    else:
        p_opt = 0
    return np.array([q_opt,p_opt])

def get_profits_m(c_vec, P_vec, N_vec, t, T, phi_m, dist_params):
    """
    This function calculates the total profits of entering a market m at time t where the distribution of types follows a pareto distribution.
    """
    t_l = dist_params[0]
    alpha = dist_params[1]
    Pi_m = (N_vec[t] - 1)*(c_vec[t]**2)/(alpha - 1) - P_vec[t]*((t_l/c_vec[t])**alpha)
    for tau in range(t+1, T): 
        Pi_m += (N_vec[tau] - 1)*(c_vec[tau]**2)/(alpha - 1) - (t_l**alpha)*P_vec[tau]*(c_vec[tau-1]**(-alpha) +  c_vec[tau]**(-alpha)) #+ N_vec[t]
    return Pi_m*phi_m

def one_shot_solution(I_vec, t,c_vec, P_vec, N_vec, T, phi_vec, dist_params_mat, M, cont_dict = {}):
    """
    This gives the optimal entry decision at time t based on I_vec, t (state variables).
    """    
    m_opt = -1
    prof_opt = 0.0
    for m in range(0,M):
        I_now = list(I_vec)
        I_now[m] = 1
        if I_vec[m] == 0:
            prof = get_profits_m(c_vec, P_vec, N_vec, t, T, phi_vec[m], dist_params_mat[m]) + cont_dict.get(tuple(I_now), 0.0) 
            if prof > prof_opt:
                prof_opt = prof
                m_opt = m     
    return np.array([prof_opt, m_opt])

def generate_vectors(M, t):
    """
    Generates all posible active market combinations at time t.
    matrix[t, i]==1 indicates market i is active at time t.  
    """
    # Get all possible positions of the ones
    positions = list(combinations(range(M), t))
    # Create matrix of zeros
    matrix = np.zeros((len(positions), M), dtype=int)
    
    # Set ones in the appropriate positions for each row
    for i, pos in enumerate(positions):
        matrix[i, list(pos)] = 1
    
    return matrix

def backward_induction(c_vec, P_vec, T, phi_vec, dist_params_mat, M):
    """
    Performs backward induction to obtain optimal entry decisions and contracts.
    Returns the value function (profits) and entry decisions at all states.  
    """
    t = min(M, T) - 1
    value_functions = {}
    entry_decisions = {}
    N_vec = [j*(j<=t)+ (1-(j<=t))*t + 1 for j in range(0,T)]
    for tau in range(t, -1, -1):
        states = generate_vectors(M, tau)
        for i in range(0, states.shape[0]):
            I_vec = states[i]
            Sol = one_shot_solution(I_vec, tau,c_vec, P_vec, N_vec, T, phi_vec, dist_params_mat, M, value_functions)
            value_functions[tuple(I_vec)] = Sol[0]
            entry_decisions[tuple(I_vec)] = Sol[1]
    return value_functions, entry_decisions

def get_profits_t(c_vec, P_t, N_t, t, M, phi_vec, dist_params_mat, e_t, I_t):
    """
    This function calculates the total profits and market share at time t.
    """
    e_vec = np.array([e_t == m for m in range(M)]) #Indicates if firm enters market m at time t
    Pi_m = {m: 0 for m in range(M)} #Saves the profits for each market at time t
    for m in range(M):
        t_l = dist_params_mat[m, 0]
        alpha = dist_params_mat[m, 1]
        Pi_entry = (N_t - 1)*(c_vec[t]**2)/(alpha - 1) - P_t*((t_l/c_vec[t])**alpha) #If entry now
        Pi_active = (N_t - 1)*(c_vec[t]**2)/(alpha - 1) - (t_l**alpha)*P_t*(c_vec[t-1]**(-alpha) +  c_vec[t]**(-alpha))# + N_vec[t]
        Pi_m[m] = (e_vec[m]*Pi_entry + (1-e_vec[m])*Pi_entry)*phi_vec[m]*I_t[m]
    return Pi_m

def survival(x, alpha, xl):
    return (xl/x)**alpha

def extract_solution(value_functions, entry_decisions, T, M, c_vec, P_vec, phi_vec, dist_params_mat):
    """
    Extracts the relevant data from a solution of the maximization problem.
    Returns: Optimal profit, vector of entry decisions, vector of network sizes (by time).
    """
    start = np.zeros(min(M, T))
    opt_prof = value_functions[tuple(start)] #Total optimal profit
    m_t = int(entry_decisions[tuple(start)])
    En = [m_t]
    #Entry decisions loop w/profits
    N = []
    if m_t > -1:
        N = np.append(N, 1)
    else: 
        N = np.append(N, 0)
    start[m_t] = 1
    prof_t = get_profits_t(c_vec, P_vec[0], N[0], 0, M, phi_vec, dist_params_mat, m_t,start)
    profits = {m:[prof_t[m]] for m in range(M)}
    mkt_shares = {m: [(start[m] == 1)*survival(c_vec[0], dist_params_mat[m, 1], dist_params_mat[m,0])] for m in range(M)} #market shares 
    for t in range(1, min(M, T)):
        m_t = int(entry_decisions[tuple(start)])
        En = np.append(En, m_t)
        if m_t > -1:
            N = np.append(N, N[-1] + 1)
        else: 
            N = np.append(N, N[-1])
        start[m_t] = 1
        prof_t = get_profits_t(c_vec, P_vec[t], N[t], t, M, phi_vec, dist_params_mat, m_t,start)
        for m in range(M):
            profits[m] = np.append(profits[m], prof_t[m])
            mkt_shares[m] = np.append(mkt_shares[m], (start[m] == 1)*survival(c_vec[t], dist_params_mat[m, 1], dist_params_mat[m,0]))
    
    for t in range(min(M, T), T):
        En = np.append(En, -1)
        N = np.append(N, N[-1])
        prof_t = get_profits_t(c_vec, P_vec[t], N[t], t, M, phi_vec, dist_params_mat, m_t,start)
        for m in range(M):
            profits[m] = np.append(profits[m], prof_t[m])
            mkt_shares[m] = np.append(mkt_shares[m], (start[m] == 1)*survival(c_vec[t], dist_params_mat[m, 1], dist_params_mat[m,0]))
    
    return opt_prof, En, N, mkt_shares, profits