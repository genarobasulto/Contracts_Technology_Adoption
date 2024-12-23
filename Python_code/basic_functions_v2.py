import numpy as np
from itertools import combinations
from multiprocessing import Pool

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
def survival(x, alpha, xl):
    """
    Survival function (1-F(x)) of the Pareto distribution with shapre parameter alpha and minimum xl.
    """
    return (xl/x)**alpha

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

def get_cut(t_l, N_t, P_vec, t, T, c_vec):
    return max(t_l, np.sqrt(2*(N_t + P_vec[t] - P_vec[t+1]) + c_vec[t]**2))

def get_profits_t(c_vec, P_vec, FC_mat, N_t, N_tm1, t, T, M, phi_vec, dist_params_mat, e_vec, I_t):
    """
    This function calculates the total profits at time t.
    """
    Pi_m = {m: 0 for m in range(M)} #Saves the profits for each market at time t
    for m in range(M): 
        t_l = dist_params_mat[m, 0]
        alpha = dist_params_mat[m, 1]
        und_th_t = get_cut(t_l, N_t, P_vec, t, T, c_vec)
        Ex1 = (alpha/(alpha-1))*und_th_t
        Ex2 = (alpha/(alpha-2))*(und_th_t**2.0)
        Pi_entry = (0.5*Ex2 - c_vec[t]*Ex1 + N_t**2 + 0.5*c_vec[t] - P_vec[t])*survival(und_th_t, alpha, t_l) - FC_mat[m,t]
        if t > 0:
            und_th_tm1 = get_cut(t_l, N_tm1, P_vec, t-1, T, c_vec) 
            new_cost = max(0.0, P_vec[t]*(survival(und_th_t, alpha, t_l) - survival(und_th_tm1, alpha, t_l)))
            Pi_active = (0.5*Ex2 - c_vec[t]*Ex1 + N_t**2 + 0.5*c_vec[t])*survival(und_th_t, alpha, t_l) - new_cost
        else: 
            Pi_active = 0.0
        Pi_m[m] = (Pi_entry*e_vec[m] + (1- e_vec[m])*Pi_active)*phi_vec[m]*I_t[m]
    return Pi_m

def backward_induction_rec(c_vec, FC_mat, P_vec,t, T, phi_vec, dist_params_mat, M, upper_ent, value_fun = {}, Next_dict = {}, min_t_dict = {}):
    """
    Performs backward induction to obtain optimal entry decisions and contracts.
    Returns the value function (profits) and entry decisions at all states.  
    """
    m_max = min(M, upper_ent*T) + 1 #This is the max number of markets at the end (Shoud this be t??)
    states = np.vstack([generate_vectors(M, m) for m in range(0, m_max)]) #Generate all posible states - permutations of m markets out of M.
    if t == T: #If last period, value functions are zero.
        return {}, {}, {}
    elif t< T:
        #Recover value functions for next period 
        value_fun, Next_dict, min_t_dict = backward_induction_rec(c_vec, FC_mat, P_vec,t + 1, T, phi_vec, dist_params_mat, M, upper_ent, value_fun, Next_dict)
        #Solve profits today for each state 
        for i in range(0, states.shape[0]):
            I_vec = states[i] #Current state 
            max_prof = 0.0 #Max profit 
            best_N = I_vec #best entry decision 
            for j in range(0, states.shape[0]): #Chech all entry decisions, save the best one 
                if (I_vec <= states[j]).all() and (sum(states[j] - I_vec) <= upper_ent): #Check if entry decision is allowed (no exit)
                    #Get total profit 
                    prof =  sum(get_profits_t(c_vec, P_vec, FC_mat, sum(states[j]), sum(I_vec), t, T, M, phi_vec, dist_params_mat, states[j]-I_vec, states[j]).values())
                    prof += value_fun.get(tuple(states[j]), 0.0)  #Plus continuation value
                    if max_prof < prof:
                        #print(I_vec, states[j], t, prof)
                        max_prof = prof
                        best_N = states[j]
            value_fun[tuple(I_vec)] = max_prof #save continuation value
            Next_dict[tuple(I_vec)] = np.append(Next_dict.get(tuple(I_vec), []), best_N) #save entry decisions
            min_t_dict[tuple(I_vec)] = t 
        return value_fun, Next_dict, min_t_dict


def extract_solution(value_functions, Next_dict, min_t_dict, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_params_mat):
    """
    Extracts the relevant data from a solution of the maximization problem. (Forward loop)
    Returns: Optimal profit, vector of entry decisions, vector of network sizes (by time).
    """
    start = np.zeros(M)
    opt_prof = value_functions[tuple(start)] #Total optimal profit
    next_n = Next_dict[tuple(start)][0:M]
    En = next_n - start
    #Entry decisions loop w/profits
    N = [sum(En)]
    prof_t = get_profits_t(c_vec, P_vec, FC_mat, N[0], 0, 0, T, M, phi_vec, dist_params_mat, En,next_n)
    profits = {m:[prof_t[m]] for m in range(M)}
    th_lt = {m:get_cut(dist_params_mat[m,0], N[0], P_vec, 0, T, c_vec) for m in range(M)}
    mkt_shares = {m: [(next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0])] for m in range(M)} #market shares 
    prices = {m: [(next_n[m] == 1)*(0.5*((dist_params_mat[m, 1]/(dist_params_mat[m, 1] - 2))*(th_lt[m]**2) - c_vec[0]**2) + N[0] - P_vec[0])*(survival(th_lt[m],  dist_params_mat[m, 1], dist_params_mat[m,0]))] for m in range(M)}
    for t in range(1, T):
        start = next_n
        min_t = min_t_dict[tuple(start)]
        next_n = Next_dict[tuple(start)][M*(t-min_t):M*(t+1 - min_t)]
        En = np.append(En, next_n - start)
        N = np.append(N, sum(next_n))
        prof_t = get_profits_t(c_vec, P_vec, FC_mat, N[t], N[t-1], t, T, M, phi_vec, dist_params_mat, next_n - start,next_n)
        th_lt = {m: get_cut(dist_params_mat[m,0], N[t], P_vec, t, T, c_vec) for m in range(M)}
        p_t = {m: [(0.5*((dist_params_mat[m, 1]/(dist_params_mat[m, 1] - 2))*(th_lt[m]**2) - c_vec[t]**2) + N[t] - P_vec[t]*En[m])*(survival(th_lt[m],  dist_params_mat[m, 1], dist_params_mat[m,0]))] for m in range(M)}
        for m in range(M):
            profits[m] = np.append(profits[m], prof_t[m])
            prices[m] = np.append(prices[m], (next_n[m] == 1)*p_t[m])
            mkt_shares[m] = np.append(mkt_shares[m], (next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0]))
    return opt_prof, En, N, mkt_shares, profits, prices

