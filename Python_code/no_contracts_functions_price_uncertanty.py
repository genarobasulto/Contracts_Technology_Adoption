import numpy as np
from itertools import combinations
from multiprocessing import Pool
from scipy.optimize import root_scalar, minimize_scalar
from numpy.random import normal

k  = 0.001

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
def survival(x, alpha, xl):
    """
    Survival function (1-F(x)) of the Pareto distribution with shapre parameter alpha and minimum xl.
    """
    return (xl/x)**alpha

def price_eq(p, c, A, t_l, alpha):
    thun = max(A/((1-p)**2), t_l) #max(p + np.sqrt(A), t_l)
    
    Dens = survival(thun,alpha, t_l)
    Q = (1-p)*((alpha/(alpha - 1.0))*thun)*Dens
    
    return -(p-c)*Q

def get_pice_m(c, A, m, dist_params_mat):
    """
    This function calculates the price for market m at time t.
    """
    t_l = dist_params_mat[m, 0]
    alpha = dist_params_mat[m, 1]
    sol = minimize_scalar(price_eq, args = (c, A, t_l, alpha), bounds=(-A, 50.0), method = 'bounded')
    return sol.x

def get_profits_t_onep(ct, FC_mat, P_t, N_t, t, M, phi_vec, dist_params_mat, e_vec, I_t):
    """
    This function calculates the total profits at time t.
    """
    Pi_m = {m: 0 for m in range(M)} #Saves the profits for each market at time t
    for m in range(M): 
        t_l = dist_params_mat[m, 0]
        alpha = dist_params_mat[m, 1]
        A = 2*(P_t - k*N_t)
        A = max(A, 0.1**10)
        p_m = get_pice_m(ct, A,m, dist_params_mat)
        thun =  max(A/((1-p_m)**2), t_l)
        Dens = survival(thun,alpha, t_l)
        Q = (1-p_m)*((alpha/(alpha - 1.0))*thun)*(Dens)
        Pi_entry = (p_m - ct)*Q*phi_vec[m] - FC_mat[m,t] #If entry now
        Pi_active = (p_m - ct)*Q*phi_vec[m]
        Pi_m[m] = (e_vec[m]*Pi_entry + (1-e_vec[m])*Pi_entry)*I_t[m]
    return Pi_m

def backward_induction_rec_onep(c_vec, FC_mat, Pt,t, T, phi_vec, dist_params_mat, M, upper_ent, value_fun = {}, Next_dict = {}, min_t_dict = {}, P_dict = {}):
    """
    Performs backward induction to obtain optimal entry decisions and contracts.
    Returns the value function (profits) and entry decisions at all states.  
    """
    m_max = min(M, upper_ent*T) + 1 #This is the max number of markets at the end (Shoud this be t??)
    states = np.vstack([generate_vectors(M, m) for m in range(0, m_max)]) #Generate all posible states - permutations of m markets out of M.
    if t == T: #If last period, value functions are zero.
        return {}, {}, {}, {}
    elif t< T:
        #Recover value functions for next period 
        Pt1 = 0.95*Pt+normal(0,0.001)
        value_fun, Next_dict, min_t_dict, P_dict = backward_induction_rec_onep(c_vec, FC_mat, Pt1,t + 1, T, phi_vec, dist_params_mat, M, upper_ent, value_fun, Next_dict, P_dict)
        #Solve profits today for each state 
        for i in range(0, states.shape[0]):
            I_vec = states[i] #Current state 
            max_prof = 0.0 #Max profit 
            best_N = I_vec #best entry decision 
            for j in range(0, states.shape[0]): #Chech all entry decisions, save the best one 
                if (I_vec <= states[j]).all() and (sum(states[j] - I_vec) <= upper_ent): #Check if entry decision is allowed (no exit)
                    #Get total profit 
                    prof =  sum(get_profits_t_onep(c_vec[t], FC_mat, Pt, sum(states[j]), t, M, phi_vec, dist_params_mat, states[j]-I_vec, states[j]).values())
                    prof += value_fun.get(tuple(states[j]), 0.0)  #Plus continuation value
                    if max_prof < prof:
                        #print(I_vec, states[j], t, prof)
                        max_prof = prof
                        best_N = states[j]
            value_fun[tuple(I_vec)] = max_prof #save continuation value
            Next_dict[tuple(I_vec)] = np.append(Next_dict.get(tuple(I_vec), []), best_N) #save entry decisions
            min_t_dict[tuple(I_vec)] = t 
        P_dict[t] = Pt
        return value_fun, Next_dict, min_t_dict, P_dict

def extract_solution_onep(value_functions, Next_dict, min_t_dict, T, M, c_vec, FC_mat, P_dict, phi_vec, dist_params_mat):
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
    prof_t = get_profits_t_onep(c_vec[0], FC_mat, P_dict[0], N[0], 0, M, phi_vec, dist_params_mat, En,next_n)
    profits = {m:[prof_t[m]] for m in range(M)}
    A = 2*(P_dict[0] - k*N[0])
    A = max(A, 0.1**10)
    p_t = {m:[get_pice_m(c_vec[0], A,m, dist_params_mat)] for m in range(M)}
    prices = {m:[(next_n[m] == 1)*p_t[m][0]] for m in range(M)}
    th_lt = {m:[max(A/((1-p_t[m][0])**2), dist_params_mat[m, 0])] for m in range(M)}
    mkt_shares = {m: [(next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0])] for m in range(M)} #market shares 

    for t in range(1, T):
        start = next_n
        min_t = min_t_dict[tuple(start)]
        next_n = Next_dict[tuple(start)][M*(t-min_t):M*(t+1 - min_t)]
        En = np.append(En, next_n - start)
        N = np.append(N, sum(next_n))
        prof_t = get_profits_t_onep(c_vec[t],FC_mat, P_dict[t], N[t], t, M, phi_vec, dist_params_mat, next_n - start,next_n)
        A = 2*(P_dict[t]- k*N[t])
        A = max(A, 0.1**10)
        p_t = {m:[(next_n[m] == 1)*get_pice_m(c_vec[t], A,m, dist_params_mat)] for m in range(M)}
        th_lt = {m:[max(A/((1-p_t[m][0])**2), dist_params_mat[m, 0])] for m in range(M)} 
        for m in range(M):
            profits[m] = np.append(profits[m], prof_t[m])
            prices[m] = np.append(prices[m], p_t[m][0])
            mkt_shares[m] = np.append(mkt_shares[m], (next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0]))

    mkt_sh = 0.0
    for m in range(M): 
        mkt_sh += mkt_shares[m][T-1]*(phi_vec[m]/np.sum(phi_vec))
    return opt_prof, En, N, mkt_shares, profits, prices, mkt_sh
 