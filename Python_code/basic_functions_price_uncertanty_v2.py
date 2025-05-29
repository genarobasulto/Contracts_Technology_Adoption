# -*- coding: utf-8 -*-
"""

Functions to solve the model in Section 3 of Basulto, 
Montgomery, and Oery (2025) under price uncertainty. 

Created on Tue Apr  1 13:46:01 2025

@author: gbasulto
"""

#Import libraries
import numpy as np
from itertools import combinations
from multiprocessing import Pool
from numpy.random import normal

k = 0.001 #Network effect scaler.

"""
Variable Definitions 
- T(int): time horizon of the model
- M(int): Number of markets 
- c_vec(Mx1 Vector[float]): Vector of marginal cost 
of hydrogen production
- Pt(float): switching costs at time t
- FC_mat(MxT array): matrix of fixed entry costs
- N_t(int): Size of the hydrogen network at time t
- N_tm1(int): Size of the hydrogen network at time t-1
- t(int): time period
- q(float): tons of hydrogen per year consumed
- N(int): Size of the hydrogen network
- th(float): consumer type. 
- c(float): cost per ton of hydrogen
- P(float): switching costs
- t_i(Indicator) = 1 if the hydrogen firm entered the market during the current period.
- phi_vec(Mx1 Vector[float]): Market size vector
- dist_params_mat(Mx2 Matrix[float]): Matrix of parameters of the pareto dist for M markets
- e_vec (Mx1 Vector[int]): Entry =1 if the firm enters the market
- I_t (Mx1 Vector[int]): Entry =1 if the firm is active in the market 
- upper_ent(Int): Maximum number of markets that the firm can enter.
"""

def vimt(q, N, th): 
    """
    Utility function of truck operators.
    """
    return (q/th)*(th - 0.5*q) + k*N
def vimt_p(q, N, th): 
    """
    Derivative of the utility
    """
    return 1.0 - q/th
def vimt_pi(c,N, th):
    """
    Inverse of the derivative of the utility 
    """
    return th*(1.0 - c) 
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

def get_cut(t_l, N_t, Pt, t, T, c_vec):
    """
    Returns the cutoff value for customers (see equation P5.3)
    -------
    """
    return max(t_l, 2.0*(Pt-k*N_t)/(1.0 - c_vec[t]**2))
    
def get_profits_t(c_vec, Pt, FC_mat, N_t, N_tm1, t, T, M, phi_vec, dist_params_mat, e_vec, I_t):
    """
    This function calculates the total profits at time t according to (rt1) and (rt2).
    """
    Pi_m = {m: 0 for m in range(M)} #Saves the profits for each market at time t
    for m in range(M): 
        t_l = dist_params_mat[m, 0]
        alpha = dist_params_mat[m, 1]
        und_th_t = get_cut(t_l, N_t, Pt, t, T, c_vec)
        Ex1 = (alpha/(alpha-1))*und_th_t
        Pi_entry = phi_vec[m]*(0.5*((1.0 - c_vec[t])**2)*Ex1 + k*N_t - Pt)*survival(und_th_t, alpha, t_l) - FC_mat[m,t]
        if t > 0:
            und_th_tm1 = get_cut(t_l, N_tm1, Pt, t-1, T, c_vec) 
            new_cost = max(0.0, Pt*(survival(und_th_t, alpha, t_l) - survival(und_th_tm1, alpha, t_l)))
            Pi_active = phi_vec[m]*((0.5*((1.0 - c_vec[t])**2)*Ex1 + k*N_t)*survival(und_th_t, alpha, t_l) - new_cost)
        else: 
            Pi_active = 0.0
        Pi_m[m] = (Pi_entry*e_vec[m] + (1- e_vec[m])*Pi_active)*I_t[m]
    return Pi_m

def backward_induction_rec(c_vec, FC_mat, Pt,t, T, phi_vec, dist_params_mat, M, upper_ent, value_fun = {}, Next_dict = {}, min_t_dict = {}, P_dict = {}):
    """
    Performs backward induction to obtain optimal entry decisions and contracts.
    Returns the value function (profits) and entry decisions at all states.  
    """
    m_max = min(M, upper_ent*T) + 1 #This is the max number of markets at the end (Shoud this be t??)
    states = np.vstack([generate_vectors(M, m) for m in range(0, m_max)]) #Generate all posible states - permutations of m markets out of M.
    if t == T: #If last period, value functions are zero.
        return {}, {}, {},{}
    elif t< T:
        #Recover value functions for next period 
        Pt1 = 0.95*Pt+normal(0,0.001)
        value_fun, Next_dict, min_t_dict, P_dict = backward_induction_rec(c_vec, FC_mat, Pt1,t + 1, T, phi_vec, dist_params_mat, M, upper_ent, value_fun, Next_dict,P_dict)
        #Solve profits today for each state 
        for i in range(0, states.shape[0]):
            I_vec = states[i] #Current state 
            max_prof = 0.0 #Max profit 
            best_N = I_vec #best entry decision 
            for j in range(0, states.shape[0]): #Chech all entry decisions, save the best one 
                if (I_vec <= states[j]).all() and (sum(states[j] - I_vec) <= upper_ent): #Check if entry decision is allowed (no exit)
                    #Get total profit 
                    prof =  sum(get_profits_t(c_vec, Pt, FC_mat, sum(states[j]), sum(I_vec), t, T, M, phi_vec, dist_params_mat, states[j]-I_vec, states[j]).values())
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

def get_avg_price(t_l, alpha, und_th, N_t, c_t, P_t):
    """
    Computes the average price at a given period.
    """
    network_t = survival(und_th, alpha + 1, t_l)*((k*N_t*alpha*(t_l**alpha))/((1 - c_t)*(alpha+1)*(und_th**(alpha + 1))))
    price_ton = (0.5 + 0.5*c_t)*(survival(und_th,  alpha, t_l)) + network_t
    return 100*price_ton #Transfrom price of Million/ton to usd/kg
def extract_solution(value_functions, Next_dict, min_t_dict, T, M, c_vec, FC_mat, P_dict, phi_vec, dist_params_mat):
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
    prof_t = get_profits_t(c_vec, P_dict[0], FC_mat, N[0], 0, 0, T, M, phi_vec, dist_params_mat, En,next_n)
    profits = {m:[prof_t[m]] for m in range(M)}
    th_lt = {m:get_cut(dist_params_mat[m,0], N[0], P_dict[0], 0, T, c_vec) for m in range(M)}
    mkt_shares = {m: [(next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0])] for m in range(M)} #market shares 
    prices = {m: [(next_n[m] == 1)*get_avg_price(dist_params_mat[m,0], dist_params_mat[m, 1], th_lt[m], N[0], c_vec[0], P_dict[0])] for m in range(M)}

    for t in range(1, T):
        start = next_n
        min_t = min_t_dict[tuple(start)]
        next_n = Next_dict[tuple(start)][M*(t-min_t):M*(t+1 - min_t)]

        En = np.append(En, next_n - start)
        N = np.append(N, sum(next_n))
        prof_t = get_profits_t(c_vec, P_dict[t], FC_mat, N[t], N[t-1], t, T, M, phi_vec, dist_params_mat, next_n - start,next_n)
        th_lt = {m: get_cut(dist_params_mat[m,0], N[t], P_dict[t], t, T, c_vec) for m in range(M)}
        p_t = {m: [get_avg_price(dist_params_mat[m,0], dist_params_mat[m, 1], th_lt[m], N[t], c_vec[t], P_dict[t])] for m in range(M)}
        for m in range(M):
            profits[m] = np.append(profits[m], prof_t[m])
            prices[m] = np.append(prices[m], (next_n[m] == 1)*p_t[m][0])
            mkt_shares[m] = np.append(mkt_shares[m], (next_n[m] == 1)*survival(th_lt[m], dist_params_mat[m, 1], dist_params_mat[m,0]))
    mkt_sh = 0.0
    for m in range(M): 
        mkt_sh += mkt_shares[m][T-1]*(phi_vec[m]/np.sum(phi_vec))
    return opt_prof, En, N, mkt_shares, profits, prices, mkt_sh
