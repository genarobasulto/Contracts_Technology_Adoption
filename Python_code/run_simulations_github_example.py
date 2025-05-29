# -*- coding: utf-8 -*-
"""
Created on Wed May 28 20:54:13 2025

@author: gbasulto
"""

import numpy as np
from basic_functions_price_uncertanty_v2 import * #Import all basic functions 
from solution_plots import * 
from no_contracts_functions_price_uncertanty import *
from comparison_plots import *
import time
import matplotlib.pyplot as plt
import pandas as pd

th = 2.0 #This is the lower bound of types 
M = 5 #Number of markets
dist_mat = np.array([[th, 2.2], [th, 2.5], [th, 3], [th, 3.5], [th, 4]]) #Parameters of the pareto distributions 
phi_vec = np.array([1,1,1, 1, 1]) #Market sizes

T = 10 #Time periods

FC_mat = np.zeros((M, T)) #Entry costs for the producer firm 
upper_ent = 2 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

print("---------------------- \n Distribution Parameters.")
print(dist_mat)
print("---------------------- \n Market Sizes.")
print(phi_vec)

c_start_vec = np.linspace(5,10, 3)
c_end_vec = np.linspace(2, 4, 3)
P0_vec = np.linspace(3.5, 4.5, 3)
#Entry costs
k = 0.001 #Network effect.

entry_cost_vec = [1.0, 3.0] #Multiplier of entry costs for robustness


def extract_entry_year(T, M, En_st):
    mkt_entry = np.hstack([np.where(En_st[t*M:M*(t+1)] == 1)[0] + 1 if len(np.where(En_st[t*M:M*(t+1)] == 1)[0]) >0 else 0  for t in range(T)]) #This saves market entry in chronological order.
    times_entry = np.hstack([(i+2025)*(np.where(En_st[i*M:M*(i+1)] == 1.0)[0].size > 0) for i in range(T)]) #This is the corresponding time of entry.
    entry_dict = {mkt_entry[t]:times_entry[t] for t in range(T)}
    return {m:entry_dict.get(m+1, 0) for m in range(M)}

def run_simulations(T, M, c_start_vec, c_end_vec,entry_cost_vec, P0_vec,phi_vec, dist_mat, print_lev = 0):
    N_sol = 1
    sol_dict = {}
    for cost_mult in entry_cost_vec:
        print('--------- \n Starting for cost_mult = ', cost_mult)
        FC_mat = cost_mult*FC_mat
        for upper_ent in [1, 2, 3]:
            for c_0 in c_start_vec:
                for c_T in c_end_vec:
                    for P0 in P0_vec:
                        c_vec = np.linspace(c_0, c_T, T) #Production marginal costs across time 
                        if print_lev == 1:
                            print("--- Starting Full Solution ---- \n")
                            print("c0 = %0.4f, cT = %0.4f, P0 = %0.4f, entry = %i \n" %(c_0, c_T, P0, upper_ent))
                        start_time = time.time() #Timer 
                        #Solve the problem
                        value_functions, entry_decisions, min_t_dict, P_dict = backward_induction_rec(c_vec, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
                        backward_time = time.time() - start_time
                        
                        #print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
                        #exctract the solution 
                        opt_prof, En, N_opt, mkt_shares_opt, profits_opt, prices_opt, mkt_sh =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec, FC_mat, P_dict, phi_vec, dist_mat)
                        avg_price_opt = np.mean(np.array([np.mean(prices_opt[m]) for m in range(M)]))
                        
                        value_functions_onep, entry_decisions_onep, min_t_dict_onep, P_dict_onep = backward_induction_rec_onep(c_vec, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
                        backward_time = time.time() - start_time
                        #print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
        
                        #exctract the solution 
                        opt_prof_onep, En_onep, N_opt_onep, mkt_shares_opt_onep, profits_opt_onep, prices_opt_onep, mkt_sh_onep =  extract_solution_onep(value_functions_onep, entry_decisions_onep, min_t_dict_onep, T, M, c_vec, FC_mat, P_dict_onep, phi_vec, dist_mat)
                        avg_price_opt_one = np.mean(np.array([np.mean(prices_opt_onep[m]) for m in range(M)]))
                        
                        
                        #Save long term conctracts sol
                        
                        sol_now= {'T':T, 'c0':c_0, 'cT':c_T, 'P0':P0, 'entry_costs':FC_mat,'Network_size':N_opt[-1],'P_vec':[[P_dict[t] for t in range(T)]], 'avg_price':avg_price_opt, 'market_share':mkt_sh, 'total_profits':opt_prof}
                        shr_now = {'shares_mkt_'+str(m):mkt_shares_opt[m] for m in range(M)}
                        entry_dict = extract_entry_year(T, M, En)
                        entry_now = {'entry_mkt_'+str(m):entry_dict[m] for m in range(M)}
                        prices_now = {'prices_mkt_'+str(m):prices_opt[m] for m in range(M)}
                        profits_now = {'profits_mkt_'+str(m):profits_opt[m] for m in range(M)}
                        
                        #Save one price solutions
                        sol_onep = {'Network_size_onep':N_opt_onep[-1],'P_vec_onep':[[P_dict_onep[t] for t in range(T)]], 'avg_price_onep':avg_price_opt_one, 'market_share_onep':mkt_sh_onep, 'total_profits_onep':opt_prof_onep}
                        shr_now_onep = {'onep_shares_mkt_'+str(m):mkt_shares_opt_onep[m] for m in range(M)}
                        entry_dict_onep = extract_entry_year(T, M, En_onep)
                        entry_now_onep = {'onep_entry_mkt_'+str(m):entry_dict_onep[m] for m in range(M)}
                        prices_now_onep = {'onep_prices_mkt_'+str(m):prices_opt_onep[m] for m in range(M)}
                        profits_now_onep = {'onep_profits_mkt_'+str(m):profits_opt_onep[m] for m in range(M)}
                        
                        
                        sol_dict[N_sol] = sol_now|shr_now|entry_now|prices_now|profits_now|sol_onep|shr_now_onep|entry_now_onep|prices_now_onep|profits_now_onep
                        N_sol+=1
                        if print_lev == 1:
                            print("Total profit = {:.2f}".format(opt_prof))
                            print("Network Size = {}".format(N_opt))
                            print("Average Price = {}".format(avg_price_opt))
                            print("Total profit One Price = {:.2f}".format(opt_prof_onep))
                            print("Entry decisions = {}".format(En_onep))
                            print("Network Size = {}".format(N_opt_onep))
                            print("\n Average Price = {}".format(avg_price_opt_one))
    sol_data = pd.DataFrame(sol_dict).T
    return sol_data, N_sol

sol_data_1, N_sol_1 = run_simulations(T, M, c_start_vec, c_end_vec, entry_cost_vec, P0_vec,phi_vec, dist_mat)


#sol_data_1.to_csv('solution_outputs_042125.csv')