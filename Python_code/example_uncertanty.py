import numpy as np
from basic_functions_price_uncertanty_v2 import * #Import all basic functions 
from solution_plots import * 
from no_contracts_functions_price_uncertanty import *
from comparison_plots import *
import time
import matplotlib.pyplot as plt
import pandas as pd


#Parameters of the pareto distributions 
scale_data = pd.read_csv('scale_parameters_5_regions_ny_long.csv')
shape_data = pd.read_csv('shape_parameters_5_regions_ny_long.csv')
dist_data = scale_data.merge(shape_data, on = 'region', how= 'left')
dist_data['alpha'] = dist_data['alpha'] 
dist_mat = np.array(dist_data[['xm', 'alpha']]) #Distribution parameters matrix
M = dist_mat.shape[0] #Number of markets
#print(dist_data.head())
phi_vec = np.array(dist_data['num_observations']).astype('float') #Market Sizes


print("---------------------- \n Distribution Parameters.")
print(dist_mat)
print("---------------------- \n Market Sizes.")
print(phi_vec)


T = 11 #Time periods
c_vec_opt = np.linspace(0.008, 0.003, T)# np.linspace(10, 3, T) #Production marginal costs across time 
c_vec_pes = np.linspace(0.010, 0.007, T)

P0 = 4.5


phi_vec = phi_vec/min(phi_vec) #Normalize market sizes
total_expected_demand = phi_vec*dist_data['alpha']*dist_data['xm']/(dist_data['alpha'] - 1) #Total demand based on Pareto dist. and market size
total_daily_demand = total_expected_demand.to_numpy()/(365) #We divide over 10 if we expect around 10% mkt. share
FC_mat = np.array([c*(total_daily_demand) for c in np.linspace(51,30, T)]).T
FC_mat
phi_vec = phi_vec/max(phi_vec)
upper_ent = 1 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

run = True
if run:
    print("--- Optimistic Case: Full Solution ---- \n")
    start_time = time.time() #Timer 
    #Solve the problem
    value_functions, entry_decisions, min_t_dict, P_dict = backward_induction_rec(c_vec_opt, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
    #print(value_functions)
    #exctract the solution 
    opt_prof, En, N_opt, mkt_shares_opt, profits_opt, prices_opt, final_shares_opt =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec_opt, FC_mat, P_dict, phi_vec, dist_mat)
    print("Total profit Optimistic = {:.2f}".format(opt_prof))
#    print("Entry decisions = {}".format(En))
    print("Network Size = {}".format(N_opt))
    avg_price_opt = np.mean(np.array([np.mean(prices_opt[m]) for m in range(M)]))
    print("Average Price = {}".format(avg_price_opt))

    print("--- Pessimistic Case: Full Solution ---- \n")
    start_time = time.time() #Timer 
    #Solve the problem
    value_functions, entry_decisions, min_t_dict, P_dict = backward_induction_rec(c_vec_pes, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
    #print(value_functions)
    opt_prof, En, N_opt, mkt_shares_pes, profits_pes, prices_pes, final_shares_opt =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec_pes, FC_mat, P_dict, phi_vec, dist_mat)
    print("Total profit Pessimistic = {:.2f}".format(opt_prof))
    print("Network Size = {}".format(N_opt))
    avg_price_pes = np.mean(np.array([np.mean(prices_pes[m]) for m in range(M)]))
    print("Average Price = {}".format(avg_price_pes))
    #print(prices_pes)

plot = True
if plot:
    mkt_plot(mkt_shares_opt, phi_vec)
    plot_profits(profits_opt)
    mkt_plot(mkt_shares_pes, phi_vec)
    plot_profits(profits_pes)

    customer_plot(mkt_shares_opt, mkt_shares_pes, phi_vec)
    plot_profits_joint(profits_pes, profits_opt)

one_price = True

if one_price:
    ## One price case
    print("--- Optimistic Case: Single Market Price Solution ---- \n")
    start_time = time.time()
    value_functions_onep, entry_decisions_onep, min_t_dict_onep, P_dict_onep = backward_induction_rec_onep(c_vec_opt, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
    #print(value_functions_onep)

    #exctract the solution 
    opt_prof_onep, En_onep, N_opt_onep, mkt_shares_opt_onep, profits_opt_onep, prices_onep_opt, shares_onep_opt =  extract_solution_onep(value_functions_onep, entry_decisions_onep, min_t_dict_onep, T, M, c_vec_opt, FC_mat, P_dict_onep, phi_vec, dist_mat)
    print("Total profit One Price = {:.2f}".format(opt_prof_onep))
    print("Entry decisions = {}".format(En_onep))
    print("Network Size = {}".format(N_opt_onep))
    avg_price_opt_one = np.mean(np.array([np.mean(prices_onep_opt[m]) for m in range(M)]))
    print("\n Average Price = {}".format(avg_price_opt_one))

    #print(prices_onep)
    print("--- Pessimistic Case: Single Market Price Solution ---- \n")
    start_time = time.time()
    value_functions_onep_pes, entry_decisions_onep_pes, min_t_dict_onep_pes,P_dict_onep_pes = backward_induction_rec_onep(c_vec_pes, FC_mat, P0, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
    #print(value_functions_onep)

    #exctract the solution 
    opt_prof_onep_pes, En_onep_pes, N_pes_onep, mkt_shares_pes_onep, profits_pes_onep, prices_onep_pes, shares_onep_pes =  extract_solution_onep(value_functions_onep_pes, entry_decisions_onep_pes, min_t_dict_onep_pes, T, M, c_vec_pes, FC_mat, P_dict_onep_pes, phi_vec, dist_mat)

    print("Total profit One Price = {:.2f}".format(opt_prof_onep_pes))
    print("Entry decisions = {}".format(En_onep_pes))
    print("Network Size = {}".format(N_pes_onep))
    avg_price_pes_one = np.mean(np.array([np.mean(prices_onep_pes[m]) for m in range(M)]))
    print("Average Price = {}".format(avg_price_pes_one))
    if False:
        for m in range(M):
            print("Market shares for market {}: {}".format(m, mkt_shares_opt_onep[m]))
            print("Profits for market {}: {}".format(m, profits_opt_onep[m]))
            print("Prices for market {}: {}".format(m, prices_onep[m]))

plot_one = True 
if plot_one: 
    plot_profits_one(profits_opt, profits_pes, profits_opt_onep, profits_pes_onep)
    plot_prices_one(prices_opt, prices_pes, prices_onep_opt, prices_onep_pes, M)
    plot_mkt_share_one(mkt_shares_opt, mkt_shares_pes, mkt_shares_opt_onep, mkt_shares_pes_onep, phi_vec)