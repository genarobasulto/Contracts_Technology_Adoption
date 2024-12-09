import numpy as np
from basic_functions_v2 import * #Import all basic functions 
from solution_plots import * 
from no_contracts_functions import *
from comparison_plots import *
import time
import matplotlib.pyplot as plt
import pandas as pd


#Parameters of the pareto distributions 
scale_data = pd.read_csv('scale_parameters_5_regions_ny_long.csv')
shape_data = pd.read_csv('shape_parameters_5_regions_ny_long.csv')
dist_data = scale_data.merge(shape_data, on = 'region', how= 'left')
dist_data['alpha'] = dist_data['alpha'] + 1.0
dist_mat = np.array(dist_data[['xm', 'alpha']]) #Distribution parameters matrix
M = dist_mat.shape[0] #Number of markets
#print(dist_data.head())
phi_vec = np.array(dist_data['num_observations']).astype('float') #Market Sizes
#phi_vec = phi_vec / np.max(phi_vec)

print("---------------------- \n Distribution Parameters.")
print(dist_mat)
print("---------------------- \n Market Sizes.")
print(phi_vec)

phi_vec = phi_vec/max(phi_vec)

T = 11 #Time periods
c_vec_opt = np.linspace(8, 3, T) # np.linspace(10, 3, T) #Production marginal costs across time 
c_vec_pes = np.linspace(10, 7, T)

P_vec_opt = 0*np.linspace(0.352, 0.04, T) # #np.linspace(0.352, 0.04, T) #TCO-fuel across time 
P_vec_opt = np.append(P_vec_opt, 0.0)
P_vec_pes = 0*np.linspace(0.449, 0.0565, T)
P_vec_pes = np.append(P_vec_pes, 0.0)

FC_mat = np.array([np.linspace(3, 1, T) for m in range(M)]) #Entry costs for the producer firm 

upper_ent = 2 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

run = True
if run:
    print("--- Optimistic Case: Full Solution ---- \n")
    start_time = time.time() #Timer 
    #Solve the problem
    value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec_opt, FC_mat, P_vec_opt, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
    #print(value_functions)
    #exctract the solution 
    opt_prof, En, N_opt, mkt_shares_opt, profits_opt, prices_opt =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec_opt, FC_mat, P_vec_opt, phi_vec, dist_mat)
    print("Total profit Optimistic = {:.2f}".format(opt_prof))
#    print("Entry decisions = {}".format(En))
    print("Network Size = {}".format(N_opt))
    avg_price_opt = np.mean(np.array([np.mean(prices_opt[m]) for m in range(M)]))
    print("Average Price = {}".format(avg_price_opt))

    print("--- Pessimistic Case: Full Solution ---- \n")
    start_time = time.time() #Timer 
    #Solve the problem
    value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec_pes, FC_mat, P_vec_pes, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
    #print(value_functions)
    opt_prof, En, N_opt, mkt_shares_pes, profits_pes, prices_pes =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec_pes, FC_mat, P_vec_pes, phi_vec, dist_mat)
    print("Total profit Pessimistic = {:.2f}".format(opt_prof))
    print("Network Size = {}".format(N_opt))
    avg_price_pes = np.mean(np.array([np.mean(prices_pes[m]) for m in range(M)]))
    print("Average Price = {}".format(avg_price_pes))
    #print(prices_pes)

plot = True
if plot:
    mkt_plot(mkt_shares_opt, phi_vec)
    #plot_profits(profits_opt)
    mkt_plot(mkt_shares_pes, phi_vec)
    #plot_profits(profits_pes)

    #customer_plot(mkt_shares_opt, mkt_shares_pes, phi_vec)
    #plot_profits_joint(profits_pes, profits_opt)

one_price = False
if one_price:
    ## One price case
    print("--- Optimistic Case: Single Market Price Solution ---- \n")
    start_time = time.time()
    value_functions_onep, entry_decisions_onep, min_t_dict_onep = backward_induction_rec_onep(c_vec_opt, FC_mat, P_vec_opt, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
    #print(value_functions_onep)

    #exctract the solution 
    opt_prof_onep, En_onep, N_opt_onep, mkt_shares_opt_onep, profits_opt_onep, prices_onep_opt =  extract_solution_onep(value_functions_onep, entry_decisions_onep, min_t_dict_onep, T, M, c_vec_opt, FC_mat, P_vec_opt, phi_vec, dist_mat)
    print("Total profit One Price = {:.2f}".format(opt_prof_onep))
    print("Entry decisions = {}".format(En_onep))
    print("Network Size = {}".format(N_opt_onep))
    avg_price_opt_one = np.mean(np.array([np.mean(prices_onep_opt[m]) for m in range(M)]))
    print("\n Average Price = {}".format(avg_price_opt_one))

    #print(prices_onep)
    print("--- Pessimistic Case: Single Market Price Solution ---- \n")
    start_time = time.time()
    value_functions_onep_pes, entry_decisions_onep_pes, min_t_dict_onep_pes = backward_induction_rec_onep(c_vec_pes, FC_mat, P_vec_pes, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
    #print(value_functions_onep)

    #exctract the solution 
    opt_prof_onep_pes, En_onep_pes, N_pes_onep, mkt_shares_pes_onep, profits_pes_onep, prices_onep_pes =  extract_solution_onep(value_functions_onep_pes, entry_decisions_onep_pes, min_t_dict_onep_pes, T, M, c_vec_pes, FC_mat, P_vec_pes, phi_vec, dist_mat)
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

plot_one = False 
if plot_one: 
    plot_profits_one(profits_opt, profits_pes, profits_opt_onep, profits_pes_onep)
    plot_prices_one(prices_opt, prices_pes, prices_onep_opt, prices_onep_pes, M)
    plot_mkt_share_one(mkt_shares_opt, mkt_shares_pes, mkt_shares_opt_onep, mkt_shares_pes_onep, phi_vec)