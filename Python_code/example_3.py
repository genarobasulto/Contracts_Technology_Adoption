import numpy as np
from basic_functions_v2 import * #Import all basic functions 
from solution_plots import * 
from no_contracts_functions import *
import time
import matplotlib.pyplot as plt
import pandas as pd

th = 2.0 #This is the lower bound of types 
M = 5 #Number of markets
dist_mat = np.array([[th, 2.2], [th, 2.5], [th, 3], [th, 3.5], [th, 4]]) #Parameters of the pareto distributions 
phi_vec = np.array([1,1,1, 1, 1]) #Market sizes

T = 10 #Time periods
c_vec = np.array([10.0, 7.0, 6, 5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.1]) #Production marginal costs across time 
P_vec = np.linspace(5, 1.5, 10) #TCO-fuel across time 
P_vec = np.append(P_vec, 0.0)

FC_mat = np.zeros((M, T)) #Entry costs for the producer firm 
upper_ent = 2 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

#Solve the problem
value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec, FC_mat, P_vec, 0, T, phi_vec, dist_mat, M, upper_ent)

#exctract the solution 
opt_prof, En, N_opt, mkt_shares, profits, prices =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_mat)

#print(value_functions[st_0], En)

#Print and plot results
if True:
    print("Total profit = {:.2f}".format(opt_prof))
    print("Entry decisions = {}".format(En))
    print("Network Size = {}".format(N_opt))
    for m in range(M):
        print("Market shares for market {}: {}".format(m, mkt_shares[m]))
        print("Profits for market {}: {}".format(m, profits[m]))

    #mkt_plot(mkt_shares, phi_vec)
    #plot_profits(profits)

## One price case
start_time = time.time()
value_functions_onep, entry_decisions_onep, min_t_dict_onep = backward_induction_rec_onep(c_vec, FC_mat, P_vec, 0, T, phi_vec, dist_mat, M, upper_ent)
backward_time = time.time() - start_time
print("Time for backward_induction_rec_onep: {:.2f} seconds".format(backward_time))
#print(value_functions_onep)
#print(value_functions)
#exctract the solution 
opt_prof_onep, En_onep, N_opt_onep, mkt_shares_opt_onep, profits_opt_onep, prices_one =  extract_solution_onep(value_functions_onep, entry_decisions_onep, min_t_dict_onep, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_mat)
print("Total profit One Price = {:.2f}".format(opt_prof_onep))
print("Entry decisions = {}".format(En_onep))
print("Network Size = {}".format(N_opt_onep))
for m in range(M):
    print("Market shares for market {}: {}".format(m, mkt_shares_opt_onep[m]))
    print("Profits for market {}: {}".format(m, profits_opt_onep[m]))
    print("Prices for market {}: {}".format(m, prices_one[m]))

customer_plot(mkt_shares, mkt_shares_opt_onep, phi_vec)
plot_profits_joint(profits, profits_opt_onep)