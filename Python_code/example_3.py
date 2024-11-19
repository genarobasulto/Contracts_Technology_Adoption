import numpy as np
from basic_functions_v2 import * #Import all basic functions 
from solution_plots import * 

th = 2.0 #This is the lower bound of types 
M = 5 #Number of markets
dist_mat = np.array([[th, 1.2], [th, 1.5], [th, 2], [th, 2.5], [th, 3]]) #Parameters of the pareto distributions 
phi_vec = np.array([1,1,1, 1, 1]) #Market sizes

T = 10 #Time periods
c_vec = np.array([10.0, 7.0, 6, 5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.1]) #Production marginal costs across time 
P_vec = np.linspace(5, 1.5, 10) #TCO-fuel across time 

FC_mat = np.zeros((M, T)) #Entry costs for the producer firm 
upper_ent = 2 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

#Solve the problem
value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec, FC_mat, P_vec, 0, T, phi_vec, dist_mat, M, upper_ent)

#exctract the solution 
opt_prof, En, N_opt, mkt_shares, profits =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_mat)

#print(value_functions[st_0], En)

#Print and plot results
if True:
    print("Total profit = {:.2f}".format(opt_prof))
    print("Entry decisions = {}".format(En))
    print("Network Size = {}".format(N_opt))
    for m in range(M):
        print("Market shares for market {}: {}".format(m, mkt_shares[m]))
        print("Profits for market {}: {}".format(m, profits[m]))

    mkt_plot(mkt_shares, phi_vec)
    plot_profits(profits)

