import numpy as np
from basic_functions_v2 import * #Import all basic functions 
from solution_plots import * 
import time

th = 2.5 #This is the lower bound of types 
M = 6 #Number of markets
dist_mat = np.array([[th, sh] for sh in np.linspace(1.1, 4, M)]) #Parameters of the pareto distributions 
phi_vec = np.ones(M) #Market sizes

T = 30 #Time periods
c_vec = np.linspace(10, 5, T) #Production marginal costs across time 
P_vec = np.linspace(5, 1.5, T) #TCO-fuel across time 
P_vec = np.append(P_vec, 0.0)

FC_mat = np.array([m*np.linspace(5, 1, T) for m in range(M)]) #Entry costs for the producer firm 

upper_ent = 3 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

start_time = time.time() #Timer 
#Solve the problem
value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec, FC_mat, P_vec, 0, T, phi_vec, dist_mat, M, upper_ent)
backward_time = time.time() - start_time
print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))

#exctract the solution 
start_time = time.time()
opt_prof, En, N_opt, mkt_shares, profits, prices =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_mat)
extract_time = time.time() - start_time
print("Time for extract_solution: {:.2f} seconds".format(extract_time))

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

