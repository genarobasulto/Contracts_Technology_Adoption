import numpy as np
from basic_functions_v2 import * #Import all basic functions 
from solution_plots import * 
import time
import matplotlib.pyplot as plt
import pandas as pd



#Parameters of the pareto distributions 
scale_data = pd.read_csv('scale_parameters_9_regions.csv')
shape_data = pd.read_csv('shape_parameters_9_regions.csv')
dist_data = scale_data.merge(shape_data, on = 'region', how= 'left')
#dist_data['xm'] = 10*dist_data['xm']
dist_mat = np.array(dist_data[['xm', 'alpha']]) #Distribution parameters matrix
M = dist_mat.shape[0] #Number of markets
#print(dist_data.head())
phi_vec = np.array(dist_data['num_observations']).astype('float') #Market Sizes
#phi_vec = phi_vec / np.max(phi_vec)

print("---------------------- \n Distribution Parameters.")
print(dist_mat)
print("---------------------- \n Market Sizes.")
print(phi_vec)

T = 10 #Time periods
c_vec = np.linspace(10, 7, T) # np.linspace(10, 7, T) #Production marginal costs across time 
plt.plot(c_vec)
plt.show()

P_vec = np.linspace(0.449, 0.0564, T) #TCO-fuel across time 

FC_mat = np.array([m*np.linspace(5, 3, T) for m in range(M)]) #Entry costs for the producer firm 

upper_ent = 1 #Max. number of new markets per time period.

st_0 = tuple(np.zeros(M)) #Initial state (no markets are active)

run = True
if run:
    start_time = time.time() #Timer 
    #Solve the problem
    value_functions, entry_decisions, min_t_dict = backward_induction_rec(c_vec, FC_mat, P_vec, 0, T, phi_vec, dist_mat, M, upper_ent)
    backward_time = time.time() - start_time
    print("Time for backward_induction_rec: {:.2f} seconds".format(backward_time))
    #print(value_functions)
    #exctract the solution 
    start_time = time.time()
    opt_prof, En, N_opt, mkt_shares, profits =  extract_solution(value_functions, entry_decisions, min_t_dict, T, M, c_vec, FC_mat, P_vec, phi_vec, dist_mat)
    extract_time = time.time() - start_time
    print("Time for extract_solution: {:.2f} seconds".format(extract_time))


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

