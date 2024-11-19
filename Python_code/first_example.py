from basic_functions import * #Import all basic functions 
from solution_plots import * 

th = 2.0 #This is the lower bound of types 
M = 2 #Number of markets
dist_mat = np.array([[th, 1.5], [th, 2]]) #Parameters of the pareto distributions 
phi_vec = np.array([1,1.5]) #Market sizes

T = 3 #Time periods
c_vec = np.array([10.0, 5.0, 4.0]) #Production marginal costs across time 
P = np.array([3.0, 2.0, 1.0]) #TCO-fuel across time 

value_functions, entry_decisions = backward_induction(c_vec, P, T, phi_vec, dist_mat, M) #Solve problem 

#exctract the solution 
opt_prof, En, N_opt, mkt_shares, profits =  extract_solution(value_functions, entry_decisions, T, M, c_vec, P, phi_vec, dist_mat)

print("Total profit = {:.2f}".format(opt_prof))
print("Entry decisions = {}".format(En))
print("Network Size = {}".format(N_opt))
print("Total profit = {:.2f}".format(opt_prof))
print("Market shares for market 0: {}".format(mkt_shares[0]))
print("Market shares for market 1: {}".format(mkt_shares[1]))
print("Profits for market 0: {}".format(profits[0]))
print("Profits for market 1: {}".format(profits[1]))


mkt_plot(mkt_shares, phi_vec)
plot_profits(profits)