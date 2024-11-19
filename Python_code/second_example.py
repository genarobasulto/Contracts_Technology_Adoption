from basic_functions import * #Import all basic functions 
from solution_plots import * 

th = 2.0 #This is the lower bound of types 
M = 5 #Number of markets
dist_mat = np.array([[th, 1.2], [th, 1.5], [th, 2], [th, 2.5], [th, 3]]) #Parameters of the pareto distributions 
phi_vec = np.array([1,1,1, 1, 1]) #Market sizes

T = 10 #Time periods
c_vec = np.array([10.0, 7.0, 6, 5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.1]) #Production marginal costs across time 
P = np.linspace(5, 1.5, 10) #TCO-fuel across time 

value_functions, entry_decisions = backward_induction(c_vec, P, T, phi_vec, dist_mat, M) #Solve problem 

#exctract the solution 
opt_prof, En, N_opt, mkt_shares, profits =  extract_solution(value_functions, entry_decisions, T, M, c_vec, P, phi_vec, dist_mat)

print("Total profit = {:.2f}".format(opt_prof))
print("Entry decisions = {}".format(En))
print("Network Size = {}".format(N_opt))
for m in range(M):
    print("Market shares for market {}: {}".format(m, mkt_shares[m]))
    print("Profits for market {}: {}".format(m, profits[m]))

mkt_plot(mkt_shares, phi_vec)
plot_profits(profits)