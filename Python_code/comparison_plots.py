import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import numpy as np
import matplotlib.pyplot as plt

def plot_profits_one(profits_dict_opt, profits_dict_pes, profits_one_opt, profits_one_pes):
    """
    Plots the total profit in pessimistic and optimistic state for one price vs personalized prices.

    Parameters:
    profits_dict (dict): A dictionary where keys are market identifiers and values 
                         are arrays of weighted profits across time periods for each market.
    """
    # Number of time periods
    num_periods = len(next(iter(profits_dict_opt.values())))
    time = np.arange(num_periods) + 2024

    # Calculate the total profit at each time period
    total_profit_opt = np.cumsum(np.sum([profits for profits in profits_dict_opt.values()], axis=0))
    total_profit_pes = np.cumsum(np.sum([profits for profits in profits_dict_pes.values()], axis=0))

    total_profit_opt_one = np.cumsum(np.sum([profits for profits in profits_one_opt.values()], axis=0))
    total_profit_pes_one = np.cumsum(np.sum([profits for profits in profits_one_pes.values()], axis=0))

    plt.figure(figsize=(15, 8))
    # Plot the total profit
    plt.fill_between(time, y1 = total_profit_pes, y2 = total_profit_opt, label="Personalized", color="blue", linestyle="-", linewidth=2, alpha = 0.5)
    plt.fill_between(time, y1 = total_profit_opt_one, y2 = total_profit_pes_one, label="Single Price", color="red", linestyle="--", linewidth=2, alpha = 0.5)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Profit")
    #plt.title("Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


def plot_prices_one(prices_opt, prices_pes, prices_one_opt, prices_one_pes, M):
    """
    Plots the total profit in pessimistic and optimistic state for one price vs personalized prices.

    Parameters:
    profits_dict (dict): A dictionary where keys are market identifiers and values 
                         are arrays of weighted profits across time periods for each market.
    """
    # Number of time periods
    num_periods = len(prices_opt[0])
    time = np.arange(num_periods) + 2024

    # Calculate the total profit at each time period

    average_price_opt = np.array([])
    average_price_pes = np.array([])
    average_price_opt_one = np.array([])
    average_price_pes_one = np.array([])
    for t in range(num_periods): 
        po = 0.0
        pp = 0.0
        po_1 = 0.0
        pp_1 = 0.0
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for m in range(M): 
            po += prices_opt[m][t]
            pp += prices_pes[m][t]
            po_1 += prices_one_opt[m][t]
            pp_1 += prices_one_pes[m][t]
            n1 += (prices_opt[m][t] != 0.0)
            n2 += (prices_pes[m][t] != 0.0)
            n3 += (prices_one_opt[m][t] != 0.0)
            n4 += (prices_one_pes[m][t] != 0.0)
        average_price_opt = np.append(average_price_opt, po/n1)
        average_price_pes = np.append(average_price_pes, pp/n2)
        average_price_opt_one = np.append(average_price_opt_one, po_1/n3)
        average_price_pes_one = np.append(average_price_pes_one, pp_1/n4)
    plt.figure(figsize=(15, 8))
    # Plot the total profit
    plt.fill_between(time, y1 = average_price_opt, y2 = average_price_pes, label="Personalized", color="blue", linestyle="-", linewidth=2, alpha = 0.5)
    plt.fill_between(time, y1 = average_price_opt_one, y2 = average_price_opt_one, label="Single Price", color="red", linestyle="--", linewidth=2, alpha = 0.5)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Average Price")
    #plt.title("Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_mkt_share_one(share_opt, share_pes, share_one_opt, share_one_pes, phi_vec):
    """
    Plots the market shares for across time 
    """
    # Number of time periods
    num_periods = len(next(iter(share_opt.values())))
    time = np.arange(num_periods) + 2024

    # Calculate the weighted total market share at each time period
    total_market_share_opt = np.zeros(num_periods)
    total_market_share_pes = np.zeros(num_periods)
    total_market_share_opt_one = np.zeros(num_periods)
    total_market_share_pes_one = np.zeros(num_periods)
    for i in range(num_periods):
        total_market_share_opt[i] = sum(share_opt[m][i] * phi_vec[m] for m in share_opt) / phi_vec.sum()
        total_market_share_pes[i] = sum(share_pes[m][i] * phi_vec[m] for m in share_pes) / phi_vec.sum()
        total_market_share_opt_one[i] = sum(share_one_opt[m][i] * phi_vec[m] for m in share_one_opt) / phi_vec.sum()
        total_market_share_pes_one[i] = sum(share_one_pes[m][i] * phi_vec[m] for m in share_one_pes) / phi_vec.sum()


    plt.figure(figsize=(15, 8))
    # Plot the market shares
    plt.fill_between(time, y1 = total_market_share_opt, y2 = total_market_share_pes, label="Personalized", color="blue", linestyle="-", linewidth=2, alpha = 0.5)
    plt.fill_between(time, y1 = total_market_share_opt_one, y2 = total_market_share_pes_one, label="Single Price", color="red", linestyle="--", linewidth=2, alpha = 0.5)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Total Market Share")
    #plt.title("Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()