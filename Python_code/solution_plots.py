import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import numpy as np
import matplotlib.pyplot as plt

def mkt_plot(mkt_shares, phi_vec):
    """
    Plots the market shares for across time 
    """
    # Number of time periods
    num_periods = len(next(iter(mkt_shares.values())))
    time = np.arange(num_periods) + 2024

    # Create the figure and axis
    plt.figure(figsize=(15, 8))

    # Plot market shares for each market
    num_markets = len(mkt_shares)
    for i, (market, shares) in enumerate(mkt_shares.items()):
        grey_value = 0.2 + 0.6 * (i / (num_markets - 1)) 
        plt.plot(time, shares, color =str(grey_value)) #label="Market {} Share".format(market)

    # Calculate the weighted total market share at each time period
    total_market_share = np.zeros(num_periods)
    for i in range(num_periods):
        total_market_share[i] = sum(mkt_shares[m][i] * phi_vec[m] for m in mkt_shares) / phi_vec.sum()

    # Plot the weighted total market share
    plt.plot(time, total_market_share, label="Total Weighted Market Share", color="blue", linestyle="--", linewidth=4)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Market Share")
    plt.title("Market Shares by Market and Weighted Total Market Share Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def customer_plot(mkt_shares_opt, mkt_shares_pes, phi_vec):
    """
    Plots the number of customers across time 
    """
    # Number of time periods
    num_periods = len(next(iter(mkt_shares_opt.values())))
    time = np.arange(num_periods) + 2024

    # Create the figure and axis
    plt.figure(figsize=(15, 8))

    
    total_market_share_opt = np.zeros(num_periods)
    total_market_share_pes = np.zeros(num_periods)
    for i in range(num_periods):
        total_market_share_opt[i] = sum(mkt_shares_opt[m][i] * phi_vec[m] for m in mkt_shares_opt)
        total_market_share_pes[i] = sum(mkt_shares_pes[m][i] * phi_vec[m] for m in mkt_shares_pes)
    # Plot the weighted total market share
    plt.plot(time, total_market_share_opt, label="Optimisitic", color="blue", linestyle="-", linewidth=4)
    plt.plot(time, total_market_share_pes, label="Pessimistic", color="red", linestyle="--", linewidth=4)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Active Trucks")
    #plt.title("Active Customers Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
    
def plot_profits(profits_dict):
    """
    Plots the profits of each market across time and the total profit.

    Parameters:
    profits_dict (dict): A dictionary where keys are market identifiers and values 
                         are arrays of weighted profits across time periods for each market.
    """
    # Number of time periods
    num_periods = len(next(iter(profits_dict.values())))
    time = np.arange(num_periods) + 2024

    # Create the figure and axis
    plt.figure(figsize=(15, 8))

    # Plot profits for each market
    for market, profits in profits_dict.items():
        plt.plot(time, profits, label="Market {} Flow Profit".format(market), color = "grey")

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Flow Profit")
    plt.title("Flow Profits by Market")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Calculate the total profit at each time period
    total_profit = np.cumsum(np.sum([profits for profits in profits_dict.values()], axis=0))

    plt.figure(figsize=(15, 8))
    # Plot the total profit
    plt.plot(time, total_profit, label="Total Profit", color="black", linestyle="--", linewidth=2)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Profit")
    plt.title("Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_profits_joint(profits_dict_opt, profits_dict_pes):
    """
    Plots the total profit in pessimistic and optimistic state.

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

    plt.figure(figsize=(15, 8))
    # Plot the total profit
    plt.plot(time, total_profit_pes, label="Optimistic", color="blue", linestyle="-", linewidth=2)
    plt.plot(time, total_profit_opt, label="Pessimistic", color="red", linestyle="--", linewidth=2)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Profit")
    #plt.title("Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
