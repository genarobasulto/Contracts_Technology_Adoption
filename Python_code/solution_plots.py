import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import numpy as np
import matplotlib.pyplot as plt

def mkt_plot(mkt_shares, phi_vec):
    """
    Plots the market shares for across time 
    """
    # Number of time periods
    num_periods = len(next(iter(mkt_shares.values())))
    time = np.arange(num_periods)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot market shares for each market
    num_markets = len(mkt_shares)
    for i, (market, shares) in enumerate(mkt_shares.items()):
        grey_value = 0.2 + 0.6 * (i / (num_markets - 1)) 
        plt.plot(time, shares, label="Market {} Share".format(market), color =str(grey_value))

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
    
import numpy as np
import matplotlib.pyplot as plt

def plot_profits(profits_dict):
    """
    Plots the profits of each market across time and the total profit.

    Parameters:
    profits_dict (dict): A dictionary where keys are market identifiers and values 
                         are arrays of weighted profits across time periods for each market.
    """
    # Number of time periods
    num_periods = len(next(iter(profits_dict.values())))
    time = np.arange(num_periods)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot profits for each market
    for market, profits in profits_dict.items():
        plt.plot(time, profits, label="Market {} Flow Profit".format(market), color = "grey")

    # Calculate the total profit at each time period
    total_profit = np.cumsum(np.sum([profits for profits in profits_dict.values()], axis=0))

    # Plot the total profit
    plt.plot(time, total_profit, label="Total Profit", color="black", linestyle="--", linewidth=2)

    # Add labels, legend, and title
    plt.xlabel("Time Period")
    plt.ylabel("Profit")
    plt.title("Flow Profits by Market and Total Profit Over Time")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
