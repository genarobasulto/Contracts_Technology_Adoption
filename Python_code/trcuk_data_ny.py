import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:\Users\gbasulto\Box\Carnegie Mellon\AirLiquide\Data\Nyc_truck_data.csv'
trucking_data = pd.read_csv(file_path)

trucking_data = trucking_data[trucking_data['state'] == 'ny']
trucking_data = trucking_data[trucking_data['type_low'] > 0.0]
trucking_data = trucking_data[trucking_data['type_avg'] > 3.0]

print(trucking_data.shape)

# Define a function to extract the first n digits if the zip code is valid
def extract_region(zip_code):
    if isinstance(zip_code, str) and zip_code.isdigit() and len(zip_code) >= 3:
        return zip_code[:2]
    else:
        return np.nan  # Return NaN for invalid entries

# Apply the function to create the 'region' column
trucking_data['region'] = trucking_data['zip_code'].apply(extract_region)

trucking_data.dropna(subset  = ['region'], inplace=True)

print(trucking_data.shape)

###################################
# Pareto Distribution Estimation. #
###################################

# Calculate the empirical CDF for each region
trucking_data['empirical_cdf'] = trucking_data.groupby('region')['type_avg'].rank(pct=True)
trucking_data['lower_empirical_cdf'] = trucking_data.groupby('region')['type_low'].rank(pct=True)
trucking_data['upper_empirical_cdf'] = trucking_data.groupby('region')['type_high'].rank(pct=True)

#This is the lower bound of the distributions.
trucking_data['xm'] = trucking_data.groupby('region')['type_avg'].transform('min')
trucking_data['xm_l'] = trucking_data.groupby('region')['type_low'].transform('min')
trucking_data['xm_u'] = trucking_data.groupby('region')['type_high'].transform('min')

print("-------------------- \n Unique Companies by Region")
print(trucking_data.groupby('region').facility_id.nunique())

print("-------------------- \n Scale Parameter Estimates by Region")
scales_by_region = trucking_data[['region', 'xm_l', 'xm', 'xm_u']].drop_duplicates().groupby('region').first().reset_index()
print(scales_by_region)
scales_by_region.to_csv('scale_parameters_5_regions_nyc.csv', index = False)


# Define a function to calculate shape parameter 'alpha' for each region
def calculate_alpha(group, min_val, x_vals):
    N = len(group)
    xm = group[min_val].iloc[0]  # xm is the minimum value within the region
    # Calculate alpha using the corrected formula
    alpha = N / (np.sum(np.log(group[x_vals])) - N*np.log(xm))
    return alpha

# Calculate alpha for each region and store in a new DataFrame
alpha_by_region = trucking_data.groupby('region').apply(calculate_alpha, 'xm', 'type_avg').reset_index(name='alpha')
alpha_by_region_l = trucking_data.groupby('region').apply(calculate_alpha, 'xm_l', 'type_low').reset_index(name='alpha_l')
alpha_by_region_u = trucking_data.groupby('region').apply(calculate_alpha, 'xm_u', 'type_high').reset_index(name='alpha_h')

alpha_by_region = alpha_by_region.merge(alpha_by_region_l, on = 'region', how = 'left')
alpha_by_region = alpha_by_region.merge(alpha_by_region_u, on = 'region', how = 'left')

# Merge alpha values back into the original dataframe
trucking_data = trucking_data.merge(alpha_by_region, on='region', how='left')
print("-------------------- \n Alphas by Region")
# Add a column with the count of observations for each region
alpha_by_region['num_observations'] = trucking_data.groupby('region')['n_trucks'].transform('sum')
print(alpha_by_region)
alpha_by_region.to_csv("shape_parameters_5_regions_ny.csv", index = False)

 # Plot Alphas by Region
plt.figure(figsize=(10, 6))
plt.plot(alpha_by_region['region'].astype(int), alpha_by_region['alpha'], label='Estimate', marker='o', linestyle='-', color='blue')
plt.plot(alpha_by_region['region'].astype(int), alpha_by_region['alpha_l'], label='Lower Value', linestyle='--', color='black')
plt.plot(alpha_by_region['region'].astype(int), alpha_by_region['alpha_h'], label='Upper Value', linestyle='--', color='red')
plt.xlabel('Region')
plt.ylabel('Alpha')
plt.title('Shape Parameter Estimates')
plt.legend()
plt.grid(True)
plt.show()



# Calculate the estimated CDF using the Pareto distribution for each region
def calculate_estimated_cdf(row, how = "mid"):
    if how == "mid":
        alpha = row['alpha']
        xm = row['xm']
        x = row['type_avg']
    elif how == "lower":
        alpha = row['alpha_l']
        xm = row['xm_l']
        x = row['type_low']
    if how == "upper":
        alpha = row['alpha_h']
        xm = row['xm_u']
        x = row['type_high']
    if x > xm:
        return 1 - (xm / x) ** alpha
    else:
        return 0  # CDF is 0 if x is below the minimum threshold

# Apply the estimated CDF function to each row
trucking_data['estimated_cdf'] = trucking_data.apply(calculate_estimated_cdf, axis=1)
trucking_data['lower_estimated_cdf'] = trucking_data.apply(calculate_estimated_cdf, axis=1, args=("lower",))
trucking_data['upper_estimated_cdf'] = trucking_data.apply(calculate_estimated_cdf, axis=1, args=("upper",))


def plot_CDF(reg, how = "mid"):
    if how == "mid":
        vars_now = ['type_avg', 'empirical_cdf', 'estimated_cdf', 'xm']
        sales = 'type_avg'
    elif how == "lower":
        vars_now = ['type_low', 'lower_empirical_cdf', 'lower_estimated_cdf', 'xm_l']
        sales = 'type_low'
    elif how == "upper":
        vars_now = ['type_high', 'upper_empirical_cdf', 'upper_estimated_cdf', 'xm_u']
        sales = 'type_high'
    # Filter the data for region 1
    region_data = trucking_data[trucking_data['region'] == reg][vars_now].drop_duplicates().sort_values(sales)

    # Plot the empirical CDF vs. the estimated CDF for region 1
    plt.figure(figsize=(10, 6))
    plt.plot(region_data[sales], region_data[vars_now[1]], label='Empirical CDF', marker='o', linestyle='-', color='blue')
    plt.plot(region_data[sales], region_data[vars_now[2]], label='Estimated CDF', marker='x', linestyle='--', color='red')
    plt.xlabel('Demand (tons of H2/year)')
    plt.ylabel('CDF')
    plt.title('Empirical vs. Estimated CDF for Region ' + reg)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_CDF('12', 'mid')