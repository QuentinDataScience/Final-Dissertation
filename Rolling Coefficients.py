#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import VAR

# Load the dataset
file_path = 'commodities.csv'
data = pd.read_csv(file_path)

# Convert Date column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

# Ensure the date index is sorted and monotonic
data = data.sort_index()

# Set the frequency of the date index to monthly
data = data.asfreq('MS')

# Prepare the data
data_prep = data[['S&P GSCI', 'GPR']].dropna()

# Define a rolling window size
window_size = 18  # 1.5 years of monthly data

# Initialize arrays to store results
rolling_coefs = []
rolling_dates = []

# Perform rolling regression with the smaller window size
for start in range(len(data_prep) - window_size + 1):
    end = start + window_size
    rolling_data = data_prep[start:end].dropna()
    if len(rolling_data) == window_size:
        rolling_var = VAR(rolling_data).fit(maxlags=12)
        rolling_coefs.append(rolling_var.params)
        rolling_dates.append(rolling_data.index[window_size // 2])  # Use the midpoint of the rolling window

# Convert the list to a numpy array for analysis
rolling_coefs = np.array(rolling_coefs)

# Extract coefficients for specific parameters
coef_snp_gsci_gpr = rolling_coefs[:, 1, 1]  # Coefficients for GPR affecting S&P GSCI

# Ensure rolling_dates is a pandas datetime index
rolling_dates = pd.to_datetime(rolling_dates)

# Plot the rolling coefficients with real year dates
plt.figure(figsize=(12, 6))
plt.plot(rolling_dates, coef_snp_gsci_gpr, linestyle='-', linewidth=2)
plt.title('Rolling Coefficients of GPR Impact on S&P GSCI')
plt.xlabel('Year')
plt.ylabel('Coefficient')
plt.grid(True)

# Set the x-axis to display dates from 2000 to 2024
plt.xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2024-01-01'))

# Set x-ticks to show every year for better readability
plt.xticks(pd.date_range(start='2000-01-01', end='2024-01-01', freq='YS').to_pydatetime(), 
           labels=[str(year) for year in range(2000, 2025)], rotation=45)

plt.tight_layout()
plt.show()




# In[ ]:




