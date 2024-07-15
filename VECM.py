#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
file_path = 'commodities.csv'  # Adjust the path to your file
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

print(data_prep.head())


# In[3]:


from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Perform the Johansen cointegration test
johansen_test = coint_johansen(data_prep, det_order=0, k_ar_diff=1)
print("Eigenvalues:", johansen_test.eig)
print("Trace Statistic:", johansen_test.lr1)
print("Critical Values (90%, 95%, 99%):", johansen_test.cvt)


# In[4]:


from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank

# Determine the number of cointegrating relationships
coint_rank = select_coint_rank(data_prep, det_order=0, k_ar_diff=1, method='trace')
print("Cointegration Rank:", coint_rank.rank)

# Select the optimal lag length
lag_order = select_order(data_prep, maxlags=12, deterministic="ci")
print("Selected Order Summary:")
print(lag_order.summary())


# In[6]:


from statsmodels.tsa.vector_ar.vecm import VECM

# Fit the VECM model
vecm = VECM(data_prep, k_ar_diff=lag_order.aic, coint_rank=coint_rank.rank, deterministic="ci")
vecm_fit = vecm.fit()

# Print the summary of the VECM fit
print(vecm_fit.summary())


# In[ ]:




