#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load dataset
file_path = "/Users/saketh/Downloads/Merged_Airlines_With_Revenue.csv"
df = pd.read_csv(file_path)

# Ensure correct format
df['Timeframe_Quarter'] = df['Timeframe_Quarter'].astype(str)
df = df.sort_values('Timeframe_Quarter')

# Filter DAL
dal_df = df[df['Stock Ticker'] == 'DAL'].copy()

# Convert Adj Close to numeric safely
dal_df['Adj Close'] = pd.to_numeric(dal_df['Adj Close'], errors='coerce')

# Drop rows with NaNs in Adj Close
dal_df = dal_df.dropna(subset=['Adj Close'])

# Extract Year and map to datetime
dal_df['Year'] = dal_df['Timeframe_Quarter'].str[:4]
dal_df['Year_Start_Date'] = dal_df['Year'].apply(lambda y: datetime.strptime(f"{y}-01-01", "%Y-%m-%d"))

# Group by year
dal_yearly_df = dal_df.groupby(['Year', 'Year_Start_Date'], as_index=False)['Adj Close'].mean()

# Plotting function
def plot_area(data, title, start_year=None):
    subset = data if not start_year else data[data['Year'] >= start_year]
    plt.figure(figsize=(12, 6))
    plt.fill_between(subset['Year_Start_Date'], subset['Adj Close'], color='#539ecd', alpha=0.7)
    plt.plot(subset['Year_Start_Date'], subset['Adj Close'], color='#336699', marker='o')
    plt.title(f'DAL Yearly Average Adj Close Price - {title}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Generate the plots
plot_area(dal_yearly_df, '1 Year', start_year='2023')
plot_area(dal_yearly_df, '3 Years', start_year='2021')
plot_area(dal_yearly_df, '5 Years', start_year='2019')
plot_area(dal_yearly_df, 'Max')


# In[13]:


from datetime import datetime
import pandas as pd 

file_path = "/Users/saketh/Desktop/DAEN 690/Airline Dataset/Merged_Airlines_With_Revenue.csv"
df = pd.read_csv(file_path)

# Filter for DAL
dal_df = df[df['Stock Ticker'] == 'DAL'].copy()

# Convert Adj Close to numeric safely
dal_df['Adj Close'] = pd.to_numeric(dal_df['Adj Close'], errors='coerce')

# Drop rows with NaNs in Adj Close
dal_df = dal_df.dropna(subset=['Adj Close'])

# Extract Year and map to datetime
dal_df['Year'] = dal_df['Timeframe_Quarter'].str[:4]
dal_df['Year_Start_Date'] = dal_df['Year'].apply(lambda y: datetime.strptime(f"{y}-01-01", "%Y-%m-%d"))

# Group by year
dal_yearly_df = dal_df.groupby(['Year', 'Year_Start_Date'], as_index=False)['Adj Close'].mean()

# Create subsets
dal_yearly_df_max = dal_yearly_df.copy()
dal_yearly_df_5y = dal_yearly_df[dal_yearly_df['Year'].astype(int) >= 2019]
dal_yearly_df_3y = dal_yearly_df[dal_yearly_df['Year'].astype(int) >= 2021]
dal_yearly_df_1y = dal_yearly_df[dal_yearly_df['Year'].astype(int) >= 2023]

# Save them to separate CSVs
dal_yearly_df_max.to_csv('DAL_MAX.csv', index=False)
dal_yearly_df_5y.to_csv('DAL_5Y.csv', index=False)
dal_yearly_df_3y.to_csv('DAL_3Y.csv', index=False)
dal_yearly_df_1y.to_csv('DAL_1Y.csv', index=False)


# In[ ]:




