#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load and prepare data
df = pd.read_csv("/Users/saketh/Downloads/Merged_Airlines_With_Revenue.csv")
df['Timeframe_Quarter'] = df['Timeframe_Quarter'].astype(str)
df = df.sort_values('Timeframe_Quarter')

# Filter DAL
dal_df = df[df['Stock Ticker'] == 'DAL'].copy()
dal_df['Year'] = dal_df['Timeframe_Quarter'].str[:4]
dal_df['Quarter'] = dal_df['Timeframe_Quarter'].str[-2:]

# Convert to datetime for line plots
quarter_to_month = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
dal_df['Quarter_Start_Date'] = dal_df['Timeframe_Quarter'].apply(
    lambda x: datetime.strptime(f"{x[:4]}-{quarter_to_month[x[-2:]]}-01", "%Y-%m-%d")
)

# Grouped bar chart metrics
bar_metrics = {
    'DAL Operational - PASSENGERS': 'PASSENGERS',
    'DAL Financial - OP_REVENUES': 'OP_REVENUES',
    'DAL Financial - LONG_TERM_DEBT': 'LONG_TERM_DEBT'
}

# Line plot metrics
line_metrics_operational = ['RPM', 'ASM', 'Seat_Utilization', 'Fuel_Efficiency', 'Aircraft_Hours_per_Dep']
line_metrics_financial = ['RASM', 'CASM', 'Revenue_per_Passenger_Mile', 'Cost_per_Mile']

# Ensure numeric
dal_df[list(bar_metrics.values()) + line_metrics_operational + line_metrics_financial] = dal_df[
    list(bar_metrics.values()) + line_metrics_operational + line_metrics_financial
].apply(pd.to_numeric, errors='coerce')

dal_df = dal_df.dropna()

# Color mapping for quarters
quarter_colors = {'Q1': '#B19CD9', 'Q2': '#2ca25f', 'Q3': '#253494', 'Q4': '#41b6c4'}

# 🟪 Grouped Bar Chart
def plot_grouped_bar(data, metric, title):
    pivot = data.pivot(index='Year', columns='Quarter', values=metric)
    pivot = pivot[["Q1", "Q2", "Q3", "Q4"]]  # ensure order

    ax = pivot.plot(kind='bar', figsize=(14, 6), color=['#B39DDB', '#26A69A', '#5C6BC0', '#4DD0E1'])
    plt.title(f'{title} (Quarterly)', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Quarter')
    plt.tight_layout()
    plt.show()

# 📈 Line Plot
def plot_line(data, metric, title_prefix):
    plt.figure(figsize=(12, 5))
    plt.plot(data['Quarter_Start_Date'], data[metric], marker='o', color='blue')
    plt.title(f'{title_prefix} - {metric} per Quarter')
    plt.xlabel('Quarter (Date)')
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ✅ Bar Plots
for title, metric in bar_metrics.items():
    plot_grouped_bar(dal_df, metric, title)

# ✅ Line Plots
for metric in line_metrics_operational:
    plot_line(dal_df, metric, title_prefix="DAL Operational")

for metric in line_metrics_financial:
    plot_line(dal_df, metric, title_prefix="DAL Financial")


# In[ ]:




