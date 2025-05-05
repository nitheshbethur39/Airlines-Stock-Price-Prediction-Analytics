#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "Merged_Airlines_With_Revenue.csv"
df = pd.read_csv(file_path)

# Filter for ALGT airline
algt_df = df[df['Stock Ticker'] == 'AAL'].copy()

# Select and convert features
bar_features = {
    'OP_REVENUES': 'Operating Revenue ($)',
    'PASSENGERS': 'Number of Passengers'
}
line_features = {
    'ASM': 'Available Seat Miles',
    'Fuel_Efficiency': 'Fuel Efficiency',
    'CASM': 'Cost per ASM'
}

# Extract year and quarter
algt_df['Year'] = algt_df['Timeframe_Quarter'].str[:4]
algt_df['Quarter'] = algt_df['Timeframe_Quarter'].str[-2:]

# Convert feature columns to numeric
for col in list(bar_features.keys()) + list(line_features.keys()):
    algt_df[col] = pd.to_numeric(algt_df[col], errors='coerce')

# Drop missing values
algt_df = algt_df.dropna(subset=list(bar_features.keys()) + list(line_features.keys()))

# --- Bar Plot Function ---
def plot_grouped_bar(df, feature, ylabel):
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    colors = ['#B39DDB', '#26A69A', '#5C6BC0', '#4DD0E1']
    width = 0.18

    grouped = df.groupby(['Year', 'Quarter'])[feature].sum().unstack().sort_index()
    labels = grouped.index.tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(16, 6))
    for i, quarter in enumerate(quarters):
        if quarter in grouped.columns:
            values = grouped[quarter].values
            plt.bar(x + i * width - 1.5 * width, values, width=width,
                    label=quarter, color=colors[i], edgecolor='white')

    plt.xticks(ticks=x, labels=labels)
    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.title(f'AAL {ylabel} by Quarter')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(title='Quarter')
    plt.tight_layout()
    plt.show()

# --- Line Plot Function ---
def plot_line(df, feature, ylabel):
    agg = df[['Timeframe_Quarter', feature]].dropna().sort_values('Timeframe_Quarter')
    agg = agg.groupby('Timeframe_Quarter')[feature].sum().reset_index()

    plt.figure(figsize=(16, 5))
    plt.plot(agg['Timeframe_Quarter'], agg[feature], marker='o', color='#336699')
    plt.xticks(rotation=45)
    plt.title(f'AAL {ylabel} Over Time')
    plt.xlabel('Quarter')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- Generate Plots ---

# Bar plots
for col, label in bar_features.items():
    plot_grouped_bar(algt_df, col, label)

# Line plots
for col, label in line_features.items():
    plot_line(algt_df, col, label)


# In[1]:


import pandas as pd
import os

# Set the file path (make sure the file is in the same directory or adjust the path)
file_path = "Merged_Airlines_With_Revenue.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Filter for AAL airline
aal_df = df[df['Stock Ticker'] == 'AAL'].copy()

# Extract year and quarter
aal_df['Year'] = aal_df['Timeframe_Quarter'].str[:4]
aal_df['Quarter'] = aal_df['Timeframe_Quarter'].str[-2:]

# Features to extract
features = {
    'OP_REVENUES': 'Operating Revenue ($)',
    'PASSENGERS': 'Number of Passengers',
    'ASM': 'Available Seat Miles',
    'Fuel_Efficiency': 'Fuel Efficiency',
    'CASM': 'Cost per ASM'
}

# Convert feature columns to numeric
for col in features.keys():
    aal_df[col] = pd.to_numeric(aal_df[col], errors='coerce')

# Drop rows with missing data in any feature
aal_df.dropna(subset=features.keys(), inplace=True)

# Create output directory
output_dir = "AAL_Feature_CSVs"
os.makedirs(output_dir, exist_ok=True)

# Save each feature to a separate CSV
for feature_col, readable_name in features.items():
    feature_df = aal_df[['Year', 'Quarter', feature_col]].groupby(['Year', 'Quarter']).sum().reset_index()
    feature_df.rename(columns={feature_col: 'Value'}, inplace=True)
    output_filename = os.path.join(output_dir, f"AAL_{feature_col}.csv")
    feature_df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")


# In[2]:


import pandas as pd
import os

# Set the file path
file_path = "Merged_Airlines_With_Revenue.csv"

# Load dataset
df = pd.read_csv(file_path)

# Filter for AAL
aal_df = df[df['Stock Ticker'] == 'AAL'].copy()

# Extract year and quarter
aal_df['Year'] = aal_df['Timeframe_Quarter'].str[:4]
aal_df['Quarter'] = aal_df['Timeframe_Quarter'].str[-2:]

# Combine into single 'Quarter' column (e.g., 2019-Q1)
aal_df['Quarter_Label'] = aal_df['Year'] + '-Q' + aal_df['Quarter']

# Selected features
selected_features = ['ASM', 'Fuel_Efficiency', 'CASM']

# Convert selected features to numeric
for col in selected_features:
    aal_df[col] = pd.to_numeric(aal_df[col], errors='coerce')

# Drop rows with missing data
aal_df.dropna(subset=selected_features, inplace=True)

# Output directory
output_dir = "AAL_Selected_Features"
os.makedirs(output_dir, exist_ok=True)

# Export each selected feature
for feature in selected_features:
    feature_df = aal_df[['Quarter_Label', feature]].groupby('Quarter_Label').sum().reset_index()
    feature_df.columns = ['Quarter', 'Value']
    output_filename = os.path.join(output_dir, f"AAL_{feature}.csv")
    feature_df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "Merged_Airlines_With_Revenue.csv"
df = pd.read_csv(file_path)

# Filter for AAL airline
aal_df = df[df['Stock Ticker'] == 'AAL'].copy()

# Feature set
features = {
    'OP_REVENUES': 'Operating Revenue ($)',
    'PASSENGERS': 'Number of Passengers',
    'ASM': 'Available Seat Miles',
    'Fuel_Efficiency': 'Fuel Efficiency',
    'CASM': 'Cost per ASM',
    'Adj Close': 'Adjusted Close Price',
    'Volume': 'Stock Volume',
    'Open': 'Open Price',
    'High': 'High Price',
    'Low': 'Low Price'
}

# Convert all to numeric
for col in features.keys():
    aal_df[col] = pd.to_numeric(aal_df[col], errors='coerce')

# Drop rows with NaN in all these features
aal_df.dropna(subset=features.keys(), how='all', inplace=True)

# Calculate total sum of each feature
feature_sums = {
    features[col]: aal_df[col].sum(skipna=True) for col in features
}

# Sort and get top 5
top_5 = dict(sorted(feature_sums.items(), key=lambda x: x[1], reverse=True)[:5])

# --- Plot Top 5 Features ---
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5.keys(), top_5.values(), color='#5C6BC0', edgecolor='black')
plt.title('Top 5 Features for AAL (by Total Value)', fontsize=14)
plt.ylabel('Total Value (Sum over all quarters)')
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "Merged_Airlines_With_Revenue.csv"
df = pd.read_csv(file_path)

# Filter for AAL
aal_df = df[df['Stock Ticker'] == 'AAL'].copy()

# Define features to analyze
features = {
    'OP_REVENUES': 'Operating Revenue ($)',
    'PASSENGERS': 'Number of Passengers',
    'ASM': 'Available Seat Miles',
    'Fuel_Efficiency': 'Fuel Efficiency',
    'CASM': 'Cost per ASM',
    'Adj Close': 'Adjusted Close Price',
    'Volume': 'Stock Volume',
    'Open': 'Open Price',
    'High': 'High Price',
    'Low': 'Low Price'
}

# Convert to numeric
for col in features.keys():
    aal_df[col] = pd.to_numeric(aal_df[col], errors='coerce')

aal_df.dropna(subset=features.keys(), how='all', inplace=True)

# Calculate sums
feature_sums = {
    features[col]: aal_df[col].sum(skipna=True) for col in features
}

# Sort and get top 5
top_5 = dict(sorted(feature_sums.items(), key=lambda x: x[1], reverse=True)[:5])

# --- Plot using log scale ---
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5.keys(), top_5.values(), color='#4DB6AC', edgecolor='black')
plt.yscale('log')
plt.title('Top 5 Features for AAL (Log Scale)', fontsize=14)
plt.ylabel('Total Value (Log Scale)', fontsize=12)
plt.xticks(rotation=30, ha='right')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:,.0f}', 
             ha='center', va='bottom', fontsize=9)

plt.grid(True, which="both", linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()


# In[ ]:




