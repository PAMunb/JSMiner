import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.stats import kendalltau
import os

# Read the CSV file into a DataFrame
df = pd.read_csv('results-without-gaps.csv')

# Drop unnecessary columns
df = df.drop(columns=['revision', 'errors','async_declarations_files','await_declarations_files','const_declarations_files','class_declarations_files',
'arrow_function_declarations_files','let_declarations_files','export_declarations_files','yield_declarations_files',
'import_statements_files','promise_declarations_files','promise_all_and_then_files','default_parameters_files',
'rest_statements_files','spread_arguments_files','array_destructuring_files','object_destructuring_files',
'optional_chain_files','template_string_expressions_files','object_properties_files','null_coalesce_operators_files',
'regular_expressions_files','hashbang_comments_files','exponentiation_assignments_files','private_fields_files',
'numeric_separator_files','big_int_files','computed_property_files'])

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Extract year and month from the 'date' column
df['year_month'] = df['date'].dt.strftime('%Y-%m')

# Group the DataFrame by project and year/month
grouped = df.groupby(['project', 'year_month'])

# Find the index of the last revision in each group
last_revision_idx = grouped['date'].idxmax()

# Get the rows corresponding to the last revision in each group
df_last_revision = df.loc[last_revision_idx]

# Define variables of interest
id_vars = ["project", "date", "statements", "files", "year_month"]
value_name = "total"
var_name = "feature"

# Melt the DataFrame to the appropriate format
melted_df = pd.melt(df_last_revision, id_vars=id_vars, value_name=value_name, var_name=var_name)

# Convert the 'date' column to datetime format
melted_df['date'] = melted_df['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))

# Convert the 'year_month' column to datetime format
melted_df['date'] = melted_df['year_month'].apply(lambda x: pd.to_datetime(x, format='%Y-%m', errors='coerce'))

# Convert the 'total' column to numeric
melted_df['total'] = pd.to_numeric(melted_df['total'], errors='coerce')

# Sort the DataFrame by 'year_month'
melted_df = melted_df.sort_values(by='year_month')

# List of features
features = [
    'async_declarations',
    'await_declarations',
    'const_declarations',
    'arrow_function_declarations',
    'let_declarations',
    'export_declarations',
    'import_statements',
    'class_declarations',
    'default_parameters',
    'rest_statements',
    'array_destructuring',
    'promise_declarations',
    'promise_all_and_then',
    'spread_arguments',
    'object_destructuring',
    'yield_declarations',
    'optional_chain',
    'template_string_expressions',
    'null_coalesce_operators',
    'hashbang_comments',
    # 'exponentiation_assignments',
    'private_fields',
    'numeric_separator',
    'object_properties',
    'big_int',
    'computed_property',
    'regular_expressions'
]

# Output folder for decomposition plots
output_folder = './decomposition'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 2: Decomposition
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()
    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')
    # total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    decomposition = seasonal_decompose(total_by_month['total'], model='additive', period=12)
    print(f'Decomposition of {feature}:')
    print(decomposition.trend.head())  # Print the first values of the trend
    print(decomposition.seasonal.head())  # Print the first values of the seasonal component
    print(decomposition.resid.head())  # Print the first values of the residuals
    pdf_filename = os.path.join(output_folder, f'decomposition_{feature}.pdf')
    decomposition.plot()
    plt.title(f'Decomposition of {feature}')
    plt.savefig(pdf_filename, format='pdf')

# Output folder for autocorrelation plots
output_folder = './auto-correlation'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 3: Autocorrelation
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()
    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')
    # total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    print(f'Autocorrelation Functions of {feature}:')
    plot_acf(total_by_month['total'], lags=50)
    plot_pacf(total_by_month['total'], lags=50)
    plt.title(f'Autocorrelation Functions of {feature}')
    pdf_filename = os.path.join(output_folder, f'auto_correlation_{feature}.pdf')
    plt.savefig(pdf_filename, format='pdf')

# Step 5: Augmented Dickey-Fuller (ADF) Test for Stationarity
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()
    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')
    # total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    result = adfuller(total_by_month['total'])
    print(f'ADF Test for {feature}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Test Results:')
    if result[1] <= 0.05:
        print('The series is stationary (rejects the null hypothesis)')
    else:
        print('The series is non-stationary (fails to reject the null hypothesis)')
    print('-' * 40)

# Step 6: Kendall Coefficient and p-value Test to evaluate trend direction
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()
    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')
    # total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    # Calculate Kendall coefficient and p-value
    coeficient_kendall, p_value = kendalltau(total_by_month['total'], total_by_month.index)
    # Check if the p-value is less than the significance level (e.g., 0.05)
    significance_level = 0.05
    print('Kendall Coefficient:', coeficient_kendall)
    print(f'p-value: {p_value:.25f}')
    print('Significance Level:', significance_level)
    if coeficient_kendall > 0:
        print(f'Increasing Trend in the Time Series for {feature}.')
    elif coeficient_kendall < 0:
        print(f'Decreasing Trend in the Time Series for {feature}.')
    else:
        print(f'No Significant Trend in the Time Series for {feature}.')