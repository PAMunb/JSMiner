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
df = pd.read_csv('/home/walterlucas/Documents/JSMiner/scripts/results-without-gaps.csv')

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
    'exponentiation_assignments',
    'private_fields',
    'numeric_separator',
    'object_properties',
    'big_int',
    'computed_property',
    'regular_expressions'
]

for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()

    total_by_month = subset.groupby(['feature', 'year_month'])['total'].sum().reset_index()

    # 1. Calcular a diferença entre os totais de ocorrências em cada mês e o mês anterior para cada feature
    total_by_month['diff'] = total_by_month['total'].diff()

    # 2. Calcular a média dessas diferenças para cada feature
    average_difference_per_feature = total_by_month['diff'].mean()

    # Exibir os resultados
    print(feature,average_difference_per_feature)


