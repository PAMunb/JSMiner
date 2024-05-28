import sys
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

# Carregue os dados
df = pd.read_csv('~/Documents/JSMiner/scripts/results-without-gaps.csv')

df = df.drop(columns=['revision', 'errors','async_declarations_files','await_declarations_files','const_declarations_files','class_declarations_files',
'arrow_function_declarations_files','let_declarations_files','export_declarations_files','yield_declarations_files',
'import_statements_files','promise_declarations_files','promise_all_and_then_files','default_parameters_files',
'rest_statements_files','spread_arguments_files','array_destructuring_files','object_destructuring_files',
'optional_chain_files','template_string_expressions_files','object_properties_files','null_coalesce_operators_files',
'regular_expressions_files','hashbang_comments_files','exponentiation_assignments_files','private_fields_files',
'numeric_separator_files','big_int_files','computed_property_files'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

last_revision_idx = df.groupby(['project'])['date'].idxmax()

df_last_revision = df.loc[last_revision_idx]

# Defina as variáveis de interesse
id_vars = ["project", "date", "statements", "files"]
value_name = "total"
var_name = "feature"

# Derreta o DataFrame para o formato apropriado
melted_df = pd.melt(df_last_revision, id_vars=id_vars, value_name=value_name, var_name=var_name)

# Converta a coluna 'date' para datetime
melted_df['date'] = melted_df['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))

# Converta a coluna 'value' para um tipo numérico
melted_df['total'] = pd.to_numeric(melted_df['total'], errors='coerce')

melted_df = melted_df.sort_values(by='date')

summary = melted_df.groupby('feature')['total'].agg(['median', 'mean', 'std', 'max', 'min']).reset_index()
print(summary)

tablefmt = 'latex_booktabs'  # Formato LaTeX
colalign = ("l", "r", "r", "r", "r", "r")

# Agora você pode continuar com a criação da tabela LaTeX
table_summary = tabulate(summary, headers='keys', tablefmt=tablefmt, colalign=colalign)

# print(table_summary)
# sys.exit()
# Filter out features with median equal to 0
nonzero_median_summary = summary[summary['median'] != 0]

# Create the boxplot
sns.boxplot(x='feature', y='total', data=melted_df[melted_df['feature'].isin(nonzero_median_summary['feature'])])
# Add labels
plt.xlabel('Features')
plt.ylabel('Total')

# Calculate median by feature for non-zero features
# medians = nonzero_median_summary.set_index('feature')['median']

# # Add median labels to the plot
# for feature, median in medians.items():
#     plt.text(x=feature, y=median, s=round(median, 2), ha='center', va='bottom')

# plt.xticks(rotation=45)

# plt.show()


pd.set_option('display.float_format', '{:.3f}'.format)
melted_df['total'] = melted_df['total'].astype(int)
total_by_feature = melted_df.groupby('feature')['total'].sum().reset_index()

# print(total_by_feature)

total_projects = df['project'].nunique()

print(total_projects)

# Verificar se a feature tem pelo menos uma ocorrência
df_feature_counts = melted_df.groupby('feature')['total'].sum().reset_index()
df_filtered = df_feature_counts[df_feature_counts['total'] > 0]

# Contar a quantidade de projetos com pelo menos uma ocorrência em cada feature
df_project_counts = melted_df[melted_df['total'] > 0].groupby('feature')['project'].nunique().reset_index()
df_project_counts.columns = ['feature', 'projects_with_occurrences']

# print(df_project_counts)
summary_project_features = df_filtered.merge(df_project_counts, on='feature', how='left')

summary_project_features['percentage (%)'] = (summary_project_features['projects_with_occurrences'] / total_projects) * 100
summary_project_features = summary_project_features[['feature', 'percentage (%)']]

# print(summary_project_features)

#first adoption
id_vars = ["project", "date", "statements", "files"]
value_name = "total"
var_name = "feature"

df_first_revision = pd.melt(df, id_vars=id_vars, value_name=value_name, var_name=var_name)

# Converta a coluna 'date' para datetime
df_first_revision['date'] = df_first_revision['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))

# Converta a coluna 'value' para um tipo numérico
df_first_revision['total'] = pd.to_numeric(df_first_revision['total'], errors='coerce')

df_first_revision['year_month'] = df_first_revision['date'].dt.strftime('%Y-%m')

df_first_revision = df_first_revision.sort_values(by='year_month')

df_filtered = df_first_revision[df_first_revision['total'] > 0]

start_adoption = df_filtered.groupby('feature')['year_month'].min().reset_index()

# print(start_adoption)

total_by_feature.set_index('feature', inplace=True)
summary_project_features.set_index('feature', inplace=True)
start_adoption.set_index('feature', inplace=True)

# Mesclar os DataFrames usando a coluna "feature" como índice
merged_df = total_by_feature.merge(summary_project_features, left_index=True, right_index=True)
merged_df = merged_df.merge(start_adoption, left_index=True, right_index=True).reset_index()

# print(merged_df)

# Remover as colunas duplicadas "feature"
# merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Definir a coluna "feature" como índice do DataFrame resultante
merged_df.set_index('feature', inplace=True)

# print(merged_df)

# Renomear o índice para 'Feature'
merged_df.index.name = 'Feature'

# Crie uma lista com os novos nomes das colunas na ordem desejada
table_columns = ['Total of Occurrences (#)', 'Projects Adoption (%)', 'First Occurrence']

features_mapping = {
    'async_declarations': 'Async Declarations',
    'await_declarations': 'Await Declarations',
    'const_declarations': 'Const Declarations',
    'arrow_function_declarations': 'Arrow Function Declarations',
    'let_declarations': 'Let Declarations',
    'export_declarations': 'Export Declarations',
    'import_statements': 'Import Statements',
    'class_declarations': 'Class Declarations',
    'default_parameters': 'Default Parameters',
    'rest_statements': 'Rest Statements',
    'array_destructuring': 'Array Destructuring',
    'promise_declarations': 'Promise Declarations',
    'promise_all_and_then': 'Promise All() and Then()',
    'spread_arguments': 'Spread Arguments',
    'object_destructuring': 'Object Destructuring',
    'yield_declarations': 'Yield Declarations',
    'optional_chain': 'Optional Chain',
    'template_string_expressions': 'Template String Expressions',
    'null_coalesce_operators': 'Null Coalesce Operators',
    'hashbang_comments': 'Hashbang Comments',
    'exponentiation_assignments': 'Exponentiation Assignments',
    'private_fields': 'Private Fields',
    'numeric_separator': 'Numeric Separator',
    'object_properties': 'Enhanced Object Properties',
    'big_int':'BigInt',
    'computed_property':'Computed Property',
    'regular_expressions':'Regular Expression',
}

# Renomear as colunas
merged_df.columns = table_columns

# Renomear os valores na coluna 'Feature' usando o mapeamento
merged_df.index = merged_df.index.map(features_mapping)

merged_df = merged_df.sort_values(by='Projects Adoption (%)',ascending=False)

tablefmt = 'latex_booktabs'  # Formato LaTeX
colalign = ("right", "right", "right", "right")

# Agora você pode continuar com a criação da tabela LaTeX
table = tabulate(merged_df, headers='keys', tablefmt=tablefmt, colalign=colalign)

print(table)
