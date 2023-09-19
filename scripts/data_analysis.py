import pandas as pd
from tabulate import tabulate

# Carregue os dados
df = pd.read_csv('/home/walterlucas/Documents/JSMiner/scripts/results-without-gaps.csv')

df = df.drop(columns=['revision', 'errors'])
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

# print(summary)

total_by_feature = melted_df.groupby('feature')['total'].sum().reset_index()

# print(total_by_feature)

df_filtered = melted_df[melted_df['total'] > 0]

# Calcular a porcentagem para cada elemento único em 'feature'
summary_project_features = df_filtered.groupby('feature').apply(lambda x: (100 * x['project'].nunique()) / 100.0).reset_index()
summary_project_features.columns = ['feature', 'percentage (%)']

# print(summary_project_features)

#first adoption
id_vars = ["project", "date", "statements", "files"]
value_name = "total"
var_name = "feature"

# Derreta o DataFrame para o formato apropriado
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
# Mesclar os DataFrames usando a coluna "feature" como índice
merged_df = pd.concat([total_by_feature, summary_project_features, start_adoption], axis=1)

# Remover as colunas duplicadas "feature"
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Definir a coluna "feature" como índice do DataFrame resultante
merged_df.set_index('feature', inplace=True)

# Renomear o índice para 'Feature'
merged_df.index.name = 'Feature'

# Crie uma lista com os novos nomes das colunas na ordem desejada
table_columns = ['Total of Occurrences (#)', 'Projects Adoption (%)', 'First Occurrence (%Y~%m)']

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
    'yield_declarations': 'Yield Declarations'
}

# Renomear as colunas
merged_df.columns = table_columns

# Renomear os valores na coluna 'Feature' usando o mapeamento
merged_df.index = merged_df.index.map(features_mapping)


tablefmt = 'latex_booktabs'  # Formato LaTeX
colalign = ("right", "right", "right", "right")

# Agora você pode continuar com a criação da tabela LaTeX
table = tabulate(merged_df, headers='keys', tablefmt=tablefmt, colalign=colalign)

print(table)
