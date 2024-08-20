import sys
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Carregue os dados
df = pd.read_csv('~/Documents/JSMiner/scripts/results-without-gaps.csv')

# Filtrar as colunas de interesse
features_columns = [
'async_declarations_files','await_declarations_files','const_declarations_files','class_declarations_files','arrow_function_declarations_files','let_declarations_files','export_declarations_files','yield_declarations_files','import_statements_files','default_parameters_files','rest_statements_files','spread_arguments_files','array_destructuring_files','object_destructuring_files','optional_chain_files','template_string_expressions_files','null_coalesce_operators_files','exponentiation_assignments_files','private_fields_files', 'numeric_separator_files','big_int_files','enhanced_property_assignment_files','computed_property_assignment_files','function_property_declaration_files'
]

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

last_revision_idx = df.groupby(['project'])['date'].idxmax()

df_last_revision = df.loc[last_revision_idx]

# print(df)
# sys.exit()

# Calcular a porcentagem de arquivos com ocorrências para cada revisão
for feature in features_columns:
    df_last_revision[feature + '_percentage'] = (df_last_revision[feature] / df['files']) * 100
    

# Defina as variáveis de interesse
id_vars = ["project", "date", "revision", "statements", "files"]
value_name = "total"
var_name = "feature"

df_summary = df_last_revision.drop(columns=['async_declarations','await_declarations','const_declarations','class_declarations',
'arrow_function_declarations','let_declarations','export_declarations','yield_declarations',
'import_statements','default_parameters',
'rest_statements','spread_arguments','array_destructuring','object_destructuring',
'optional_chain','template_string_expressions','null_coalesce_operators','exponentiation_assignments','private_fields',
'numeric_separator','big_int','enhanced_property_assignment','computed_property_assignment','function_property_declaration'])

df_summary = df_summary.drop(columns=['errors','async_declarations_files','await_declarations_files','const_declarations_files','class_declarations_files',
'arrow_function_declarations_files','let_declarations_files','export_declarations_files','yield_declarations_files',
'import_statements_files','default_parameters_files',
'rest_statements_files','spread_arguments_files','array_destructuring_files','object_destructuring_files',
'optional_chain_files','template_string_expressions_files','null_coalesce_operators_files','exponentiation_assignments_files','private_fields_files',
'numeric_separator_files','big_int_files','enhanced_property_assignment_files','computed_property_assignment_files','function_property_declaration_files'])

df_summary.to_csv('last_revision_files_occurrences_percentage.csv', index=False)

# Derreta o DataFrame para o formato apropriado
melted_df = pd.melt(df_summary, id_vars=id_vars, value_name=value_name, var_name=var_name)

# Converta a coluna 'date' para datetime
melted_df['date'] = melted_df['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))

# Converta a coluna 'value' para um tipo numérico
melted_df['total'] = pd.to_numeric(melted_df['total'], errors='coerce')

melted_df = melted_df.sort_values(by='date')

summary = melted_df.groupby('feature')['total'].agg(['median', 'mean', 'std', 'max', 'min']).reset_index()
# print(summary)
# exit()
# Calcular a média das porcentagens para cada feature por projeto
# project_feature_means = df.groupby('project')[[feature + '_percentage' for feature in features_columns]].mean().reset_index()

# print(project_feature_means)
# sys.exit()

# Remover a coluna 'project' ao calcular a média geral
mean_feature_usage = df_last_revision[[feature + '_percentage' for feature in features_columns]].mean().reset_index()
mean_feature_usage.columns = ['feature', 'mean_percentage']

# print(mean_feature_usage)
# sys.exit()

# Mapear os nomes das features para uma forma mais legível
features_mapping = {
    'async_declarations_files_percentage': 'Async Declarations',
    'await_declarations_files_percentage': 'Await Operators',
    'const_declarations_files_percentage': 'Const Declarations',
    'arrow_function_declarations_files_percentage': 'Arrow Function Declarations',
    'let_declarations_files_percentage': 'Let Declarations',
    'export_declarations_files_percentage': 'Export Declarations',
    'import_statements_files_percentage': 'Import Statements',
    'class_declarations_files_percentage': 'Class Declarations',
    'default_parameters_files_percentage': 'Default Parameters',
    'rest_statements_files_percentage': 'Rest Statements',
    'array_destructuring_files_percentage': 'Array Destructuring',
    'spread_arguments_files_percentage': 'Spread Arguments',
    'object_destructuring_files_percentage': 'Object Destructuring',
    'yield_declarations_files_percentage': 'Yield Operators',
    'optional_chain_files_percentage': 'Optional Chain',
    'template_string_expressions_files_percentage': 'Template String Expressions',
    'null_coalesce_operators_files_percentage': 'Null Coalesce Operators',
    # 'exponentiation_assignments_files_percentage': 'Exponentiation Assignments',
    'private_fields_files_percentage': 'Private Fields',
    'numeric_separator_files_percentage': 'Numeric Separator',
    'big_int_files_percentage':'BigInt',
    'enhanced_property_assignment_files_percentage' : 'Enhanced Property Assignment',
    'computed_property_assignment_files_percentage' : 'Computed Property Assignment',
    'function_property_declaration_files_percentage' : 'Function Property Declarations'
}

mean_feature_usage['feature'] = mean_feature_usage['feature'].map(features_mapping)

# print(mean_feature_usage)
# sys.exit()

# Criar a tabela LaTeX
tablefmt = 'latex_booktabs'  # Formato LaTeX
colalign = ("right", "right")

table = tabulate(mean_feature_usage.dropna().sort_values(by='mean_percentage', ascending=False), 
                 headers=['Feature', 'Mean Usage (%)'], 
                 tablefmt=tablefmt, 
                 colalign=colalign)

print(table)

# Criar o gráfico boxplot
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_feature_usage.sort_values(by='mean_percentage', ascending=False), x='mean_percentage', y='feature', palette='viridis')
plt.title('Mean Usage of Features')
plt.xlabel('Mean Percentage (%)')
plt.ylabel('Feature')
plt.grid(axis='x')
plt.tight_layout()

# Ajustar layout
plt.tight_layout()

# Salvar o gráfico em um arquivo PDF
with PdfPages('features_percentage_adoption_files_barplot.pdf') as pdf:
    pdf.savefig()
    
# Mostrar o gráfico
# plt.show()