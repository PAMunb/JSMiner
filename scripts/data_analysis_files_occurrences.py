import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Carregue os dados
df = pd.read_csv('~/Documents/JSMiner/scripts/results-without-gaps.csv')

# Filtrar as colunas de interesse
features_columns = [
    'async_declarations_files', 'await_declarations_files', 'const_declarations_files',
    'class_declarations_files', 'arrow_function_declarations_files', 'let_declarations_files',
    'export_declarations_files', 'yield_declarations_files', 'import_statements_files',
    'promise_declarations_files', 'promise_all_and_then_files', 'default_parameters_files',
    'rest_statements_files', 'spread_arguments_files', 'array_destructuring_files',
    'object_destructuring_files', 'optional_chain_files', 'template_string_expressions_files',
    'object_properties_files', 'null_coalesce_operators_files', 'regular_expressions_files',
    'hashbang_comments_files', 'exponentiation_assignments_files', 'private_fields_files',
    'numeric_separator_files', 'big_int_files', 'computed_property_files'
]

# Calcular a porcentagem de arquivos com ocorrências para cada revisão
for feature in features_columns:
    df[feature + '_percentage'] = (df[feature] / df['files']) * 100
    
# print(df)

# Calcular a média das porcentagens para cada feature por projeto
project_feature_means = df.groupby('project')[[feature + '_percentage' for feature in features_columns]].mean().reset_index()

# print(project_feature_means)

# Remover a coluna 'project' ao calcular a média geral
mean_feature_usage = project_feature_means[[feature + '_percentage' for feature in features_columns]].mean().reset_index()
mean_feature_usage.columns = ['feature', 'mean_percentage']

# Mapear os nomes das features para uma forma mais legível
features_mapping = {
    'async_declarations_files_percentage': 'Async Declarations',
    'await_declarations_files_percentage': 'Await Declarations',
    'const_declarations_files_percentage': 'Const Declarations',
    'arrow_function_declarations_files_percentage': 'Arrow Function Declarations',
    'let_declarations_files_percentage': 'Let Declarations',
    'export_declarations_files_percentage': 'Export Declarations',
    'import_statements_files_percentage': 'Import Statements',
    'class_declarations_files_percentage': 'Class Declarations',
    'default_parameters_files_percentage': 'Default Parameters',
    'rest_statements_files_percentage': 'Rest Statements',
    'array_destructuring_files_percentage': 'Array Destructuring',
    'promise_declarations_files_percentage': 'Promise Declarations',
    'promise_all_and_then_files_percentage': 'Promise All() and Then()',
    'spread_arguments_files_percentage': 'Spread Arguments',
    'object_destructuring_files_percentage': 'Object Destructuring',
    'yield_declarations_files_percentage': 'Yield Declarations',
    'optional_chain_files_percentage': 'Optional Chain',
    'template_string_expressions_files_percentage': 'Template String Expressions',
    'null_coalesce_operators_files_percentage': 'Null Coalesce Operators',
    'hashbang_comments_files_percentage': 'Hashbang Comments',
    'exponentiation_assignments_files_percentage': 'Exponentiation Assignments',
    'private_fields_files_percentage': 'Private Fields',
    'numeric_separator_files_percentage': 'Numeric Separator',
    'object_properties_files_percentage': 'Enhanced Object Properties',
    'big_int_files_percentage': 'BigInt',
    'computed_property_files_percentage': 'Computed Property',
    'regular_expressions_files_percentage': 'Regular Expression'
}

mean_feature_usage['feature'] = mean_feature_usage['feature'].map(features_mapping)

print(mean_feature_usage)

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