import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('/home/walterlucas/Documents/JSMiner/scripts/results-without-gaps.csv')

df = df.drop(columns=['revision', 'errors'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df['year_month'] = df['date'].dt.strftime('%Y-%m')

# Use a função groupby para agrupar o DataFrame por projeto e ano/mês
grouped = df.groupby(['project', 'year_month'])

# Encontre o índice da última revisão em cada grupo
last_revision_idx = grouped['date'].idxmax()

df_last_revision = df.loc[last_revision_idx]

# Defina as variáveis de interesse
id_vars = ["project", "date", "statements", "files", "year_month"]
value_name = "total"
var_name = "feature"

# Derreta o DataFrame para o formato apropriado
melted_df = pd.melt(df_last_revision, id_vars=id_vars, value_name=value_name, var_name=var_name)

# Converta a coluna 'date' para datetime
melted_df['date'] = melted_df['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))

melted_df['date'] = melted_df['year_month'].apply(lambda x: pd.to_datetime(x, format='%Y-%m', errors='coerce'))
# Converta a coluna 'total' para um tipo numérico
melted_df['total'] = pd.to_numeric(melted_df['total'], errors='coerce')

melted_df = melted_df.sort_values(by='year_month')

# Lista de recursos (features)
features = [
    'async_declarations',
    'await_declarations',
    'const_declarations',
    'class_declarations',
    'arrow_function_declarations',
    'let_declarations',
    'export_declarations',
    'yield_declarations',
    'import_statements',
    'promise_declarations',
    'promise_all_and_then',
    'default_parameters',
    'rest_statements',
    'spread_arguments',
    'array_destructuring',
    'object_destructuring'
]

# Etapa 1: Visualização de Dados
plt.figure(figsize=(12, 6))
for feature in features:
    subset = melted_df[melted_df['feature'] == feature]
    plt.plot(subset['date'], subset['total'], label=feature)
plt.title('Séries Temporais dos Recursos')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Etapa 2: Decomposição
for feature in features:
    subset = melted_df[melted_df['feature'] == feature]
    subset = subset.set_index('date')
    decomposition = seasonal_decompose(subset['total'], model='additive', period=12)  # Pode ajustar o período conforme necessário
    print(f'Decomposição de {feature}:')
    print(decomposition.trend.head())  # Imprime os primeiros valores da tendência
    print(decomposition.seasonal.head())  # Imprime os primeiros valores da sazonalidade
    print(decomposition.resid.head())  # Imprime os primeiros valores dos resíduos
    decomposition.plot()
    plt.title(f'Decomposição de {feature}')
    plt.show()

# Etapa 3: Autocorrelação
for feature in features:
    subset = melted_df[melted_df['feature'] == feature]
    subset = subset.set_index('date')
    print(f'Funções de Autocorrelação de {feature}:')
    plot_acf(subset['total'], lags=50)
    plot_pacf(subset['total'], lags=50)
    plt.title(f'Funções de Autocorrelação de {feature}')
    plt.show()

# Etapa 4: Diferenciação
for feature in features:
    subset = melted_df[melted_df['feature'] == feature]
    subset = subset.set_index('date')
    subset_diff = subset['total'].diff().dropna()
    print(f'Série Temporal Diferenciada de {feature}:')
    print(subset_diff.head())  # Imprime os primeiros valores da série diferenciada
    plt.plot(subset_diff)
    plt.title(f'Série Temporal Diferenciada de {feature}')
    plt.show()

# Etapa 5: Teste de Dickey-Fuller Aumentado (ADF) para Estacionariedade
for feature in features:
    subset = melted_df[melted_df['feature'] == feature]
    subset = subset.set_index('date')
    result = adfuller(subset['total'])
    print(f'Teste ADF para {feature}:')
    print('Estatística ADF:', result[0])
    print('Valor-p:', result[1])
    print('Valores Críticos:', result[4])
    print('Resultados do Teste:')
    if result[1] <= 0.05:
        print('A série é estacionária (rejeita a hipótese nula)')
    else:
        print('A série não é estacionária (falha em rejeitar a hipótese nula)')
    print('-' * 40)