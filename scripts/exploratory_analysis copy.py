import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.stats import kendalltau
import os

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

# print(melted_df)

output_folder = './decomposition'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


plt.figure(figsize=(12, 6))

# Etapa 1: Visualização de Dados
# for feature in features:

#     subset = melted_df[melted_df['feature'] == feature].copy()
    
#     total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

#     subset = subset.merge(total_by_month, on='date', how='left')
#     subset = subset.sort_values(by='date')

#     X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
#     y = total_by_month['total']  # Use a coluna 'total' como variável dependente

#     # Aplicar a raiz quadrada a 'total'
#     total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
#     total_by_month = total_by_month.sort_values(by='date')

#     print(total_by_month)
    
#     plt.plot(total_by_month['date'], total_by_month['sqrt_total'], label=feature)

# plt.title('Séries Temporais dos Recursos')
# plt.xlabel('Data')
# plt.ylabel('Valor')
# plt.legend()
# plt.show()

# Etapa 2: Decomposição
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')

    X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
    y = total_by_month['total']  # Use a coluna 'total' como variável dependente

    # Aplicar a raiz quadrada a 'total'
    total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    decomposition = seasonal_decompose(total_by_month['sqrt_total'], model='additive', period=12)  # Pode ajustar o período conforme necessário
    print(f'Decomposição de {feature}:')
    print(decomposition.trend.head())  # Imprime os primeiros valores da tendência
    print(decomposition.seasonal.head())  # Imprime os primeiros valores da sazonalidade
    print(decomposition.resid.head())  # Imprime os primeiros valores dos resíduos
    # decomposition.plot()
    # plt.title(f'Decomposição de {feature}')
    # plt.show()
    # Crie o nome do arquivo PDF com base na feature
    pdf_filename = os.path.join(output_folder, f'decomposition_{feature}.pdf')
    
    # Plote a decomposição
    decomposition.plot()
    plt.title(f'Decomposição de {feature}')
    plt.savefig(pdf_filename, format='pdf')


# # Etapa 3: Autocorrelação
# for feature in features:
#     subset = melted_df[melted_df['feature'] == feature].copy()
    
#     total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

#     subset = subset.merge(total_by_month, on='date', how='left')
#     subset = subset.sort_values(by='date')

#     X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
#     y = total_by_month['total']  # Use a coluna 'total' como variável dependente

#     # Aplicar a raiz quadrada a 'total'
#     total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
#     total_by_month = total_by_month.set_index('date')
#     print(f'Funções de Autocorrelação de {feature}:')
#     plot_acf(total_by_month['sqrt_total'], lags=50)
#     plot_pacf(total_by_month['sqrt_total'], lags=50)
#     plt.title(f'Funções de Autocorrelação de {feature}')
#     plt.show()

# # Etapa 4: Diferenciação
# for feature in features:
#     subset = melted_df[melted_df['feature'] == feature].copy()
    
#     total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

#     subset = subset.merge(total_by_month, on='date', how='left')
#     subset = subset.sort_values(by='date')

#     X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
#     y = total_by_month['total']  # Use a coluna 'total' como variável dependente

#     # Aplicar a raiz quadrada a 'total'
#     total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
#     total_by_month = total_by_month.set_index('date')
#     subset_diff = total_by_month['sqrt_total'].diff().dropna()
#     print(f'Série Temporal Diferenciada de {feature}:')
#     print(subset_diff.head())  # Imprime os primeiros valores da série diferenciada
#     plt.plot(subset_diff)
#     plt.title(f'Série Temporal Diferenciada de {feature}')
#     plt.show()

# # Etapa 5: Teste de Dickey-Fuller Aumentado (ADF) para Estacionariedade
# for feature in features:
#     subset = melted_df[melted_df['feature'] == feature].copy()
    
#     total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

#     subset = subset.merge(total_by_month, on='date', how='left')
#     subset = subset.sort_values(by='date')

#     X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
#     y = total_by_month['total']  # Use a coluna 'total' como variável dependente

#     # Aplicar a raiz quadrada a 'total'
#     total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
#     total_by_month = total_by_month.set_index('date')
#     result = adfuller(total_by_month['sqrt_total'])
#     print(f'Teste ADF para {feature}:')
#     print('Estatística ADF:', result[0])
#     print('Valor-p:', result[1])
#     print('Valores Críticos:', result[4])
#     print('Resultados do Teste:')
#     if result[1] <= 0.05:
#         print('A série é estacionária (rejeita a hipótese nula)')
#     else:
#         print('A série não é estacionária (falha em rejeitar a hipótese nula)')
#     print('-' * 40)


# Etapa 6: Teste de coeficiente de Kendall e o valor-p para avaliar se a tendência é crescente, decrescente ou não existe tendência na serie
for feature in features:
    subset = melted_df[melted_df['feature'] == feature].copy()
    
    total_by_month = subset.groupby(['feature', 'date'])['total'].sum().reset_index()

    subset = subset.merge(total_by_month, on='date', how='left')
    subset = subset.sort_values(by='date')

    X = sm.add_constant(total_by_month['date'].index)  # Use o índice como variável independente
    y = total_by_month['total']  # Use a coluna 'total' como variável dependente

    # Aplicar a raiz quadrada a 'total'
    total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])
    total_by_month = total_by_month.set_index('date')
    # Calcule o coeficiente de Kendall e o valor-p
    coeficiente_kendall, valor_p = kendalltau(total_by_month.index, total_by_month['sqrt_total'])

    # Verifique se o valor-p é menor que o nível de significância (por exemplo, 0.05)
    nivel_de_significancia = 0.05
    # if valor_p < nivel_de_significancia:
    #     print('A série temporal possui uma tendência (crescente ou decrescente).')
    # else:
    #     print('A série temporal não possui uma tendência significativa.')

    # Imprima o coeficiente de Kendall para verificar a direção da tendência
    print('Coeficiente de Kendall:', coeficiente_kendall)
    print(f'Valor-p: {valor_p:.25f}')
    print('Nível de significância:', nivel_de_significancia)

    if coeficiente_kendall > 0:
        print(f'Tendência Crescente na Série Temporal para {feature}.')
    elif coeficiente_kendall < 0:
        print(f'Tendência Decrescente na Série Temporal para {feature}.')
    else:
        print(f'Não Existe uma Tendência Significativa na Série Temporal para {feature}.')