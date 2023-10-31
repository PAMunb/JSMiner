import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

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
# Converta a coluna 'value' para um tipo numérico
melted_df['total'] = pd.to_numeric(melted_df['total'], errors='coerce')

melted_df = melted_df.sort_values(by='year_month')

# Lista de recursos (features)
# features = [
#     'async_declarations',
#     'const_declarations',
#     'arrow_function_declarations',
#     'let_declarations',
#     'object_destructuring'
# ]

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

plt.figure(figsize=(16, 8))

# Configurações do gráfico com seaborn
sns.set(style="whitegrid", font_scale=1.2)

# Iterar sobre os recursos e ajustar modelos de regressão e gerar gráficos
for feature in features:
    df_feature = melted_df[melted_df['feature'] == feature].copy()
    total_by_month = df_feature.groupby(['year_month'])['total'].sum().reset_index()

    X = sm.add_constant(total_by_month.index)  # Use o índice como variável independente
    y = total_by_month['total']  # Use a coluna 'total' como variável dependente

    # Aplicar a raiz quadrada a 'total'
    total_by_month['sqrt_total'] = np.sqrt(total_by_month['total'])

    # Criar as variáveis X e y com as colunas transformadas
    X_sqrt = sm.add_constant(total_by_month.index)
    y_sqrt = total_by_month['sqrt_total']
    
    # Cálculo da suavização loess
    loess_result = lowess(total_by_month['sqrt_total'], total_by_month.index, frac=0.1)
    total_by_month['loess'] = loess_result[:, 1]

    # Encontrar o primeiro ponto de inclinação significativa
    window_size = 12  # Tamanho da janela deslizante
    slopes = []
    for i in range(len(total_by_month) - window_size + 1):
        x_window = np.arange(i, i + window_size)  # Corrigir o eixo x
        y_window = total_by_month['loess'].iloc[i:i + window_size]

        # Ajustar um modelo de regressão linear
        slope, _ = np.polyfit(x_window, y_window, 1)
        slopes.append(slope)

    # Encontrar o primeiro ponto de inclinação significativa
    idx = np.argmax(slopes)
    start_point = (total_by_month['year_month'].iloc[idx], total_by_month['loess'].iloc[idx])

   # Calcular a tendência suavizada (loess) normalizada
    # sns.lineplot(data=total_by_month, x='year_month', y='loess', label=f'{feature.replace("_", " ").title()}', errorbar=None, estimator=None, lw=2)
    sns.lineplot(data=total_by_month, x='year_month', y='sqrt_total', label=f'{feature.replace("_", " ").title()} - Total Occurrences', errorbar=None, estimator=None, lw=2)

    # Marcar o primeiro ponto de inclinação significativa
    plt.plot(start_point[0], start_point[1], 'ro')

# Configurações do gráfico
plt.title('Smoothed Trends of Different Features with Start Point')
plt.xlabel('Date (Year)')
plt.ylabel('Total Occurrences (#)')
plt.xticks(rotation=45)

x_ticks = np.arange(0, len(total_by_month), 12)  # Por exemplo, mostra um ponto a cada 12 meses
plt.xticks(x_ticks, total_by_month['year_month'].iloc[x_ticks].apply(lambda x: x[:4]), rotation=45)  # Exibe apenas o ano


plt.legend(fontsize='small')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# Exibir o gráfico
plt.savefig('all_features_trends_with_start.pdf', bbox_inches='tight')  # Use bbox_inches='tight' para evitar que a legenda seja cortada
plt.show()
