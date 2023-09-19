import pandas as pd

# Carregue o DataFrame original
df_original = pd.read_csv('/home/walterlucas/Documents/JSMiner/scripts/results.csv', delimiter=',')

# Defina a coluna 'date' como data
df_original['date'] = pd.to_datetime(df_original['date'], format='%d-%m-%Y')  # Use '%d-%m-%Y' em vez de '%Y-%m-%d'
# df_original['date'] = df_original['date'].dt.strftime('%d-%m-%Y')
# Crie um DataFrame final vazio para concatenar os resultados de cada projeto
df_final = pd.DataFrame(columns=df_original.columns)

# Lista de projetos únicos
projetos_unicos = df_original['project'].unique()

# Crie um DataFrame de referência para o projeto atual com todas as datas no intervalo desejado
date_range = pd.date_range(start="2012-01-01", end="2023-06-30", freq="M")
date_range_str = date_range.strftime('%d-%m-%Y')
df_referencia = pd.DataFrame({'date': date_range_str})

# Itere por cada projeto
for projeto in projetos_unicos:
    df_projeto = df_original[df_original['project'] == projeto]

    # Converta a coluna 'date' do DataFrame de referência para o mesmo tipo de dados
    df_referencia['date'] = pd.to_datetime(df_referencia['date'], format='%d-%m-%Y')
    df_referencia = df_referencia.sort_values(by='date')

    # Mescle o DataFrame de referência com o DataFrame do projeto, preenchendo com zero
    df_projeto = df_referencia.merge(df_projeto, on='date', how='outer').fillna(0)
    df_projeto.iloc[:, 2:] = df_projeto.iloc[:, 2:].fillna(0)

    # Inicialize as variáveis para rastrear a linha anterior
    linha_anterior = df_projeto.iloc[0].copy()
    # print(linha_anterior)    
    # Itere pelas linhas do DataFrame do projeto (começando da segunda linha)
    for index, row in df_projeto.sort_values(by='date').iterrows():
        if str(row['revision']) != str(linha_anterior['revision']) and str(linha_anterior['revision']) != '0' and str(row['revision']) == '0':
            # Copie os valores da terceira coluna em diante da linha anterior para a linha atual
            df_projeto.iloc[index, 2:] = linha_anterior[2:]
        # Atualize a linha anterior
        linha_anterior = df_projeto.iloc[index].copy()

    df_projeto['project'] = projeto
    df_projeto = df_projeto.sort_values(by='date')

    df_final = pd.concat([df_final, df_projeto], ignore_index=True)

df_final.to_csv('results-without-gaps.csv', index=False)