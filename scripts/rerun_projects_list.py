import csv
import os

# Obtém o caminho absoluto do diretório atual
current_directory = os.getcwd()

# Arquivos de entrada e saída
last_revision_file = 'last_revision_files_occurrences_percentage.csv'
repositories_file = 'javascript-repositories.csv'
output_file = 'rerun_repositories.csv'

# print(current_directory)
# Carregar lista de projetos do primeiro CSV
projects = set()
with open(current_directory+'/'+last_revision_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    # print(reader)
    next(reader)  # Pular o cabeçalho
    for row in reader:
        project = row[0]  # Considerando que o nome do projeto está na primeira coluna
        projects.add(project)

# Processar o segundo CSV e filtrar as linhas com os projetos correspondentes
with open(repositories_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Ler o cabeçalho

    # Criar o arquivo de saída e escrever o cabeçalho
    with open(output_file, mode='w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)

        # Filtrar as linhas
        for row in reader:
            project_name = row[1]  # Nome do repositório está na segunda coluna
            if any(project in project_name for project in projects):
                writer.writerow(row)

# print(f'Filtragem concluída. O arquivo resultante está em "{output_file}".')
