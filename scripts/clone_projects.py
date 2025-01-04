import csv
import git
import os
import stat
import sys
import subprocess

# python3 runner.py /path/to/directory/ /path/to/file.csv

# git config --global core.longpaths true

cwd = sys.argv[1]
clear = []
count = 1
with open(sys.argv[2], newline='', encoding='latin-1') as f:
    projects = csv.reader(f, delimiter=',')
    for project in projects:
        if project[1] == "name":
            continue
        
        print(count)
        projectName = project[1].split('/')
        path = os.path.join(cwd, projectName[1].strip())  # Melhor usar os.path.join
        url = "https://github.com/" + project[1] + ".git"
        
        print(path, url)
        try:
            # Verifica se o diretório do repositório existe e se é um repositório Git válido
            if os.path.isdir(path):
                if os.path.isdir(os.path.join(path, '.git')):  # Verifica se é um repositório Git
                    print(f"Repositório {projectName[1].strip()} já está clonado.")
                    
                    # Captura o nome da branch principal usando git symbolic-ref
                    result = subprocess.run(
                        ['git', '-C', path, 'symbolic-ref', 'refs/remotes/origin/HEAD'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    if result.returncode == 0:
                        # Pega o nome da branch principal (ex: refs/remotes/origin/main)
                        main_branch = result.stdout.strip().split('/')[-1]
                        print(f"Branch principal identificada: {main_branch}")
                        
                        # Reseta e faz checkout para a branch principal
                        print(f"Resetando e fazendo checkout para a branch {main_branch}...")
                        subprocess.run(['git', '-C', path, 'reset', '--hard'], check=True)
                        subprocess.run(['git', '-C', path, 'checkout', main_branch], check=True)
                    else:
                        print(f"Erro ao identificar a branch principal para o repositório {projectName[1].strip()}.")
                    continue  # Se já estiver clonado e resetado, passa para o próximo projeto
                else:
                    # Caso contrário, remova o diretório e clone novamente
                    print(f"Removendo diretório inválido para: {projectName[1].strip()}")
                    os.chmod(path, stat.S_IWUSR)  # Torna o diretório gravável
                    os.system(f"rmdir /s /q {path}")  # Remove o diretório inválido
                    print(f"Clonando novamente: {projectName[1].strip()}")
                    git.Git(cwd).clone(url.strip())
            else:
                # Se o diretório não existir, clone o repositório normalmente
                print(f"Clonando: {projectName[1].strip()}")
                git.Git(cwd).clone(url.strip())

            count += 1
        except UnicodeDecodeError as e:
            print(f"Erro ao decodificar o projeto: {project[1]}")
            print(e)
            clear.append(projectName[1].strip())
            if os.path.isdir(path):
                os.chmod(path, stat.S_IWUSR)
                os.system(f"rmdir /s /q {path}")
        except Exception as e:
            print(f"Erro ao clonar o projeto: {project[1]}")
            print(e)
            clear.append(projectName[1].strip())
            if os.path.isdir(path):
                os.chmod(path, stat.S_IWUSR)
                os.system(f"rmdir /s /q {path}")

    print(clear)
