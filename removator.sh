#!/bin/bash

# Pasta onde os diretórios estão localizados
pasta_dataset="dataset"

# Pasta onde os arquivos .csv são verificados
pasta_verificacao="js-miner-out"

# Ler nomes da segunda coluna do arquivo javascript-repositories.csv
cut -d ',' -f 2 scripts/javascript-repositories.csv | tail -n +2 | while IFS= read -r nome; do
    # Verificar se existe um arquivo .csv na pasta de verificação
    if [ -e "$pasta_verificacao/$nome.csv" ]; then
        # Remover o diretório
        rm -r "$pasta_dataset/$nome"
        echo "Diretório $nome removido."
    else
        echo "Diretório $nome não possui um arquivo .csv em $pasta_verificacao. Não será removido."
    fi
done
