#!/bin/bash

# Defina o limite de memória em gigabytes (27GB neste caso)
MEMORY_LIMIT_GB=27

# Defina o limite de arquivos CSV
CSV_FILE_LIMIT=40

# Caminho para o diretório contendo os arquivos .csv
CSV_DIRECTORY="../jsminer-out"

# Obtenha o uso atual de memória RAM
MEMORY_USAGE=$(free -g | awk '/^Mem/{print $3}')

# Conte o número de arquivos .csv no diretório
CSV_FILE_COUNT=$(find "$CSV_DIRECTORY" -type f -name "*.csv" | wc -l)

# Verifique se o uso de memória ultrapassou o limite ou se há 40 ou mais arquivos .csv
if [ "$MEMORY_USAGE" -ge "$MEMORY_LIMIT_GB" ] || [ "$CSV_FILE_COUNT" -ge "$CSV_FILE_LIMIT" ]; then
    echo "Condições atendidas para parar o programa Java. Memória: $MEMORY_USAGE GB, Arquivos CSV: $CSV_FILE_COUNT."

    # Pare a execução do programa Java (substitua 'nome_do_programa.jar' pelo nome do seu arquivo .jar)
    pkill -f "java -jar -Xmx26g target/JSMiner-1.0-SNAPSHOT.jar -d dataset/ -s 30 -ft 1"
    # Verifique se há 40 ou mais arquivos .csv para decidir se desliga a máquina
    if [ "$CSV_FILE_COUNT" -ge "$CSV_FILE_LIMIT" ]; then
        echo "Número de arquivos CSV é $CSV_FILE_COUNT. Desligando a máquina."
        sudo shutdown -h now
    else
        echo "Executando os scripts necessários."

        # Execute os scripts necessários
        ./removator.sh

        # Reinicie a execução do programa Java
        nohup java -jar -Xmx26g target/JSMiner-1.0-SNAPSHOT.jar -d dataset/ -s 30 -ft 1 >/dev/null 2>&1 &
    fi
else
    echo "Uso de memória dentro dos limites e menos de $CSV_FILE_LIMIT arquivos CSV. Não é necessário fazer nada."
fi

