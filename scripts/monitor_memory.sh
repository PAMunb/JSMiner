#!/bin/bash

export JAVA_HOME=/home/walterlucas/.sdkman/candidates/java/11.0.23-ms/
export PATH=$JAVA_HOME/bin:$PATH

# Defina o limite de memória em gigabytes (27GB neste caso)
MEMORY_LIMIT_GB=27

# Defina o limite de arquivos CSV
CSV_FILE_LIMIT=234

# Caminho para o diretório contendo os arquivos .csv
CSV_DIRECTORY="/home/walterlucas/jsminer-out"

# Obtenha o uso atual de memória RAM
MEMORY_USAGE=$(free -g | awk '/^Mem/{print $3}')

# Conte o número de arquivos .csv no diretório
CSV_FILE_COUNT=$(find "$CSV_DIRECTORY" -type f -name "*.csv" | wc -l)

# Função para obter o PID do processo Java
get_java_pid() {
    ps ax | grep '[j]ava -jar -Xmx26g /home/walterlucas/JSMiner/target/JSMiner-1.0.4-SNAPSHOT.jar' | awk '{print $1}'
}

# Verifique se o uso de memória ultrapassou o limite ou se há 234 ou mais arquivos .csv
if [ "$MEMORY_USAGE" -ge "$MEMORY_LIMIT_GB" ] || [ "$CSV_FILE_COUNT" -ge "$CSV_FILE_LIMIT" ]; then
    echo "Condições atendidas para parar o programa Java. Memória: $MEMORY_USAGE GB, Arquivos CSV: $CSV_FILE_COUNT."

    # Obter o PID do processo Java
    JAVA_PID=$(get_java_pid)

    # Pare a execução do programa Java
    if [ -n "$JAVA_PID" ]; then
        echo "PID do processo Java: $JAVA_PID"
        kill -9 "$JAVA_PID"
    else
        echo "Processo Java não encontrado."
    fi

    # Verifique se há 234 ou mais arquivos .csv para decidir se desliga a máquina
    if [ "$CSV_FILE_COUNT" -ge "$CSV_FILE_LIMIT" ]; then
        echo "Número de arquivos CSV é $CSV_FILE_COUNT. Desligando a máquina."
        sudo shutdown -h now
    else
        echo "Executando os scripts necessários."

        # Execute os scripts necessários
        sh /home/walterlucas/JSMiner/removator.sh

        # Reinicie a execução do programa Java
        nohup java -jar -Xmx26g /home/walterlucas/JSMiner/target/JSMiner-1.0.4-SNAPSHOT.jar -d /home/walterlucas/JSMiner/dataset/ -s 30 -ft 3 >/home/walterlucas/JSMiner/nohup.out 2>&1 &
    fi
else
    echo "Uso de memória dentro dos limites e menos de $CSV_FILE_LIMIT arquivos CSV. Não é necessário fazer nada."
fi
