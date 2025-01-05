#!/bin/bash

export JAVA_HOME=/home/walterlucas/.sdkman/candidates/java/11.0.23-ms/
export PATH=$JAVA_HOME/bin:$PATH

# Defina o limite de memória em gigabytes (27GB neste caso)
MEMORY_LIMIT_GB=30

# Caminho para o diretório contendo os subdiretórios
DATASET_DIRECTORY="/home/walterlucas/JSMiner/dataset"

# Obtenha o uso atual de memória RAM
MEMORY_USAGE=$(free -g | awk '/^Mem/{print $3}')

# Conte o número de subdiretórios no diretório dataset
SUBDIRECTORY_COUNT=$(ls "$DATASET_DIRECTORY" | wc -l)
echo "Quantidade de REPOSITORIOS: $SUBDIRECTORY_COUNT"

# Função para obter o PID do processo Java
get_java_pid() {
    ps ax | grep '[j]ava -jar -Xmx26g /home/walterlucas/JSMiner/target/JSMiner-1.0.5-SNAPSHOT.jar' | awk '{print $1}'
}

# Verifique se há um processo Java em execução
JAVA_PID=$(get_java_pid)
echo "Valor de JAVA_PID: $JAVA_PID"
if [ -z "$JAVA_PID" ]; then
    
    # Execute os scripts necessários
    sh /home/walterlucas/JSMiner/removator.sh
    
    # Verifique se não há subdiretórios para decidir se desliga a máquina
    if [ "$SUBDIRECTORY_COUNT" -eq 0 ]; then
        echo "Nenhum subdiretório encontrado. Desligando a máquina."
        sudo shutdown -h now
        exit 0  # Sai do script após o desligamento
    fi

    echo "Nenhum processo Java encontrado. Iniciando o programa Java."
    
    nohup java -jar -Xmx26g /home/walterlucas/JSMiner/target/JSMiner-1.0.5-SNAPSHOT.jar -d /home/walterlucas/JSMiner/dataset/ -s 30 -ft 1 >/home/walterlucas/JSMiner/nohup.out 2>&1 &
    
    sleep 5  # Aguarde um pouco para garantir que o processo foi iniciado
else
    echo "Processo Java já está em execução. PID: $JAVA_PID"
fi

# Verifique se o uso de memória ultrapassou o limite ou se não há subdiretórios
if [ "$MEMORY_USAGE" -ge "$MEMORY_LIMIT_GB" ] || [ "$SUBDIRECTORY_COUNT" -eq 0 ]; then
    echo "Condições atendidas para parar o programa Java. Memória: $MEMORY_USAGE GB, Subdiretórios: $SUBDIRECTORY_COUNT."

    # Obter o PID do processo Java novamente
    JAVA_PID=$(get_java_pid)

    # Pare a execução do programa Java
    if [ -n "$JAVA_PID" ]; then
        echo "PID do processo Java: $JAVA_PID"
        kill -9 "$JAVA_PID"
    else
        echo "Processo Java não encontrado."
    fi

    # Verifique se não há subdiretórios para decidir se desliga a máquina
    if [ "$SUBDIRECTORY_COUNT" -eq 0 ]; then
        echo "Nenhum subdiretório encontrado. Desligando a máquina."
        sudo shutdown -h now
    else
        echo "Executando os scripts necessários."

        # Execute os scripts necessários
        sh /home/walterlucas/JSMiner/removator.sh

        # Reinicie a execução do programa Java
        nohup java -jar -Xmx26g /home/walterlucas/JSMiner/target/JSMiner-1.0.5-SNAPSHOT.jar -d /home/walterlucas/JSMiner/dataset/ -s 30 -ft 1 >/home/walterlucas/JSMiner/nohup.out 2>&1 &
    fi
else
    echo "Uso de memória dentro dos limites e há subdiretórios em $DATASET_DIRECTORY. Não é necessário fazer nada."
fi
