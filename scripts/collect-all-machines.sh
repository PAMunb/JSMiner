#!/bin/bash

# Lista de IPs das máquinas
IPS="34.105.55.44 34.174.14.55 35.202.100.142 35.228.223.205"
# IPS="34.174.11.22 35.202.333.444 35.228.567.890"

# Usuário e caminho remoto
USER="walterlucas"
REMOTE_PATH="/home/walterlucas/jsminer-out"

# Caminho local onde os arquivos serão copiados
LOCAL_PATH="."

# Loop para executar o scp para cada IP
for IP in $IPS; do
  echo "Copiando de $USER@$IP:$REMOTE_PATH para $LOCAL_PATH"
  scp -r "$USER@$IP:$REMOTE_PATH" "$LOCAL_PATH"
done

ls jsminer-out | grep .csv | wc -l
