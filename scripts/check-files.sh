#!/bin/bash

# Diretório dos arquivos .csv
CSV_DIR="../../jsminer-out"

# Arquivos de log
LOG_FILE_CC="count_lines_log.txt"
LOG_FILE_CHECKOUT_CONFLICT="checkout_conflict_log.txt"
LOG_FILE_EXCEPTION="exception_log.txt"

# Limpar o arquivo de log se já existir
> "$LOG_FILE_CC"
> "$LOG_FILE_CHECKOUT_CONFLICT"
> "$LOG_FILE_EXCEPTION"

# Contar "Checkout conflict" em arquivos .txt no diretório especificado
for file in "$CSV_DIR"/*.txt; do
    count=$(grep -c "Checkout conflict" "$file")
    echo "$file: $count 'Checkout conflict' lines" >> "$LOG_FILE_CHECKOUT_CONFLICT"
done

# Contar "Exception:" em arquivos .txt no diretório especificado
for file in "$CSV_DIR"/*.txt; do
    count=$(grep -c "Exception:" "$file")
    echo "$file: $count 'Exception:' lines" >> "$LOG_FILE_EXCEPTION"
done

# Contar linhas em arquivos .csv no diretório especificado
for file in "$CSV_DIR"/*.csv; do
    count=$(wc -l < "$file")
    echo "No arquivo $file, encontradas $count linhas" >> "$LOG_FILE_CC"
done

# Mensagem final
echo "Log da execução salvo em $LOG_FILE_CC, $LOG_FILE_EXCEPTION, $LOG_FILE_CHECKOUT_CONFLICT"

# Arquivo de entrada
input_file="collections-failed.txt"

# Extraindo nomes de projetos únicos e imprimindo a lista a ser removida
awk 'NR>3 {print $2}' FS="[', ]+" "$input_file" | sort | uniq

for file in "$CSV_DIR"/*.csv; do
    awk 'FNR==2 {print $0}' "$file"
    awk 'END {print}' "$file"
done > start-end-by-projects.txt