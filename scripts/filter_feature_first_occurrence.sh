#!/bin/bash

# Verifica se o número correto de argumentos foi fornecido
if [ "$#" -ne 3 ]; then
    echo "Uso: $0 <string_data> <numero_da_coluna> <valor_na_coluna>"
    exit 1
fi

#Numero das colunas no csv
# 1: project
# 2: date
# 3: revision
# 4: files
# 5: async_declarations
# 6: await_declarations
# 7: const_declarations
# 8: class_declarations
# 9: arrow_function_declarations
# 10: let_declarations
# 11: export_declarations
# 12: yield_declarations
# 13: import_statements
# 14: promise_declarations
# 15: promise_all_and_then
# 16: default_parameters
# 17: rest_statements
# 18: spread_arguments
# 19: array_destructuring
# 20: object_destructuring
# 21: optional_chain
# 22: template_string_expressions
# 23: object_properties
# 24: null_coalesce_operators
# 25: regular_expressions
# 26: hashbang_comments
# 27: exponentiation_assignments
# 28: private_fields
# 29: numeric_separator
# 30: big_int
# 31: computed_property
# 32: async_declarations_files
# 33: await_declarations_files
# 34: const_declarations_files
# 35: class_declarations_files
# 36: arrow_function_declarations_files
# 37: let_declarations_files
# 38: export_declarations_files
# 39: yield_declarations_files
# 40: import_statements_files
# 41: promise_declarations_files
# 42: promise_all_and_then_files
# 43: default_parameters_files
# 44: rest_statements_files
# 45: spread_arguments_files
# 46: array_destructuring_files
# 47: object_destructuring_files
# 48: optional_chain_files
# 49: template_string_expressions_files
# 50: object_properties_files
# 51: null_coalesce_operators_files
# 52: regular_expressions_files
# 53: hashbang_comments_files
# 54: exponentiation_assignments_files
# 55: private_fields_files
# 56: numeric_separator_files
# 57: big_int_files
# 58: computed_property_files
# 59: errors
# 60: statements

# Atribui os argumentos a variáveis
STRING_DATA=$1
NUMERO_COLUNA=$2
VALOR_COLUNA=$3

# Executa o comando awk com os argumentos fornecidos
# Exemplo sh filter_feature_first_occurrence.sh 01-2012 24 1

awk -F, -v date="$STRING_DATA" -v col="$NUMERO_COLUNA" -v val="$VALOR_COLUNA" '
BEGIN {OFS = FS}
NR == 1 {print; next}
$2 ~ date && $col == val' filtered-results.csv > feature_first_occurrences.csv
