#!/bin/bash

# Define a variável com a lista de projetos, separada por '|'
projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|cesium|limesurvey|countly-server|erp5|releases-comm-central|copenhagenjs.dk|compromise|openstreetbrowser|nightwatch|jsbin|tal|wirecloud|todomvc|framework|tilemill|windshaft-cartodb|node-restify|rainwave|ably-js|svg.js|windshaft|underscore|prose|async|knex|vimium|webplotdigitizer|pyret-lang|recurly-js|ourumbraco|profiler"

# Caminho do arquivo CSV de entrada
input_file="javascript-repositories.csv"

# Caminho do arquivo CSV de saída para os projetos removidos
removed_output_file="removed_projects.csv"

# Caminho do arquivo CSV de saída para os projetos restantes
remaining_output_file="remaining_projects.csv"

# Filtra o arquivo CSV, removendo as linhas que correspondem aos projetos da lista e as salvando em um arquivo separado
awk -F, -v projects="$projects" '
BEGIN {
    # Cria um padrão de expressão regular a partir da lista de projetos
    split(projects, project_arr, "|")
    for (i in project_arr) {
        project_list[project_arr[i]]
    }
}
NR==1 {
    # Armazena o cabeçalho
    print $0 > "'"$remaining_output_file"'"
    print $0 > "'"$removed_output_file"'"
}
NR>1 {
    # Extrai o nome do projeto após o caractere "/"
    split($2, name_parts, "/")
    project_name = name_parts[2]

    # Verifica se o nome do projeto está na lista de projetos
    if (project_name in project_list) {
        print $0 > "'"$removed_output_file"'"
    } else {
        print $0 > "'"$remaining_output_file"'"
    }
}
' "$input_file"