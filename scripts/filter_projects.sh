#!/bin/bash

# Arquivo de entrada
input_file="results.csv"
# Arquivo de saída
output_file="filtered-results.csv"

# Lista de nomes de projetos a serem removidos
projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|jw-community|cesium|limesurvey|countly-server|erp5|releases-comm-central"

# Filtra as linhas que não correspondem ao padrão
grep -Ev "^($projects)" "$input_file" > "$output_file"

echo "Linhas filtradas foram salvas em $output_file"