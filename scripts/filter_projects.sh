#!/bin/bash

# Arquivo de entrada
input_file="results.csv"
# Arquivo de saída
output_file="filtered-results.csv"

# Lista de nomes de projetos a serem removidos
# projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|cesium|limesurvey|countly-server|erp5|releases-comm-central"


projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|cesium|limesurvey|countly-server|erp5|releases-comm-central|copenhagenjs.dk|synapsewebclient|compromise|openstreetbrowser|nightwatch|jsbin|tal|wirecloud|todomvc|stylus|framework|tilemill|katello|windshaft-cartodb|node-restify|rainwave|encoded|nw.js|ably-js|svg.js|windshaft|underscore|prose|async|knex|wulin_master|vimium|webplotdigitizer|etherpad-lite|pyret-lang|recurly-js|ourumbraco|profiler|edirom-online|webappdirac"

# Filtra as linhas que não correspondem ao padrão
grep -Ev "^($projects)" "$input_file" > "$output_file"

echo "Linhas filtradas foram salvas em $output_file"