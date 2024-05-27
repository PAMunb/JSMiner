#!/bin/bash

# Arquivo de entrada
input_file="results.csv"
# Arquivo de saída
output_file="filtered-results.csv"

#excluidos que tiveram code-review revisados e com achados interessantes
#cartodb,prey-node-client,qooxdoo,countly-server,cesium**

# Lista de nomes de projetos a serem removidos
# 32 por checkout-conflicts and exceptions
# projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|cesium|limesurvey|countly-server|erp5|releases-comm-central"

projects="bigbluebutton|biwascheme|browserify|canjs|cartodb|connect-api-examples|e2openplugin-openwebif|etherpad-lite|geonode|iteexe|jw-community|mediathread|node-tap|one|opendsa|paper.js|patterns|prey-node-client|qooxdoo|salesforcemobilesdk-shared|sequenceserver|tapjs|www.tjvantoll.com|zotero|acorn|rhino|hue|cesium|limesurvey|countly-server|erp5|releases-comm-central|copenhagenjs.dk|compromise|openstreetbrowser|nightwatch|jsbin|tal|wirecloud|todomvc|framework|tilemill|windshaft-cartodb|node-restify|rainwave|ably-js|svg.js|windshaft|underscore|prose|async|knex|vimium|webplotdigitizer|pyret-lang|recurly-js|ourumbraco|profiler"

# Filtra as linhas que não correspondem ao padrão
grep -Ev "^($projects)" "$input_file" > "$output_file"

echo "Linhas filtradas foram salvas em $output_file"