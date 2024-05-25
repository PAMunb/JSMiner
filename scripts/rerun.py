import pandas as pd

# Lista de strings para filtrar na coluna 'name'
filter_names = [
    "copenhagenjs.dk", "synapsewebclient", "compromise", "openstreetbrowser", "nightwatch",
    "jsbin", "tal", "wirecloud", "todomvc", "stylus", "framework",
    "tilemill", "katello", "windshaft-cartodb", "node-restify",
    "rainwave", "encoded", "nw.js", "ably-js", "svg.js", "windshaft", "underscore",
    "prose", "async", "knex", "wulin_master", "vimium", "webplotdigitizer", "etherpad-lite",
    "pyret-lang", "recurly-js", "ourumbraco", "profiler", "edirom-online", "webappdirac"
]

# copenhagenjs.dk
# browserify#
# synapsewebclient
# compromise
# openstreetbrowser
# nightwatch
# canjs#
# biwascheme#
# jsbin
# tal
# opendsa#
# wirecloud
# todomvc
# stylus
# framework
# zotero#
# tilemill
# katello
# connect-api-examples#
# windshaft-cartodb
# node-restify
# rainwave
# encoded
# nw.js
# ably-js
# svg.js
# windshaft
# jw-community#
# underscore
# paper.js#
# prose
# async
# knex
# wulin_master
# vimium
# webplotdigitizer
# etherpad-lite
# pyret-lang
# one#
# recurly-js
# ourumbraco
# profiler
# edirom-online
# webappdirac

# Carregar o arquivo CSV
df = pd.read_csv('javascript-repositories.csv')

# Filtrar as linhas onde a coluna 'name' contém qualquer uma das strings na lista
filtered_df = df[df['name'].apply(lambda x: any(name in x for name in filter_names))]

# Salvar o novo CSV com as linhas filtradas
filtered_df.to_csv('re-run.csv', index=False)