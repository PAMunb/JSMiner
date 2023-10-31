import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import Word2Vec,FastText
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar o NLTK
nltk.download('punkt')
nltk.download('stopwords')

script_path = os.path.dirname(os.path.abspath(__file__))

# Função para extrair o texto do arquivo README.md de uma URL
def extract_readme_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extrair o texto da classe 'id=readme'
        readme_element = soup.find('div', {"id": "readme"})
        readme_text = ""
        if readme_element:
            readme_text = readme_element.get_text()
        
        # Extrair o texto da primeira ocorrência da classe 'BorderGrid-cell'
        border_grid_element = soup.find('div', {"class": "BorderGrid-cell"})
        border_grid_text = ""
        if border_grid_element:
            border_grid_text = border_grid_element.get_text()
        
        # Combina o texto do README e da BorderGrid-cell
        combined_text = readme_text + " " + border_grid_text
        
        # Remove links usando expressão regular
        combined_text = re.sub(r'http\S+', '', combined_text)
        
        return combined_text.strip()  # Retorna o texto limpo
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
        return None

# Função para pré-processar o texto
def preprocess_text(text):
    # Remover caracteres especiais e números
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Converter para letras minúsculas
    text = text.lower()
    # Tokenizar o texto
    words = word_tokenize(text)
    # Remover stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Carregar a lista de repositórios para treinamento (projects.csv)
train_df = pd.read_csv(script_path+'/projects.csv')
# Carregar a lista de repositórios a serem classificados (javascript-repositories.csv)
classify_df = pd.read_csv(script_path+'/javascript-repositories.csv')

# Lista de URLs dos repositórios para treinamento
train_repo_urls = train_df['url'].tolist()
# Lista de projetos para treinamento
train_projects = train_df['repository'].tolist()

# Lista de URLs dos repositórios a serem classificados
classify_repo_urls = classify_df['url'].tolist()
# Lista de projetos a serem classificados
classify_projects = classify_df['repository'].tolist()

# Remover os repositórios de treinamento que também estão na lista de classificação
filtered_train_repo_urls = [url for url, project in zip(train_repo_urls, train_projects) if project not in classify_projects]

if os.path.isfile(script_path+"/readme_lda_model"):
    # Se o arquivo do modelo existir, carregue-o
    train_lda_model = models.LdaModel.load(script_path+"/readme_lda_model")

    train_dictionary = train_lda_model.id2word

    train_corpus = corpora.MmCorpus(script_path+'/train_corpus.mm')

else:
    # Inicializar a lista de documentos de treinamento
    train_documents = []

    # Extrair os documentos de treinamento
    for url in filtered_train_repo_urls:
        readme_text = extract_readme_text(url)
        if readme_text:
            preprocessed_text = preprocess_text(readme_text)
            train_documents.append(preprocessed_text)

    # Criar o dicionário e o corpus de treinamento
    train_dictionary = corpora.Dictionary(train_documents)
    train_corpus = [train_dictionary.doc2bow(doc) for doc in train_documents]

    # Treinar o modelo LDA com base na base de treinamento
    train_lda_model = models.LdaModel(train_corpus, num_topics=8, id2word=train_dictionary, passes=40)

    train_lda_model.save(script_path+"/readme_lda_model")
    
    corpus_path = script_path+'/train_corpus.mm'
    # Salve o corpus no formato Matrix Market
    corpora.MmCorpus.serialize(corpus_path, train_corpus)




# Inicializar a lista de documentos a serem classificados
classify_documents = []

# Extrair os documentos a serem classificados
for url in classify_repo_urls:
    readme_text = extract_readme_text(url)
    if readme_text:
        preprocessed_text = preprocess_text(readme_text)
        classify_documents.append(preprocessed_text)

# Criar o dicionário e o corpus para a classificação
classify_dictionary = corpora.Dictionary(classify_documents)
classify_corpus = [classify_dictionary.doc2bow(doc) for doc in classify_documents]

# Classificar os documentos com base no modelo LDA treinado
classify_topic_assignments = []
for doc_bow in classify_corpus:
    topic_distribution = train_lda_model[doc_bow]
    dominant_topic = max(topic_distribution, key=lambda item: item[1])[0]
    classify_topic_assignments.append(dominant_topic)

# Definir rótulos para os tópicos
topic_labels = {
    0: "Plugin",
    1: "Module",
    2: "Extension",
    3: "API",
    4: "Database",
    5: "Framework",
    6: "Library",
    7: "Application"
}

# Atribuir categorias aos repositórios classificados
classify_repo_categories = [topic_labels[topic] for topic in classify_topic_assignments]

# Imprimir categorias dos repositórios classificados
for url, category in zip(classify_repo_urls, classify_repo_categories):
    print(f"Repository: {url}")
    print(f"Category: {category}")
    print()


# Visualizar os tópicos
vis_data = gensimvis.prepare(train_lda_model, train_corpus, train_dictionary, sort_topics=False)
pyLDAvis.show(vis_data, local=False)