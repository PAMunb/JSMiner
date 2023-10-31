import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import Word2Vec
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import re
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

# Carregar o arquivo CSV com os rótulos
labels_df = pd.read_csv(script_path + '/projects_labels.csv')

# Unir as categorias com os projetos de treinamento usando os nomes dos repositórios
train_df = pd.merge(train_df, labels_df, on='repository')

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['category'], test_size=0.2, random_state=42)

# Criar um vetorizador TF-IDF para converter o texto em recursos numéricos
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treinar um classificador Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test_tfidf)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')

# Agora, você pode usar o modelo treinado para classificar novos projetos com base no texto do README.
# Para fazer isso, siga o mesmo processo de pré-processamento e vetorização do texto e, em seguida, use clf.predict() para obter a categoria prevista.

# Exemplo de classificação de um novo projeto
new_project_text = "This is the README text of a new project. It is a framework for web development."
new_project_text = preprocess_text(new_project_text)
new_project_text = ' '.join(new_project_text)  # Converter de volta para texto
new_project_tfidf = tfidf_vectorizer.transform([new_project_text])
predicted_category = clf.predict(new_project_tfidf)[0]
print(f'Categoria prevista para o novo projeto: {predicted_category}')