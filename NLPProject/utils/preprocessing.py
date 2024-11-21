import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# Download todos os recursos necessários
def download_nltk_resources():
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger',
        'punkt_tab',
        'punkt_tabs'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            print(f"Não foi possível baixar {resource}, mas continuando...")


# Baixar recursos
download_nltk_resources()


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Aviso: usando conjunto vazio de stopwords")
            self.stop_words = set()

    def preprocess(self, text):
        """
        Realiza o pré-processamento completo do texto
        """
        # Converter para minúsculas
        text = str(text).lower()

        # Remover HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenização simples (split por espaços)
        tokens = text.split()

        # Remover stopwords e lematização
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]

        # Juntar tokens de volta em uma string
        return ' '.join(tokens)