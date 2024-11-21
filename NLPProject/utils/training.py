from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np


class SentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()

    def train(self, X, y):
        """
        Treina o modelo usando TF-IDF e Naive Bayes
        """
        # Vetorização TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X)

        # Treinar o classificador
        self.classifier.fit(X_tfidf, y)

    def predict(self, texts):
        """
        Realiza previsões para novos textos
        """
        # Vetorizar os textos
        X_tfidf = self.vectorizer.transform(texts)

        # Fazer previsões
        return self.classifier.predict(X_tfidf)

    def predict_proba(self, texts):
        """
        Retorna as probabilidades das previsões
        """
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_tfidf)

    def save_model(self, vectorizer_path, classifier_path):
        """
        Salva o modelo treinado
        """
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.classifier, classifier_path)

    @classmethod
    def load_model(cls, vectorizer_path, classifier_path):
        """
        Carrega um modelo previamente treinado
        """
        instance = cls()
        instance.vectorizer = joblib.load(vectorizer_path)
        instance.classifier = joblib.load(classifier_path)
        return instance