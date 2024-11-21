from utils.preprocessing import TextPreprocessor
from utils.training import SentimentClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


def main():
    # Criar diretórios necessários
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/imdb', exist_ok=True)

    # Carregar e preparar dados
    print("Carregando dados...")
    # Dataset de exemplo mais robusto e balanceado
    data = pd.DataFrame({
        'text': [
            # Reviews positivos
            "This movie was excellent! Great performance by all actors.",
            "Amazing film, the best I've seen this year!",
            "Incredible storyline and perfect execution.",
            "A masterpiece of modern cinema.",
            "Brilliant performance by the entire cast.",
            "Really enjoyed watching this movie.",
            "Great direction and amazing cinematography.",
            "One of the best films I've ever seen.",
            "Fantastic plot with unexpected twists.",
            "Very entertaining and well made movie.",

            # Reviews negativos
            "Terrible waste of time. Do not watch this movie.",
            "One of the worst films ever made.",
            "Poor acting and terrible script.",
            "Complete disaster, avoid at all costs.",
            "Extremely disappointing and boring.",
            "Waste of money, don't bother watching.",
            "Awful movie with terrible plot.",
            "Really bad acting and direction.",
            "The worst movie I've seen this year.",
            "Completely missed the mark, very poor."
        ],
        'sentiment': [
            # 1 para positivo, 0 para negativo
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    })

    # Pré-processar textos
    print("Pré-processando textos...")
    preprocessor = TextPreprocessor()
    data['processed_text'] = data['text'].apply(preprocessor.preprocess)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'],
        data['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=data['sentiment']  # Garantir distribuição balanceada
    )

    # Treinar modelo
    print("Treinando modelo...")
    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)

    # Avaliar modelo
    print("\nAvaliando modelo...")
    predictions = classifier.predict(X_test)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, predictions, zero_division=1))

    # Salvar modelo
    print("\nSalvando modelo...")
    classifier.save_model(
        'models/vectorizer.pkl',
        'models/classifier.pkl'
    )

    print("Treinamento concluído! Modelo salvo em /models/")

    # Teste com algumas frases novas
    print("\nTestando modelo com algumas frases novas:")
    test_phrases = [
        "This was a great movie, I really enjoyed it!",
        "Terrible film, complete waste of time.",
        "Pretty decent movie with some good moments."
    ]

    for phrase in test_phrases:
        processed = preprocessor.preprocess(phrase)
        prediction = classifier.predict([processed])[0]
        proba = classifier.predict_proba([processed])[0]
        sentiment = "positivo" if prediction == 1 else "negativo"
        confidence = max(proba)
        print(f"\nFrase: {phrase}")
        print(f"Sentimento: {sentiment}")
        print(f"Confiança: {confidence:.2f}")


if __name__ == "__main__":
    main()