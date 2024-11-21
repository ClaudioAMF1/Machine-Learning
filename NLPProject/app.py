from flask import Flask, request, jsonify
from utils.preprocessing import TextPreprocessor
from utils.training import SentimentClassifier
import os

app = Flask(__name__)

# Inicializar preprocessador e classificador
preprocessor = TextPreprocessor()

# Carregar modelo se existir
if os.path.exists('models/vectorizer.pkl') and os.path.exists('models/classifier.pkl'):
    classifier = SentimentClassifier.load_model('models/vectorizer.pkl', 'models/classifier.pkl')
    print("Modelo carregado com sucesso!")
else:
    print("Modelo não encontrado. Execute train_model.py primeiro!")
    classifier = None


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'Nenhum texto fornecido'}), 400

        # Pré-processar texto
        processed_text = preprocessor.preprocess(text)

        # Fazer previsão
        if classifier:
            # Fazer previsão e obter probabilidades
            probabilities = classifier.predict_proba([processed_text])[0]
            prediction = classifier.predict([processed_text])[0]

            # Mapear resultado
            sentiment = 'positivo' if prediction == 1 else 'negativo'
            confidence = float(max(probabilities))

            return jsonify({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'processed_text': processed_text
            })
        else:
            return jsonify({'error': 'Modelo não carregado'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/vectorize', methods=['POST'])
def vectorize():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'Nenhum texto fornecido'}), 400

        # Pré-processar texto
        processed_text = preprocessor.preprocess(text)

        if classifier:
            # Vetorizar texto usando TF-IDF
            vector = classifier.vectorizer.transform([processed_text])

            # Criar dicionário com termos e seus valores TF-IDF
            feature_names = classifier.vectorizer.get_feature_names_out()
            vector_dict = {
                term: float(value)
                for term, value in zip(feature_names, vector.toarray()[0])
                if value > 0
            }

            return jsonify({
                'text': text,
                'processed_text': processed_text,
                'vector': vector_dict
            })
        else:
            return jsonify({'error': 'Modelo não carregado'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)