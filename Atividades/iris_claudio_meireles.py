# -*- coding: utf-8 -*-
"""iris_Claudio_Meireles.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k3HtJZL7db94vyoRmHI_izpFhG5I9B1R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Carrega e realiza análise exploratória detalhada"""
    print("1. Análise Exploratória de Dados (EDA)")

    # Carregando o dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("\nVisão geral dos dados:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    print("\nVerificando valores ausentes:")
    print(df.isnull().sum())

    # Visualizações
    # 1. Distribuição das classes
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Species')
    plt.title('Distribuição das Espécies de Iris')
    plt.xticks(rotation=45)
    plt.show()

    # 2. Pairplot
    sns.pairplot(df, hue='Species', diag_kind='hist')
    plt.show()

    # 3. Matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()

    # 4. Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, feature in enumerate(iris.feature_names):
        sns.boxplot(data=df, x='Species', y=feature, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_xticklabels(axes[i//2, i%2].get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

    return df

def train_and_evaluate_classifier(name, clf, X_train, X_test, y_train, y_test):
    """Treina e avalia um classificador com validação cruzada"""
    print(f"\nAvaliando {name}...")

    # Validação cruzada
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"CV Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Treino e avaliação
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

    return clf, y_pred, cv_scores

def perform_clustering(X_scaled, y_true):
    """Realiza análise de clusterização"""
    print("\n4. Aplicação dos Algoritmos de Clusterização")

    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Hierarchical Clustering
    hierarch = AgglomerativeClustering(n_clusters=3)
    hierarch_labels = hierarch.fit_predict(X_scaled)

    # Visualização dos resultados
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    # Classes reais
    scatter = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
    axes[0].set_title('Classes Reais')
    plt.colorbar(scatter, ax=axes[0])

    # K-means
    scatter = axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
    axes[1].set_title('Clusters K-means')
    plt.colorbar(scatter, ax=axes[1])

    # DBSCAN
    scatter = axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
    axes[2].set_title('Clusters DBSCAN')
    plt.colorbar(scatter, ax=axes[2])

    # Hierarchical
    scatter = axes[3].scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarch_labels, cmap='viridis')
    axes[3].set_title('Clusters Hierárquicos')
    plt.colorbar(scatter, ax=axes[3])

    plt.tight_layout()
    plt.show()

    # Dendrograma
    plt.figure(figsize=(10, 7))
    linkage_matrix = linkage(X_scaled, method='ward')
    dendrogram(linkage_matrix)
    plt.title('Dendrograma - Clustering Hierárquico')
    plt.show()

    # Avaliação dos clusters
    print("\n5. Avaliação dos Resultados da Clusterização")
    for name, labels in [('K-Means', kmeans_labels),
                        ('DBSCAN', dbscan_labels),
                        ('Hierarchical', hierarch_labels)]:
        if len(set(labels)) > 1:  # Ignora DBSCAN se todos forem ruído
            print(f"Silhouette Score para {name}: {silhouette_score(X_scaled, labels):.3f}")

    return kmeans_labels, dbscan_labels, hierarch_labels

def main():
    # 1. Carregamento e EDA
    df = load_and_explore_data()

    # 2. Preparação dos dados
    X = df.drop('Species', axis=1)
    y = pd.Categorical(df['Species']).codes

    # Split com stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Classificação
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}
    for name, clf in classifiers.items():
        model, preds, cv_scores = train_and_evaluate_classifier(
            name, clf, X_train_scaled, X_test_scaled, y_train, y_test
        )
        results[name] = {
            'model': model,
            'predictions': preds,
            'cv_scores': cv_scores
        }

    # 4. Clusterização
    X_scaled = scaler.fit_transform(X)
    perform_clustering(X_scaled, y)

    # 5. Comparação final dos modelos
    cv_means = pd.DataFrame([
        {
            'Modelo': name,
            'CV Mean Score': scores['cv_scores'].mean(),
            'CV Std': scores['cv_scores'].std()
        }
        for name, scores in results.items()
    ])

    print("\nComparação dos modelos:")
    print(cv_means.sort_values('CV Mean Score', ascending=False))


    print("\nConclusão:")
    print("\n1. Análise dos resultados de classificação:")
    print("   - Todos os modelos apresentaram boa performance, com acurácia entre 89% e 93% no conjunto de teste")
    print("   - A Regressão Logística teve o melhor desempenho na validação cruzada (CV Mean: 98.1%, Std: 0.023)")
    print("   - SVM ficou em segundo lugar (CV Mean: 97.1%, Std: 0.023)")
    print("   - Os outros modelos obtiveram performance similar entre si (CV Mean ~95%)")
    print("\n2. Análise dos padrões de erro:")
    print("   - A classe Setosa (0) foi perfeitamente classificada por todos os modelos")
    print("   - Existe confusão na classificação entre Versicolor (1) e Virginica (2)")
    print("   - A precisão para Versicolor variou entre 0.78-0.92")
    print("   - A precisão para Virginica variou entre 0.82-1.00")

    print("\n3. Resultados da Clusterização:")
    print("   - K-Means obteve o melhor Silhouette Score (0.480)")
    print("   - Hierarchical Clustering ficou em segundo (0.447)")
    print("   - DBSCAN apresentou o menor score (0.357)")
    print("   - Os scores moderados indicam alguma sobreposição entre clusters")

    print("\n4. Comparação Classificação vs Clusterização:")
    print("   - A classificação supervisionada obteve resultados superiores")
    print("   - A clusterização conseguiu identificar grupos, mas com menor precisão")
    print("   - A diferença de performance evidencia a importância dos labels para este problema")

if __name__ == "__main__":
    main()