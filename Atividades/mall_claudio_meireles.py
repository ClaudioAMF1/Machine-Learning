# -*- coding: utf-8 -*-
"""mall_Claudio_Meireles.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wWm9XvRLk5QvfALX4-RD8HZ8Mx-yuDzY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Carrega e realiza análise exploratória detalhada dos dados"""
    print("1. Análise Exploratória dos Dados")

    # Carregamento dos dados
    url = 'https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/refs/heads/main/clustering/mall_customers.csv'
    df = pd.read_csv(url)
    print("\nPrimeiras linhas do dataset:")
    print(df.head())

    # Codificando variável categórica
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Análise univariada
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    # Distribuições e boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        # Histogramas
        sns.histplot(data=df, x=col, kde=True, ax=axes[0,i])
        axes[0,i].set_title(f'Distribuição de {col}')

        # Boxplots
        sns.boxplot(data=df, y=col, ax=axes[1,i])
        axes[1,i].set_title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.show()

    # Análise bivariada
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', style='Gender')
    plt.title('Renda vs Score de Gastos por Gênero')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender', style='Gender')
    plt.title('Idade vs Score de Gastos por Gênero')
    plt.tight_layout()
    plt.show()

    # Correlações
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.show()

    return df

def prepare_data(df):
    """Prepara os dados para clusterização com diferentes conjuntos de features"""
    print("\n2. Preparação dos Dados")

    # Diferentes conjuntos de features para análise
    feature_sets = {
        'income_spending': ['Annual Income (k$)', 'Spending Score (1-100)'],
        'all_features': ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    }

    scaled_data = {}
    scaler = StandardScaler()

    for name, features in feature_sets.items():
        X = df[features].values
        X_scaled = scaler.fit_transform(X)
        scaled_data[name] = X_scaled

    return scaled_data, feature_sets

def hierarchical_clustering(X, feature_names):
    """Realiza e visualiza clustering hierárquico"""
    print("\n3. Análise de Clustering Hierárquico")

    # Criando o linkage para o dendrograma
    linkage_matrix = linkage(X, method='ward')

    # Plotando o dendrograma
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Dendrograma do Clustering Hierárquico')
    plt.xlabel('Número da Amostra')
    plt.ylabel('Distância')
    plt.show()

    # Aplicando Clustering Hierárquico
    hc = AgglomerativeClustering(n_clusters=5)
    labels = hc.fit_predict(X)

    # Visualização dos clusters
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Clusters - Hierarchical Clustering')
    plt.colorbar(scatter)
    plt.show()

    return labels

def optimize_kmeans(X):
    """Encontra o número ótimo de clusters usando vários métodos"""
    print("\n4. Otimização do Número de Clusters")

    max_clusters = 10
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plotando resultados
    plt.figure(figsize=(12, 5))

    # Método do cotovelo
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Método do Cotovelo')

    # Silhouette score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Análise Silhouette')

    plt.tight_layout()
    plt.show()

    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k

def apply_clustering(X, feature_names, n_clusters):
    """Aplica diferentes algoritmos de clustering"""
    print("\n5. Aplicando Algoritmos de Clustering")

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    # Visualização dos resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # K-means
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title('Clusters K-means')
    plt.colorbar(scatter1, ax=axes[0])

    # DBSCAN
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])
    axes[1].set_title('Clusters DBSCAN')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.show()

    return kmeans_labels, dbscan_labels, kmeans.cluster_centers_

def analyze_clusters(df, labels, feature_names):
    """Realiza análise detalhada dos clusters"""
    print("\n6. Análise Detalhada dos Clusters")

    df_analysis = df.copy()
    df_analysis['Cluster'] = labels

    # Perfil dos clusters
    for cluster in range(len(set(labels))):
        print(f"\nCluster {cluster}:")
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        print(cluster_data[feature_names].describe())

        # Visualizações por cluster
        plt.figure(figsize=(15, 5))

        for i, feature in enumerate(feature_names):
            plt.subplot(1, len(feature_names), i+1)
            sns.boxplot(data=df_analysis, x='Cluster', y=feature)
            plt.title(f'{feature} por Cluster')

        plt.tight_layout()
        plt.show()

    # Análise de distribuição por gênero
    plt.figure(figsize=(10, 5))
    cross_tab = pd.crosstab(df_analysis['Cluster'], df['Gender'])
    cross_tab.plot(kind='bar', stacked=True)
    plt.title('Distribuição de Gênero por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.legend(title='Gênero')
    plt.tight_layout()
    plt.show()

def perform_pca_analysis(X, feature_names):
    """Realiza análise de componentes principais"""
    print("\n7. Análise de Componentes Principais (PCA)")

    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Variância explicada
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Análise de Variância Explicada')
    plt.grid(True)
    plt.show()

    # Contribuição das features
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_names
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
    plt.title('Contribuição das Features para os Componentes Principais')
    plt.show()

    return X_pca, loadings

def main():
    # 1. Carregamento e EDA
    df = load_and_explore_data()

    # 2. Preparação dos dados
    scaled_data, feature_sets = prepare_data(df)

    # 3. Análise para cada conjunto de features
    for name, X in scaled_data.items():
        print(f"\nAnálise para {name}:")
        features = feature_sets[name]

        # 3.1 Clustering Hierárquico
        hier_labels = hierarchical_clustering(X, features)

        # 3.2 Otimização K-means
        optimal_k = optimize_kmeans(X)
        print(f"Número ótimo de clusters: {optimal_k}")

        # 3.3 Aplicação dos algoritmos
        kmeans_labels, dbscan_labels, centroids = apply_clustering(X, features, optimal_k)

        # 3.4 Análise dos clusters
        analyze_clusters(df, kmeans_labels, features)

        # 3.5 PCA
        X_pca, loadings = perform_pca_analysis(X, features)

if __name__ == "__main__":
    main()