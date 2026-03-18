# models/clustering.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class MidClusteringModel:

    def __init__(self, n_clusters=10):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.kmeans.fit(X_pca)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def predict(self, X):
        Z = self.transform(X)
        cluster = self.kmeans.predict(Z)

        # 简单 confidence（距离）
        dist = np.min(self.kmeans.transform(Z), axis=1)

        return cluster, dist

    def visualize(self, X, clusters, save_path=None):
        """
        可视化聚类结果，使用PCA降维后的前两个主成分
        """
        Z = self.transform(X)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(Z[:, 0], Z[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Clustering Visualization (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()