import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# Шаг 1: подготовка данных
X = df[numeric_features]

# Шаг 2: масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 3: UMAP для понижения размерности
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Шаг 4: кластеризация на UMAP-пространстве
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_umap)

# Шаг 5: добавляем к DataFrame
df['cluster'] = clusters
df['umap_x'] = X_umap[:, 0]
df['umap_y'] = X_umap[:, 1]

# Шаг 6: визуализация
plt.figure(figsize=(10, 6))
plt.scatter(df['umap_x'], df['umap_y'], c=df['cluster'], cmap='Spectral', s=30)
plt.title('UMAP + KMeans Clustering')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.colorbar(label='Cluster')
plt.show()
