# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('beratbadan.csv')
X = dataset.iloc[:, [2, 3]].values
 
# Menjalankan K-Means Clustering ke dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter=100, n_init=10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
dataset["Class"] = y_kmeans
print(dataset) 
 
# Visualisasi hasil clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters nama')
plt.xlabel('tinggi badan)')
plt.ylabel('berat badan')
plt.legend()
plt.show()
