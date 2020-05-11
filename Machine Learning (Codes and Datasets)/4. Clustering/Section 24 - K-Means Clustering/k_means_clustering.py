# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

# Visualising the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0,1],s = 100,label = 'Cluster 1', color='red')
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1,1],s = 100,label = 'Cluster 2', color='blue')
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2,1],s = 100,label = 'Cluster 3', color='green')
plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3,1],s = 100,label = 'Cluster 4', color='yellow')
plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4,1],s = 100,label = 'Cluster 5', color='magenta')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, color = 'black', label = 'Centroid')
plt.title("Cluster of Customers")
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.show()