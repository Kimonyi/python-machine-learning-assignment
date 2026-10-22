from sklearn.cluster import KMeans
import numpy as np

# Sample data: Customer features (e.g., Annual Income, Spending Score)
X = np.array([[15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76]])

# Initialize and fit the model (Group into 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Predict cluster for a new data point
new_customer = np.array([[16, 50]])
cluster = kmeans.predict(new_customer)
print(f"Unsupervised ML - K-Means Clustering:")
print(f"The new customer belongs to cluster: {cluster[0]}")