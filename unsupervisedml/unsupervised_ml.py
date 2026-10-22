import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Prepare the customer data using a Python list (Spending, Visits)
    customer_data = [
        [2000, 3],    # Customer 1
        [2500, 5],    # Customer 2
        [1500, 2],    # Customer 3
        [3000, 4],    # Customer 4
        [1800, 3],    # Customer 5
        [2200, 4],    # Customer 6
        [1900, 2],    # Customer 7
        [2700, 5],    # Customer 8
        [2100, 3],    # Customer 9
        [3000, 4]     # Customer 10
    ]

    # Convert list to a NumPy array for processing
    X = np.array(customer_data)

    # 2. Use StandardScaler to scale the spending and visits features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Create a K-Means model using n_clusters=3, random_state=42, and n_init=10
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    # 4. Train the model using the customer dataset
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # 5. Display the cluster assigned to each customer
    print("--- Customer Assignments ---")
    for i, cluster in enumerate(labels):
        print(f"Customer {i+1} belongs to Cluster {cluster}")

    # 6. Display the total number of customers in each cluster
    print("\n--- Cluster Summary Totals ---")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} customer(s)")

    # ==================== BONUS TASK: MATPLOTLIB GRAPH ====================
    print("\nGenerating customer segmentation graph... Close the window to finish.")
    
    # Create the scatter plot using original unscaled data coordinates for realistic axis labels
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, edgecolors='black')
    
    # Add titles and axis labels
    plt.title('Customer Segmentation via K-Means Clustering')
    plt.xlabel('Monthly Spending (KES)')
    plt.ylabel('Monthly Visits')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Generate custom descriptive text legend from your clusters
    plt.legend(*scatter.legend_elements(), title="Clusters")
    
    # Show the plot window
    plt.show()

if __name__ == "__main__":
    main()
