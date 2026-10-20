from sklearn.cluster import KMeans

# prepare Students study and task data
x = [[1,2],[2,1],[2,3],[5,6],[6,5],[6,7],[9,9],[10,8],[9,10]]

#create  the model
model = KMeans (n_clusters=3)

#training the model
model.fit(x)

#predict the clusters
groups = model.predict(x)

print("Clusters:", groups)
for i in range(len(x)):
    print("Students", i + 1, "belongs to cluster", groups[i] + 1)
