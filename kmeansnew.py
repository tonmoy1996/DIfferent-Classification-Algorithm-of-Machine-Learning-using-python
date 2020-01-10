import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

data=pd.read_csv('MALLDATA.csv')

x=data.iloc[:,[3,4]].values

wccs=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wccs.append(kmeans.inertia_)
    
plt.plot(range(1,11),wccs)
plt.title("The ELbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wccs Value")
plt.show()

# applying k-means to mall cluster
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

#visualize the cluster and mall data 

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,cmap='red',label = 'Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,cmap='green',label = 'Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,cmap='blue',label = 'Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,cmap='purple',label = 'Cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,cmap='cyan',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()










