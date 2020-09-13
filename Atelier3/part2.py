import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('/Users/HP/Desktop/S4/Machine Learning/Dataset/CC GENERAL.csv',index_col='CUST_ID') 
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
# using only Spending_Score and income variable for easy visualisation 
X = df.iloc[:, [0, 13]].values 
# Using the elbow method to find the optimal number of clusters 
from sklearn.cluster import KMeans 
wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) 
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)

#plot
plt.figure(figsize=(10,5)) 
sns.lineplot(range(1, 11), wcss,marker='o',color='red') 
plt.title('The Elbow Method') 
plt.xlabel('Number of clusters') 
plt.ylabel('WCSS') 
plt.show()

# Fitting K-Means to the dataset 
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42) 
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters 
plt.figure(figsize=(15,7)) 
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50) 
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50) 
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50) 
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=300,marker=',') 
plt.grid(False) 
plt.title('Clusters of customers') 
plt.legend() 
plt.show()
