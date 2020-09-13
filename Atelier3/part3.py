import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('/Users/HP/Desktop/S4/Machine Learning/Dataset/CC GENERAL.csv',index_col='CUST_ID') 
df.drop_duplicates(inplace=True)
# using only Spending_Score and income variable for easy visualisation 
X = df.iloc[:, [2, 3]].values 
# Using the elbow method to find the optimal number of clusters 
import skfuzzy as fuzzy
wcss = [] 
for i in range(1, 11): 
    cmeans = fuzzy.cluster.cmeans( X, i , 2, error=0.005, maxiter=1000, init=None) 
    wcss.append(cmeans)

# #plot
# plt.figure(figsize=(10,5)) 
# sns.lineplot(range(1, 11), wcss,marker='o',color='red') 
# plt.title('The Elbow Method') 
# plt.xlabel('Number of clusters') 
# plt.ylabel('WCSS') 
# plt.show()

# Fitting K-Means to the dataset 
cmeans = fuzzy.cluster.cmeans( X, i , 2, error=0.005, maxiter=1000, init=None) 
# y_cmeans = cmeans.fit_predict(X)

# Visualising the clusters 
plt.figure(figsize=(15,7)) 
sns.scatterplot(X[cmeans == 0, 0], X[cmeans == 0, 1], index ='CUST_ID', color = 'yellow', label = 'Cluster 1',s=50) 
sns.scatterplot(X[cmeans == 1, 0], X[cmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50) 
sns.scatterplot(X[cmeans == 2, 0], X[cmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50) 
#sns.scatterplot(cmeans.cluster_centers_[:, 0], cmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=300,marker=',') 
plt.grid(False) 
plt.title('Clusters of customers') 
plt.legend() 
plt.show()
