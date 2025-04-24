
# coding: utf-8

# Importing all relevant python libraries for this project

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading banknote data file

# In[19]:


raw_data = pd.read_csv('Banknote_authentication_dataset.csv')
#V1 is variance
variance = raw_data['V1']
#V2 is skewness
skewness = raw_data['V2']


# Calculating min, max, mean, and standard deviation for each column

# In[20]:


minimum = np.min(raw_data, 0)
maximum = np.max(raw_data, 0)
means = np.mean(raw_data, 0)
std_devs = np.std(raw_data, 0)
std_devs


# Basic visualisation 

# In[29]:


plt.scatter(variance, skewness)

plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.title('Variance vs Skewness of Banknotes')


# KMeans Clustering

# In[26]:


from sklearn.cluster import KMeans


# In[27]:


V1_V2 = np.column_stack((variance,skewness))
km_result = KMeans(n_clusters=2).fit(V1_V2)


# Plotting cluster centres

# In[30]:


clusters = km_result.cluster_centers_

plt.scatter(variance, skewness)


plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.title('Variance vs Skewness of Banknotes')

plt.scatter(clusters[:,0], clusters[:,1],s=100)


# Checking if the algorithm is stable
# 
# K-means is sensitive to initial centroid positions, so running it multiple times and checking if the results are similar is a good way to assess stability.
# If the cluster centers are similar across runs, the algorithm is stable.

# In[31]:


# Run K-means multiple times and compare cluster centers
n_runs = 10
cluster_centers_list = []

for _ in range(n_runs):
    km_result = KMeans(n_clusters=2, n_init=1).fit(V1_V2)
    cluster_centers_list.append(km_result.cluster_centers_)

# Compare cluster centers across runs
for i, centers in enumerate(cluster_centers_list):
    print(f"Run {i+1} Cluster Centers:\n{centers}\n")


# Scikit-learn's KMeans has an n_init parameter (default: 10) that automatically runs the algorithm multiple times with different initializations and picks the best result (lowest inertia). By using _init > 1, the results are already somewhat stabilized.

# In[33]:


km_result = KMeans(n_clusters=2, n_init=10).fit(V1_V2)  # Default is n_init=10
clusters = km_result.cluster_centers_
clusters


# # Final clusters

# In[43]:


# Apply K-means
km_result = KMeans(n_clusters=2, n_init=10).fit(V1_V2)
labels = km_result.labels_  # Cluster assignments (0 or 1)
clusters = km_result.cluster_centers_  # Cluster centers

# Create scatter plot 
scatter = plt.scatter(variance, skewness, c=labels, cmap='viridis', alpha=0.5, label='Data points')
plt.scatter(clusters[:, 0], clusters[:, 1], s=200, c='red', marker='X', label='Cluster centers')

plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.title('K-means Clustering (2 Clusters) of Banknotes')
plt.legend()

plt.show()

