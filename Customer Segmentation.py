#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import math
import datetime
import scipy
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_excel("Online Retail.xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


# Checking for NUlls

df.isnull().sum()


# In[8]:


# Customers cannot be segmented without the Customer IDs. dropping the records which has CustomerID as NULLs

df.dropna(inplace=True)


# In[9]:


df.shape


# ### Fundamental Analysis

# #### Country Distribution

# In[10]:


plt.figure(figsize=(12,5))
sns.countplot(x=df['Country'], palette='Set1')
plt.xticks(rotation=90,ha='right')
plt.title("Country Distribution")
plt.xlabel('Country')
plt.ylabel('Count');


# #### Year Distribution

# In[11]:


plt.figure(figsize=(8,5))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
sns.countplot(x=df['InvoiceDate'].dt.year,palette= 'Set2')
plt.xticks(rotation=0,ha='right')
plt.title("Year Distribution")
plt.xlabel('Year')
plt.ylabel('Count');


# #### Month Distribution

# In[12]:


plt.figure(figsize=(8,5))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
sns.countplot(x=df['InvoiceDate'].dt.month_name(),palette= 'Spectral')
plt.xticks(rotation=45,ha='right')
plt.title("Month Distribution")
plt.xlabel('Months')
plt.ylabel('Count');


# #### Week Distribution

# In[13]:


plt.figure(figsize=(8,5))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
sns.countplot(x=df['InvoiceDate'].dt.day_name(),palette= 'Set1')
plt.xticks(rotation=0,ha='right')
plt.title("Week Distribution")
plt.xlabel('WeekDays')
plt.ylabel('Count');


# ## Recency, Frequency and Monetary Fields Calculation

# In[14]:


df['Monetary'] = df['Quantity']*df['UnitPrice']
df_m = df.groupby('CustomerID')['Monetary'].sum()
df_m = df_m.reset_index()
df_m = df_m[~(df_m['Monetary']<=0)]
df_m.head()


# In[15]:


df_f = df.groupby('CustomerID')['InvoiceNo'].count()
df_f = df_f.reset_index()
df_f.columns = ['CustomerID', 'Frequency']
df_f.head()


# In[16]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
max_date = max(df['InvoiceDate'])
max_date


# In[17]:


df['diff_days'] = max_date+datetime.timedelta(days=1) - df['InvoiceDate']
df.head()


# In[18]:


df_r = df.groupby('CustomerID')['diff_days'].min()
df_r = df_r.reset_index()
df_r.head()


# In[19]:


df_r['Recency'] = df_r['diff_days'].dt.days
df_r.drop('diff_days', axis=1, inplace=True)
df_r.head()


# ## RFM Dataframe Creation

# In[20]:


df_rfm = pd.merge(df_r, df_f, on="CustomerID", how='inner')
df_rfm = pd.merge(df_rfm, df_m, on="CustomerID", how="inner")

df_rfm.head()


# ### Outliers Handling

# In[21]:


df_rfm.boxplot()


# In[22]:


def outliers_removal(df, col):
    Q1 = df_rfm[col].quantile(0.25)
    Q3 = df_rfm[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    return df


# In[23]:


df_rfm = outliers_removal(df_rfm, 'Monetary')
df_rfm.boxplot()


# In[24]:


df_rfm = outliers_removal(df_rfm, 'Frequency')
df_rfm.boxplot()


# In[25]:


df_rfm = outliers_removal(df_rfm, 'Recency')
df_rfm.boxplot()


# ### Rescaling Different Attributes

# In[26]:


df_rfm['Recency_log'] = df_rfm.Recency.apply(math.log)
df_rfm['Frequency_log'] = df_rfm.Frequency.apply(math.log)
df_rfm['Monetary_log'] = df_rfm.Monetary.apply(math.log)
df_rfm.head()
rfm_scaled = df_rfm[['Recency_log', 'Frequency_log', 'Monetary_log']]
rfm_scaled.columns = ['Recency', 'Frequency', 'Monetary']
rfm_scaled.head()


# ### k-Means with some of the arbitaries

# In[27]:


km = KMeans(n_clusters=5, max_iter=50)
km.fit(rfm_scaled)


# In[28]:


km.labels_


# ## Elbow-curve Vs SSD

# In[29]:


ssd = []
range_n_clusters = list(range(2,10))
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_scaled)
    
    ssd.append(kmeans.inertia_)
    

plt.plot(range_n_clusters,ssd, marker='o')
plt.title('Clusters Vs SSD')
plt.xlabel('No of Clusters')
plt.ylabel('Intertia')


# In[30]:


for n in range_n_clusters:
    km = KMeans(n_clusters=n, max_iter=50)
    km.fit(rfm_scaled)
    labels = km.labels_
    silhouette_avg = silhouette_score(rfm_scaled, labels)
    print(f"For n_clusters={n}, Silhouette Score is {silhouette_avg}")


# In[31]:


import timeit
st = timeit.default_timer()

def km_fit(df):
    km = KMeans(n_clusters=2, max_iter=50)
    km.fit(df)
    return km

km_times = []
for i in range(100):
    st = timeit.default_timer()
    km = km_fit(rfm_scaled)
    km_times.append(timeit.default_timer() - st)
    

km = km_fit(rfm_scaled)
km_time = np.mean(km_times)

km_time


# In[32]:


y = km.labels_


# In[33]:


sum(y==0)


# In[34]:


df_rfm['label'] = km.labels_
df_rfm.head()


# In[35]:


X=rfm_scaled.values
plt.scatter(X[y==0, 0], X[y==0, 1], s=50, c='red')
plt.scatter(X[y==1, 0], X[y==1, 1], s=50, c='blue')
# plt.scatter(X[y==2, 0], X[y==2, 1], s=50, c='green')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=100, color='cyan')
plt.legend()


# In[36]:


km.cluster_centers_


# In[37]:


import matplotlib
fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111,projection='3d')
xs=X[:,0]
ys=X[:,1]
# zs=X[:,2]
colors = ['red', 'blue', 'green']
ax.scatter(xs,ys,s=15,c=df_rfm['label'],cmap=matplotlib.colors.ListedColormap(colors))
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color = 'cyan', marker=',', 
            label = 'Centroids', s=100, alpha=1)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.legend()
plt.show()


# In[38]:


silhouette_km = silhouette_score(rfm_scaled, y)
silhouette_km


# ## DBSCAN

# In[39]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df_rfm[['Recency_log', 'Monetary_log', 'Frequency_log']])
distances, indices = nbrs.kneighbors(df_rfm[['Recency_log', 'Monetary_log', 'Frequency_log']])


# In[40]:


# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances_new = distances[:,1]
plt.figure(figsize=(9, 7))
plt.plot(distances_new)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[41]:


from sklearn.cluster import DBSCAN

dbsc_times = []

for i in range(100):
    st = timeit.default_timer()
    dbscan_opt=DBSCAN(eps=0.5,min_samples=6)
    dbscan_opt.fit(df_rfm[['Recency_log', 'Monetary_log', 'Frequency_log']])
    dbsc_times.append(timeit.default_timer() - st)

db_time = np.mean(dbsc_times)
db_time


# In[42]:


df_rfm['DBSCAN_opt_labels']=dbscan_opt.labels_
df_rfm['DBSCAN_opt_labels'].value_counts()


# In[43]:


fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111,projection='3d')
xs=df_rfm.Recency_log
ys=df_rfm.Frequency_log
zs=df_rfm.Monetary_log
# xs, ys, zs = X_scaled[:,0], X_scaled[:,1], X_scaled[:,2]
ax.scatter(xs,ys,zs,s=15,c=df_rfm['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors))
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.show()


# In[44]:


from sklearn import metrics

silhouette_dbsc = metrics.silhouette_score(df_rfm[['Recency_log', 'Monetary_log', 'Frequency_log']], df_rfm['DBSCAN_opt_labels'])
silhouette_dbsc


# In[45]:


df_stats = pd.DataFrame(
    [
        ['Silhouette Score', silhouette_km, silhouette_dbsc],
        ['RunTime', km_time, db_time]
    ],
    columns=['Metrics', 'K Means', 'DBSCAN']
)


# In[48]:


print(df_stats)


# In[47]:


df_stats.plot.bar(x='Metrics', rot=0)

