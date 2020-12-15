from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale,RobustScaler,StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
# Scale standardisation of numerical values
data_raw = pd.read_excel("/Users/admin2/Documents/Telco_customer_churn (1).xlsx")
tdf = data_raw
tdf.drop(labels=['Customer ID','Count', 'Quarter'],axis=1,inplace=True)

numCols = tdf.select_dtypes("number").columns
catCols = tdf.select_dtypes("object").columns
numCols= list(set(numCols))
catCols= list(set(catCols))


numerical_features = pd.DataFrame(StandardScaler().fit_transform(tdf[tdf._get_numeric_data().columns]),
                                  index=tdf.index,columns=tdf._get_numeric_data().columns)

categorical_features  = pd.get_dummies(tdf[catCols])

Xy_scaled = pd.concat([numerical_features,categorical_features],axis=1)
 

Xy_scaled_minkowski = squareform(pdist(Xy_scaled, 'minkowski'))

# find k clusters
results_kmedoids = dict()

k_cand = list(range(2,8))
#k_cand.extend(list(np.arange(10,55,5)))

for k in k_cand:
    # initiate k random medoids - sets k clusters
    initial_medoids = np.random.randint(0,1000,size=k)
    kmedoids_instance = kmedoids(Xy_scaled_minkowski,initial_medoids, data_type='distance_matrix')    

    # run cluster analysis and obtain results
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()

    # convert cluster output
    cluster_array = pd.DataFrame([(x,e) for e,i in enumerate(clusters) for x in i if len(i)>1]).sort_values(by=0)[1].values
    
    # score
    score1 = silhouette_score(Xy_scaled_minkowski, cluster_array, metric='precomputed')
    score2 = silhouette_score(Xy_scaled, cluster_array,metric='correlation')
    
    # store
    results_kmedoids[k] = {'k':cluster_array,'s1':score1,'s2':score2}
import matplotlib.pyplot as plt   
plt.plot([i for i in results_kmedoids.keys()],[i['s1'] for i in results_kmedoids.values()],label='Minkowski')
plt.plot([i for i in results_kmedoids.keys()],[i['s2'] for i in results_kmedoids.values()],label='correlation')
plt.legend()
plt.xticks(k_cand);

mb = pd.Series(cluster_array)  # converting numpy array into pandas series object 
data_raw['clust'] = mb 
clust_data = data_raw.iloc[:,[27,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                              16,17,18,19,20,21,22,23,24,25,26]]
clust_data.head()

clust_data.iloc[:, 2:27].groupby(clust_data.clust).mean()

clust_data.to_csv("Kmeans_university2.csv", encoding = "utf-8")
