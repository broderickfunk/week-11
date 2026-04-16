import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from time import time

def kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    return model.cluster_centers_, model.labels_


diamonds = sns.load_dataset("diamonds")
diamonds_num = diamonds.select_dtypes(include='number')

def kmeans_diamonds(n, k):
    X = diamonds_num.iloc[:n].to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return sum(times) / n_iter