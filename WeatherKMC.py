from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from timeit import default_timer as timer  
from numba import jit, cuda
import logging;

logging.disable(logging.WARNING)

df = pd.read_csv(r'us_accidents_weather_data.csv')
data = df[['Severity', 'Temperature(F)','Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)','Wind_Speed(mph)', 'Precipitation(in)']]
data = data.dropna()
data = data[data['Severity'] == 4]
data = data.reset_index(drop=True)
data = data.head(150)
print(len(data))
print(data.head(5))
k = 3
eps = 5
iterationsMax = 20
clusters = pd.DataFrame(data = 0.0, columns=data.columns, index=range(k))
c = np.full(len(data), -1)

# data: points df
# clusters: clusters df
@jit(target_backend='cuda', nogil=True, cache=True) 
def initClusters(data, clusters):
    for i, ii in enumerate(clusters.iloc[0]):
        clmax = data.max().iloc[i]
        clmin = data.min().iloc[i]
        for j,jj in clusters.iterrows():
            clusters.iloc[j].iloc[i] = (clmax - clmin) * np.random.random() + clmin

initClusters(data, clusters)

# p: index of point 
# c: index of cluster
# clusters: clusters df
# data: points df
@jit(target_backend='cuda', nogil=True, cache=True) 
def sse(p: int, c: int, clusters: pd.DataFrame, data: pd.DataFrame) -> float:
    error: float = 0.0
    for i, ii in enumerate(clusters):
        error += (clusters.iloc[c][i] - data.iloc[p][i])**2
    return error

# clusters: clusters df
# data: points df
# c: the cluster assignment array
@jit(target_backend='cuda') 
def reassignPoints(data, clusters, c):
    for i,ii in data.iterrows():
        bestIndex = -1
        bestSSE = 100000000
        for j,jj in clusters.iterrows():
            currentSSE = sse(i, j, clusters, data)
            if(currentSSE < bestSSE):
                bestSSE = currentSSE
                bestIndex = j
        c[i] = bestIndex

# clusters: clusters df
# data: points df
# c: the cluster assignment array
@jit(target_backend='cuda') 
def totalError(c, clusters, data):
    error = 0.0
    for i in range(len(c)):
        error = error + sse(i, c[i], clusters, data)
    return(error)

# c: the cluster assignment array
# t: the target cluster
@jit(target_backend='cuda') 
def clusterContent(c, t):
    total = 0
    for i in range(len(c)):
        if (c[i] == t):
            total = total + 1
    return (total)

# clusters: clusters df
# data: points df
# c: the cluster assignment array
# cluster: the focal cluster
@jit(target_backend='cuda') 
def clusterError(c, cluster, clusters, data):
    error = 0.0
    for i in range(len(c)):
        if(c[i] == cluster):
            error = error + sse(i, c[i], clusters, data)
    return(error)

@jit(target_backend='cuda') 
def moveClusters(c, clusters, data):
    for i,ii in clusters.iterrows():
        for j,jj in enumerate(clusters.iloc[i]):
            sum = 0.0
            totP = 0.0
            for k in range(len(c)):
                if (c[k] == i):
                    sum = sum + data.iloc[k][j]
                    totP = totP + 1.0
            if (totP != 0):
                clusters.iloc[i][j] = sum/totP

@jit(target_backend='cuda') 
def KMeans2(data, k, eps, iterationsMax):
    clusters = pd.DataFrame(data = 0.0, columns=data.columns, index=range(k))
    c = np.full(len(data), -1)
    initClusters(data, clusters)

    oldError = -1 * eps
    newError = 0

    for a in range(1,iterationsMax):
        reassignPoints(data, clusters, c)
        moveClusters(c, clusters, data)
        newError = totalError(c, clusters, data)
        #print(newError)
        if(abs(newError - oldError) < eps):
            break
        oldError = newError
    #print(newError)
    output = clusters
    cle = np.full(k,-1.0)
    ccn = np.full(k, -1)
    for i in range(k):
        cle[i] = round(clusterError(c, i, clusters, data),3)
        ccn[i] = clusterContent(c, i)
    output['error'] = cle
    output['numOfPoints'] = ccn

    return(output)

start = timer()
result = KMeans2(data, k, eps, iterationsMax)
print(result)
print("without GPU:", timer()-start)  