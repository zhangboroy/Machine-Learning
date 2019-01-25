import random
import math
import copy
import numpy
import matplotlib.pyplot as plt

#read the input file and return a list of data
def ReadData(fileName):
    #open the input file
    file = open(fileName, "r")

    #set an empty data list
    data = []

    for line in file:
        temp=line.strip().split(' ')
        data.append((float(temp[0]), float(temp[1])))

    return data

#return the min and max of every input variable
def MinMax(data):
    #initiate minX, maxX, minY, maxY
    minX = data[0][0]
    maxX = data[0][0]
    minY = data[0][1]
    maxY = data[0][1]

    #check the data 1 by 1
    for i in range(1, len(data)):
        if data[i][0]<minX:
            minX = data[i][0]
        if data[i][0]>maxX:
            maxX = data[i][0]
        if data[i][1]<minY:
            minY = data[i][1]
        if data[i][1]>maxY:
            maxY = data[i][1]

    return minX, maxX, minY, maxY

#return square of Euclidean distance
def SquaredDistance(point1, point2):
    return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2

#compute the centroid
def ComputeCentroid(data, cluster, k):
    #initiate centroid position
    centroid = []
    clusterCount = [0]*k
    for i in range(k):
        centroid.append([0,0])

    for i in range(len(data)):
        centroid[cluster[i]][0] += data[i][0]
        centroid[cluster[i]][1] += data[i][1]
        clusterCount[cluster[i]] += 1

    for i in range(k):
        if clusterCount[i]>0:
            centroid[i][0] = centroid[i][0]/clusterCount[i]
            centroid[i][1] = centroid[i][1]/clusterCount[i]

    return centroid
        
#K-means clusting, return k centroids and all data's clusters
def K_means(data, k):
    #get the min and max of X and Y
    minX, maxX, minY, maxY = MinMax(data)
    #set an empty centroid list
    centroid = [None]*k
    #set an empty distance list
    squaredDistance = [None]*k
    #set 2 empty lists of all data's clusters
    cluster = [None]*len(data)
    oldCluster = [0]*len(data)
    
    for i in range(k):
        #randomize the k centroids
        centroid[i] = [random.uniform(minX, maxX), random.uniform(minY, maxY)]

    while cluster!=oldCluster:
        oldCluster = copy.deepcopy(cluster)
        #assign all points to the closest centroid
        for i in range(len(data)):
            for j in range(k):
                squaredDistance[j] = SquaredDistance(data[i], centroid[j])

            cluster[i] = squaredDistance.index(min(squaredDistance))

        centroid = ComputeCentroid(data, cluster, k)

    return centroid, cluster

#compute Sum of Squared Errors(SSE)
def ComputeSSE(data, centroid, cluster):
    sse = 0
    for i in range(len(data)):
        sse += SquaredDistance(data[i], centroid[cluster[i]])

    return sse

#plot scatter graph
def ScatterPlot(data, cluster, title, centroid = None):
    #set color
    colorChoice = ("b", "c", "g", "k", "m", "r", "w", "y")*100
    color = []
    for i in range(len(data)):
        color.append(colorChoice[cluster[i]])

    #plot points in different clusters
    tempData = numpy.array(data)
    plt.scatter(tempData[:,0], tempData[:,1], c=color)

    #plot centroids of different clusters
    if centroid!=None:
        tempCentroid = numpy.array(centroid)
        plt.scatter(tempCentroid[:,0], tempCentroid[:,1], c="r")
    
    plt.title(title + " Result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return

#plot the Convergence Curve
def ConvergenceCurvePlot(data, kList, times):
    #initiate 2 lists for the coordinates
    x = []
    y = []

    #try different K for K-means
    for k in kList:
        print ("Running K-means for K=" + str(k) + "...")
        sse = []
        #try multiple times to get min SSE for a specific K
        if x==[]:
            for i in range(times):
                centroid, cluster = K_means(data, k)
                sse.append(ComputeSSE(data, centroid, cluster))
        else:
            while len(sse)<10 or min(sse)>=y[-1]:
                centroid, cluster = K_means(data, k)
                sse.append(ComputeSSE(data, centroid, cluster))
                
        x.append(k)
        minSSE = min(sse)
        y.append(minSSE)
        print ("  MSE: "+str(minSSE))

    #plot the curve
    plt.plot(x, y)
    plt.title('Convergence Curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

#run K-means for K=k several times and plot the best result
def K_MeansPlot(data, k, times):
    #run K-means for K=k several times and get the best result
    centroid = []
    cluster = []
    sse = []

    for i in range(times):
        tempCentroid, tempCluster = K_means(data, k)
        sse.append(ComputeSSE(data, tempCentroid, tempCluster))
        centroid.append(tempCentroid)
        cluster.append(tempCluster)

    index = sse.index(min(sse))
    
    #plot scatter graph
    ScatterPlot(data, cluster[index], "K-means Clustering", centroid[index])

#compute the inter-cluster distance
def InterClusterDistance(cluster1, cluster2, method):
    if method == "Distance between Centroids":
        #compute the 2 centroids
        clusterIndex = ((0,)*len(cluster1), (0,)*len(cluster2))
        centroid = (ComputeCentroid(cluster1, clusterIndex[0], 1), ComputeCentroid(cluster2, clusterIndex[1], 1))
        #compute the inter-cluster distance
        return math.sqrt(SquaredDistance(centroid[0][0], centroid[1][0]))
    else:
        #compute all n1*n2 distances
        squaredDistance = []
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                squaredDistance.append(SquaredDistance(cluster1[i], cluster2[j]))

        #compute the inter-cluster distance
        if method == "MIN":
            return math.sqrt(min(squaredDistance))
        elif method == "MAX":
            return math.sqrt(max(squaredDistance))
        elif method == "Group Average":
            return sum(math.sqrt(d) for d in squaredDistance)/len(cluster1)/len(cluster2)

#hierarchical clustering
def HierarchicalClustering(data, clusters, method):
    #initiate the clusters
    cluster = []
    for i in range(len(data)):
        cluster.append([])
        cluster[i].append(data[i])

    #initiate the 2 merged clusters' indexes
    merge = [None]*2

    #select the 2 merged clusters
    for k in range(len(data)-clusters):
        #initiate the min inter-cluster distance
        minInterClusterDistance = -1
        
        #compute the inter-cluster distance 1 by 1 to get the min inter-cluster distance and 2 merged clusters
        for i in range(len(cluster)):
            if len(cluster[i])>0:
                for j in range(i+1, len(cluster)):
                    if len(cluster[j])>0:
                        tempInterClusterDistance = InterClusterDistance(cluster[i], cluster[j], method)
                        if tempInterClusterDistance<minInterClusterDistance or minInterClusterDistance<0:
                            minInterClusterDistance = tempInterClusterDistance
                            merge[0] = i
                            merge[1] = j

        #merge the 2 clusters
        while len(cluster[merge[1]])>0:
            cluster[merge[0]].append(cluster[merge[1]].pop())

    #initiate the clustering result
    clusterResult = []
    clusterIndex = 0
    clusteredData = []
    
    #generate the clustered data and cluster index
    for i in range(len(data)):
        if len(cluster[i])>0:
            for j in range(len(cluster[i])):
                clusteredData.append(cluster[i][j])
                clusterResult.append(clusterIndex)
            clusterIndex += 1

    return clusteredData, clusterResult


#read the data
data = ReadData("A.txt")
#plot the Convergence Curve
ConvergenceCurvePlot(data, range(2,11), 10)
#run K-means for K=3 10 times and plot the best result
K_MeansPlot(data, 3, 10)

#read the data
data = ReadData("B.txt")
#run hierarchical clustering for 2 clusters with 4 types of inter-cluster distance
method = ["MIN", "MAX", "Group Average", "Distance between Centroids"]
for i in range(4):
    print ("Running " + method[i] + " hierarchical clustering...")
    clusteredData, cluster = HierarchicalClustering(data, 2, method[i])
    ScatterPlot(clusteredData, cluster, method[i] + " Hierarchical Clustering")
