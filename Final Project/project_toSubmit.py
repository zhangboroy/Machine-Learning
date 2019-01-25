import json
import platform
import math
from copy import deepcopy
import time
import random

#
# The ReadData() function reads the short text from the input file, then generates
# a list of data. Each entry of the list is an array of size 4 containing the
# original cluster No. (data[i][0]), the cluster No. after classification
# (data[i][1]), the total number of words (data[i][2]), and a dictionary containing
# the frequency of each word (eg. data[i][3][``apple'']).
#
def ReadData(inputFileName):
    # Initializing return variables
    data = []       # the data

    # Attempting to open the input file
    try:
        file = open(inputFileName, "r")
    except IOError:
        print("Error reading file.")
        return data

    index = 0
    #read the short text 1 by 1
    for line in file:
        text = json.loads(line)
        #transfer the short text into a word vector:
        #set an empty word vector
        data.append([None]*4)
        data[index][3] = {}
        #set the original cluster No.
        data[index][0] = text["clusterNo"]

        #read the words 1 by 1
        words = text["textCleaned"].strip().split(' ')
        for word in words:
            #add the word into the word vector or update the count
            data[index][3].setdefault(word, 0)
            data[index][3][word]+=1

        #add total word number into the word vector
        data[index][2] = len(words)
        index += 1

    file.close()
    return data


#
# The ComputeProbability() function computes the probability of document "d",
# choosing each of the K existing clusters as well as a new cluster (test)
#
# Parameters:
# ===========
# d        = the documents and associated word data to calculate probabilities for
#            d[0] is the original cluster number of the document
#            d[1] is the current cluster number
#            d[2] is the total number of words in the document
#            d[3] is the list of words and their frequencies in the document
# clusters = the list of clusters that currently exist
#            clusters[z][0] is the total number of documents in the zth cluster
#            clusters[z][1] is the total number of words in the zth cluster
#            clusters[z][2] is the list of word frequencies in the zth cluster
# V        = the size of the vocabulary for all currently recorded documents
# D        = the number of current recorded documents
# alpha    = concentration parameter for each cluster
# beta     = pseudo number of occurrences of each word in a new cluster
#
# Nomenclature Notes:
# ===================
# mz       = number of documents in cluster z
# nz       = number of words in cluster z
# Nd       = number of words (in a document)
# nzw      = frequency of a particular word ocurring in cluster z
# Ndw      = frequency of a particular word ocurring in a particular document
# mK       = number of documents in the kth cluster
# nK       = number of words in the kth cluster
# nwK      = frequency of a particular word in the kth cluster (in practice this is a sparse list)
# zd       = cluster index of a particular document
#
# ** NOTE: These nomenclature notes correspond with the variable designations used in the
#          original paper.
#
#
def ComputeProbability(d, clusters, V, D, alpha, beta):
    zd  = 1         # the index to the current cluster of the document
    Nd  = 2         # the index to the number of words in the document
    Ndw = 3         # index to the list containing the words and their frequencies used in the document
    mz  = 0         # the index to the number of documents in the cluster
    nz  = 1         # the index to the number of words in the cluster
    nzw = 2         # the index to the frequency of each word in the cluster (a dict)
    
    probability = [None]*(len(clusters)+1)
    for z in range(len(clusters)):
        p1 = 1      # product #1
        p2 = 1      # product #2
        for word in d[Ndw]:
            for j in range(d[Ndw][word]):
                if (word in clusters[z][nzw]):
                    p1 *= clusters[z][nzw][word] + beta + j
                else:
                    p1 *= beta + j

        for i in range(d[Nd]):
            p2 *= clusters[z][nz] + V*beta + i

        probability[z] = (clusters[z][mz]*p1) / ((D - 1 + alpha*D)*p2)

    p1 = 1      # product #1
    p2 = 1      # product #2
    for word in d[Ndw]:
        for j in range(d[Ndw][word]):
            p1 *= beta + j
    for i in range(d[Nd]):
        p2 *= V*beta + i

    probability[len(clusters)] = (alpha*D*p1) / ((D - 1 + alpha*D)*p2)
    
    return probability

#select the cluster based on the probability
def SelectCluster(probability):
    summedProbability = sum(probability)
    
    for i in range(len(probability)):
        probability[i] = probability[i]/summedProbability
        
    p = random.random()
    for i in range(len(probability)):
        if (p<probability[i]):
            return i
        else:
            p -= probability[i]

#    
# The MStream() function computes the MStream algorithm as described in the original
# paper.
#
# Parameters:
# ===========
# dt         = the list of documents and associated word data to run the algorithm on
#              dt[ID][0] is the original cluster number of the document ID
#              dt[ID][1] is the current cluster number
#              dt[ID][2] is the total number of words in the document
#              dt[ID][3] is the list of words and their frequencies in the document
# clusters   = the list of clusters that currently exist
#              clusters[z][0] is the total number of documents in the zth cluster
#              clusters[z][1] is the total number of words in the zth cluster
#              clusters[z][2] is the list of word frequencies in the zth cluster
# vocabulary = set of unique words used in documents till now
# D          = number of current recorded documents
# I          = number of iterations to perform
# alpha      = concentration parameter for each cluster
# beta       = pseudo number of occurrences of each word in a new cluster
#
# Nomenclature Notes:
# ===================
# mz         = number of documents in cluster z
# nz         = number of words in cluster z
# Nd         = number of words (in a document)
# nzw        = frequency of a particular word ocurring in cluster z
# Ndw        = frequency of a particular word ocurring in a particular document
# mK         = number of documents in the kth cluster
# nK         = number of words in the kth cluster
# nwK        = frequency of a particular word in the kth cluster (in practice this is a sparse list)
# zd         = cluster index of a particular document
#
# ** NOTE: These nomenclature notes correspond with the variable designations used in the
#          original paper.
#
def MStream(dt, clusters, vocabulary, D, I, alpha, beta):
    # Nomenclature reference constants
    zd  = 1 # index to the current cluster number
    Nd  = 2 # index to the total number of words in the document
    Ndw = 3 # index to the list containing the words and their frequencies used in the document
    mK  = 0 # index to the number documents in the kth cluster
    nK  = 1 # index to the number of words in the kth cluster
    nwK = 2 # index to the frequency of each particular word in the kth cluster

    # One pass clustering process
    for ID in range(len(dt)):
        #update V and D
        vocabulary.update(set(dt[ID][Ndw]))
        V = len(vocabulary)
        D += 1
        
        # Compute the probability of document d choosing each of the K existing
        # clusters and a new cluster.
        probability = ComputeProbability(dt[ID], clusters, V, D, alpha, beta)

        # Sample cluster index z for document d according to the above K + 1
        # probabilities.
        dt[ID][zd] = SelectCluster(probability)

        # If a new cluster is chosen, the corresponding probability will be the last
        # element in the probability vector which will have one element
        # more than the clusters array. When this is the case, len(clusters)
        # will be equal to (len(probability)-1), which will be the index of
        # the last element in the probability vector.
        if dt[ID][zd]==len(clusters):
            # Initialize mK, nK, and nwK as zero
            clusters.append([0, 0, {}])
        
        # Update the number of documents (mz += 1)
        clusters[dt[ID][zd]][mK] += 1
        # Update the number of words (nz += Nd)
        clusters[dt[ID][zd]][nK] += dt[ID][Nd]
        # Update the number of every word
        for word in dt[ID][Ndw]:
            clusters[dt[ID][zd]][nwK].setdefault(word, 0)
            clusters[dt[ID][zd]][nwK][word] += dt[ID][Ndw][word]

    # Update clustering process
    for iter in range(1, I):
        for ID in range(len(dt)):
            # Update the number of documents so that the current document is
            # removed from the current cluster assignment, pending reassignment.
            clusters[dt[ID][zd]][mK] -= 1
            # Update the number of words to removed the words from this document
            # from the current cluster assignment.
            clusters[dt[ID][zd]][nK] -= dt[ID][Nd]
            # Update the frequency of every word in the data set of currently
            # assigned clusters to remove the data from this document, pending
            # reassignment to another cluster.
            for word in dt[ID][Ndw]:
                clusters[dt[ID][zd]][nwK][word] -= dt[ID][Ndw][word]
                if clusters[dt[ID][zd]][nwK][word]==0:
                    del clusters[dt[ID][zd]][nwK][word]
                
            # Compute the probability of document dt[ID] choosing each of the K
            # existing clusters and a new cluster.
            probability = ComputeProbability(dt[ID], clusters, V, D, alpha, beta)

            # Sample cluster index z for document dt[ID] according to the above
            # K + 1 probabilities.
            dt[ID][zd] = SelectCluster(probability)

            # If a new cluster is chosen
            if (dt[ID][zd]==len(clusters)):
                # Initialize mK, nK, and nwK as zero
                clusters.append([0, 0, {}])
            
            # Update the number of documents
            clusters[dt[ID][zd]][mK] += 1
            # Update the number of words
            clusters[dt[ID][zd]][nK] += dt[ID][Nd]
            # Update the number of every word
            for word in dt[ID][Ndw]:
                clusters[dt[ID][zd]][nwK].setdefault(word, 0)
                clusters[dt[ID][zd]][nwK][word] += dt[ID][Ndw][word]

    return

#
# The MstreamF() function calculates MStreamF algorithm as described in the
# original paper.  This function operates similarly to the MStream() function, 
# which it calls, but incorporates rules for "forgetting" outdated information.
#
# Parameters:
# ===========
# data       = a list of batches of input data (the dimensionality of this data is one higher than
#              the dimensionality of the input to the MStream function)               
#              data[t] is the t-th batch of input data
# clusters   = the list of clusters that currently exist
#              clusters[z][0] is the total number of documents in the zth cluster
#              clusters[z][1] is the total number of words in the zth cluster
#              clusters[z][2] is the list of word frequencies in the zth cluster
# Bs         = the number of stored batches
# I          = number of iterations to perform
# alpha      = concentration parameter for each cluster
# beta       = pseudo number of occurrences of each word in a new cluster
#
def MStreamF(data, clusters, Bs, I, alpha, beta):
    # Nomenclature reference constants
    zd  = 1 # index to the current cluster number
    Nd  = 2 # index to the total number of words in the document
    Ndw = 3 # index to the list containing the words and their frequencies used in the document
    mK  = 0 # index to the number documents in the kth cluster
    nK  = 1 # index to the number of words in the kth cluster
    nwK = 2 # index to the frequency of each particular word in the kth cluster

    #initialize the vocabulary set and total document number
    vocabulary = set()
    D = 0
    # CF vectors of batches
    clustersHistory = []
    for t in range(len(data)):
        if t>0:
            print ("\nMStreamF ", t*len(data[t-1]), " - ", t*len(data[t-1])+len(data[t])-1)
        else:
            print ("\nMStreamF ", t, " - ", t+len(data[t])-1)
            
        startTime = time.time()
        # If the number of stored batches is larger than Bs, we delete the oldest
        # batch from current CF vectors.
        if t > Bs:
            for i in range(len(clustersHistory[0])):
                # Update the number of documents
                clusters[i][mK] -= clustersHistory[0][i][mK]
                # Update the number of words
                clusters[i][nK] -= clustersHistory[0][i][nK]
                # Update the number of every word
                for word in clustersHistory[0][i][nwK]:
                    clusters[i][nwK][word] -= clustersHistory[0][i][nwK][word]
                    if clusters[i][nwK][word]==0:
                        del clusters[i][nwK][word]
            
            # Remove the CF vectors of the oldest batch b.
            clustersHistory.pop(0)

            #update vocabulary and D
            vocabulary = set()
            for i in range(len(clusters)):
                vocabulary.update(set(clusters[i][nwK]))

            D -= len(data[t-1])

        # Initialize CF vectors of bath t with current CF vectors.
        clustersHistory.append(deepcopy(clusters))
        # Clustering documents of batch t with MStream.
        MStream(data[t], clusters, vocabulary, D, I, alpha, beta)
        # Compute CF vectors of batch t
        for i in range(len(clusters)):
            if i>=len(clustersHistory[-1]):
                clustersHistory[-1].append([0, 0, {}])
            # Update the number of documents
            clustersHistory[-1][i][mK] = clusters[i][mK] - clustersHistory[-1][i][mK]
            # Update the number of words
            clustersHistory[-1][i][nK] = clusters[i][nK] - clustersHistory[-1][i][nK]
            # Update the number of every word
            for word in clusters[i][nwK]:
                clustersHistory[-1][i][nwK].setdefault(word, 0)
                clustersHistory[-1][i][nwK][word] = clusters[i][nwK][word] - clustersHistory[-1][i][nwK][word]
                if clustersHistory[-1][i][nwK][word]==0:
                    del clustersHistory[-1][i][nwK][word]

        D += len(data[t])
        #output running time and NMI
        endTime = time.time()
        print ("Running Time: ", endTime - startTime)
        NMI = ComputeNMI(data[t], clusters)
        print ("NMI: ", NMI)
        
    return

# the ComputeNMI() function computes NMI for the whole dataset
def ComputeNMI(data, clusters):
    N = len(data)
    K = len(clusters)
    nc = {}
    nck = {}
    nk = [0]*K
    #count nc, nk, nck
    for i in range(len(data)):
        #update nc
        nc.setdefault(str(data[i][0]), 0)
        nc[str(data[i][0])] += 1
        #update nk
        nk[data[i][1]] += 1
        #update nck
        nck.setdefault(str(data[i][0]), [0]*K)
        nck[str(data[i][0])][data[i][1]] += 1

    #the Numerator of NMI
    sum1 = 0
    #the left part of the Denominator of NMI
    sum2 = 0
    #the right part of the Denominator of NMI
    sum3 = 0
    for c in nc:
        if nc[c]>0:
            sum2 += nc[c] * math.log(nc[c]/N, 2)
        for k in range(K):
            if nck[c][k]>0:
                sum1 += nck[c][k] * math.log(N*nck[c][k]/(nc[c]*nk[k]), 2)

    for k in range(K):
        if nk[k]>0:
            sum3 += nk[k] * math.log(nk[k]/N, 2)

    #return NMI
    return sum1 / math.sqrt(sum2 * sum3)

#
# The RunMStream() function uses the MStream function to cluster the whole dataset.
#
# Parameters:
# ===========
# data          = the whole dataset
# clusters      = an empty list for cluster information
# documentStart = a list of indexes of the first document for each batch
# domumentEnd   = a list of indexes of the end for each batch
# I             = the number of iterations of each batch
# alpha         = concentration parameter for each cluster
# beta          = pseudo number of occurrences of each word in a new cluster
#
def RunMStream(data, clusters, documentStart, domumentEnd, I, alpha, beta):
    vocabulary = set()
    for i in range(len(documentStart)):
        print ("\nMStream ", documentStart[i], " - ", domumentEnd[i]-1)
        startTime = time.time()
        MStream(data[documentStart[i]:domumentEnd[i]], clusters, vocabulary, documentStart[i], I, alpha, beta)
        endTime = time.time()
        print ("Running Time: ", endTime - startTime)
        NMI = ComputeNMI(data[documentStart[i]:domumentEnd[i]], clusters)
        print ("NMI: ", NMI)
    
    return

#
# The RunMStreamF() function uses the MStreamF function to cluster the whole dataset.
#
# Parameters:
# ===========
# data          = the whole dataset
# clusters      = an empty list for cluster information
# documentStart = a list of indexes of the first document for each batch
# domumentEnd   = a list of indexes of the end for each batch
# Bs            = the number of stored batches
# I             = the number of iterations of each batch
# alpha         = concentration parameter for each cluster
# beta          = pseudo number of occurrences of each word in a new cluster
#
def RunMStreamF(data, clusters, documentStart, domumentEnd, Bs, I, alpha, beta):
    dataStream = []
    for i in range(len(documentStart)):
        #add each batch of data into the stream
        dataStream.append(data[documentStart[i]:domumentEnd[i]])

    MStreamF(dataStream, clusters, Bs, I, alpha, beta)
    return


# the main program to run our experiment
print("Initiating run on " + platform.system() + " platform (release " + platform.release() + ")")
if (platform.system() == "Linux"):
    data = ReadData("data/TREC")
else:
    data = ReadData("data\\TREC")

clusters = []
# the indexes for TREC or TREC-T data
documentStart = [0, 1896, 3793, 5690, 7587, 9484, 11381, 13278, 15175, 17072, 18969, 20866, 22763, 24660, 26557, 28454]
domumentEnd = [1896, 3793, 5690, 7587, 9484, 11381, 13278, 15175, 17072, 18969, 20866, 22763, 24660, 26557, 28454, 30322]
# the indexes for News or News-T data
# documentStart = [0, 695, 1391, 2087, 2783, 3479, 4175, 4871, 5567, 6263, 6959, 7655, 8351, 9047, 9743, 10439]
# domumentEnd = [695, 1391, 2087, 2783, 3479, 4175, 4871, 5567, 6263, 6959, 7655, 8351, 9047, 9743, 10439, 11109]
RunMStream(data, clusters, documentStart, domumentEnd, 10, 0.03, 0.03)
NMI = ComputeNMI(data, clusters)
print ("total NMI: ", NMI)
clusters = []
RunMStreamF(data, clusters, documentStart, domumentEnd, 1, 10, 0.03, 0.03)
NMI = ComputeNMI(data, clusters)
print ("total NMI: ", NMI)
