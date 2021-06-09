# Machine Learning Project - Step 4
# *****************************************
# COLLABORATION STATEMENT:
# https://www.w3schools.com/
# https://numpy.org/
# https://www.askpython.com/python/array/python-add-elements-to-an-array
# https://www.kite.com/python/ 
# https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
# *****************************************

import KMC_functions as kmc 
import numpy as np
from csv import writer
from csv import reader


# *******************************************************************
# DRIVER / SCRIPT
# *******************************************************************

# Select K random centroid points. These should fall within the range of the feature sets.
K = 2

#Generate a 2D array for centroids
#column 0 = glucose
#column 1 hemoglobin
centroids = np.random.rand(K,2) 

print("Generated centroids:", centroids)

g,h,c = kmc.openckdfile()

#The number of points will be the size of any g, h, or c
dataLength = len(g)

#Normalize hemoglobin and glucose arrays
gs, gmin, gmax = kmc.normalizeData(g)
hs, smin, smax = kmc.normalizeData(h)

loop = 1
iterations = 0
while(loop):
    #Count how many iterations are needed
    iterations+= 1

    #Calculate distances from points to centroids
    distanceArr = kmc.calculateDistArr(K, gs, hs, centroids)

    #Find the assignment (from 0 to K) for each point
    assignment = kmc.calculateAssignment(distanceArr, K, dataLength)

    #Update centroids
    newCentroids = kmc.updateCentroids(K, gs, hs, assignment, centroids)

    #If centroids did not change, stop the loop
    if ((newCentroids == centroids).all()):
        loop = 0

    centroids = newCentroids

#Denormalize the centroids
dCentroids = kmc.denormalize(centroids, h, g, K)

print(iterations, "iterations.")
print("Centroid features (locations): ")
for i in range (K):
    print("Centroid", i, dCentroids[i][0], dCentroids[i][1])
    
# print("Original classifications", c[35:55])
print("KMC Classifications:")
for i in range(dataLength):
    print(assignment[i])

print("Graphing...")
kmc.graphingKMeans(g, h, assignment, dCentroids, K)