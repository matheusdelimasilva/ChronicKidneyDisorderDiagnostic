# Machine Learning Project - Functions for Step 4
# *****************************************
# https://www.w3schools.com/
# https://numpy.org/
# https://www.askpython.com/python/array/python-add-elements-to-an-array
# https://www.kite.com/python/ 
# https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
# *****************************************


import numpy as np
import matplotlib.pyplot as plt
import random

# *******************************************************************
# FUNCTIONS
# *******************************************************************
'''
PURPOSE: opens a file 'ckd.csv' and gets glucose and hemoglobin levels, also gets their classifications 
PARAMETERS: none
RETURNS: glucose and hemoglobin levels, and their classifications 
'''
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose[0:], hemoglobin[0:], classification[0:]

'''
PURPOSE: normalize a NumPy array (glucose and hemoglobin both will need to be normalized), calculating min and max values
PARAMETERS: NumPy array to be normalized
RETURNS: the normalized array, the minimum value of the old array, and the maximum value of the old array.
'''
def normalizeData(array):
    #Find min
    min = np.amin(array)
    #Fin max
    max = np.amax(array)
    #Claculate the scaled values
    scaled = (array - min)/(max - min)
    #Return them
    return scaled, min, max

'''
PURPOSE: helper function for calculateDistArr. Calculates the distance between gs and hs and provided centroids. 
PARAMETERS: which centroid (k), glucose array, hemoglobin array, and centroid arrays
RETURNS: calculated distance
'''
def calculateDistForK(k, gs, hs, centroids):
    #Create the array to store the calculated distances
    calcDist = np.zeros(len(gs))

    #Get the kth point 
    gCent = centroids[k, 0]
    hCent = centroids[k,1]

    #Calculate the distance from centroid to 
    # kth glucose and hemoglobin points
    distanceG = (gs - gCent)**2
    distanceH = (hs - hCent)**2
    calcDist = (distanceG + distanceH)*0.5

    return calcDist

'''
PURPOSE: calculate the distances between points and centroids
PARAMETERS: number of centroids (K), glucose array, hemoglobin array, and centroid arrays
RETURNS: array with distances from glucose and hemoglobin points to
centroid points
'''
def calculateDistArr(K, gs, hs, centroids):

    #Initial distance array has 0 rows
    distanceArr = np.zeros((0, len(gs)))

    for k in range(K):
        #get a new row for k using a helper function
        newRow = calculateDistForK(k, gs, hs, centroids)

        #add it to distance array, this is the k-th centroid
        #distances
        distanceArr = np.vstack([distanceArr, newRow])

    return distanceArr

'''
PURPOSE: uses the distance array to find the centroid closest to
each point, assigning it a classification (from 0 to K)
PARAMETERS: array with distances for each point, number of centroids (K), number of points (length of data taken from file)
RETURNS: array with assignments for all points
'''
def calculateAssignment(distanceArr, K, dataLength):
    # For each point
    assignment = np.zeros(dataLength)
    dist = np.zeros(K)
    for i in range(dataLength):
        for k in range(K):
            #Distance from point i from centroid k is 
            # distanceArr[k,i]
            dist[k] = distanceArr[k,i]

        #Find the index of the closest centroid for this point
        #This is its classification (from 0 to K)
        assignment[i] = np.argmin(dist)
    
    return assignment 

'''
PURPOSE: Updates the location of each centroid by finding the mean of each feature of the observations (data points) currently assigned to that centroid
PARAMETERS: number of centroids (K), glucose array, hemoglobin array, and centroids arrays
RETURNS: updated centroid array
'''
def updateCentroids(K, gs, hs, assignment, centroids):
    #Create newCentroids that will store the updated values
    newCentroids = np.zeros((K, 2)) 

    for k in range(K):
        #if there is any assignment == k
        if((assignment == k).any()):
            newCentroids[k,0] = float(np.mean(gs[assignment==k]))
            newCentroids[k,1] = float(np.mean(hs[assignment==k]))
        # print("Centroid",k,"location is now", newCentroids[k,0], newCentroids[k,1])
    return newCentroids

'''
PURPOSE: "Denormalizes" the centroids' data. Transform values
ranging from 0 to 1 into values ranging from max and min (of
glucose and hemoglobins). 
PARAMETERS: denormalized centroids array, denormalized hemoglobin
array and K
RETURNS: denormalized centroids array
'''
def denormalize(centroids, h, g, K):
    #Find min
    minH = np.amin(h)
    minG = np.amin(g)
    #Fin max
    maxH = np.amax(h)
    maxG = np.amax(g)
    #Calculate the denormalized values
    #dCentroids = np.zeros((2, K))
    dCentroids = centroids
    for k in range(K):
        dCentroids[k,0] = (centroids[k][0] * (maxG - minG)) + minG
        dCentroids[k,1] = (centroids[k][1] * (maxH - minH)) + minH
    
    #Return
    return dCentroids

'''
PURPOSE: Graphs hemoglobin and glucose levels with calculated centroids
PARAMETERS: glucose array, hemoglobin array, their calculated assignments, and centroids arrays
RETURNS: nothing
'''
def graphingKMeans(glucose, hemoglobin, assignment, centroids, K):
    plt.figure()

    #Converting to int
    assignment = assignment.astype(int)

    for i in range(assignment.max()+1):
        rcolor = np.random.rand(3,)
        plt.plot(hemoglobin[assignment==i],glucose[assignment==i], ".", label = "Class " + str(i), color = rcolor)
        plt.plot(centroids[i, 1], centroids[i, 0], "D", label = "Centroid " + str(i), color = rcolor)
        
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.show()

