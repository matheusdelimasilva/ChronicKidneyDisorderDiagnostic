
By: Matheus de Lima Silva

# ************************************************************** #
#                          PROJECT                               #
# ************************************************************** #

This project aims to help help doctors diagnose chronic kidney 
disease (CKD). It uses Machine Learning algorithms to analyze 
hemoglobin and glucose levels from patients to diagnose if a 
patient  is likely to have CKD or not. 

Two different algorithms were used: 
- Nearest Neighbor Classification Algorithm (NN_KNN)
- K-Means Clustering Classification Algorithm (KMC)

# ************************************************************** #
#                          FILES:                                #
# ************************************************************** #
ckd.csv: comma-separated values in file. Each column represents a
different observation from a patient: Blood Glucose Levels,
Hemoglobin levels, its classification (0 or 1), respectively.

NN_KNN_script.py: Nearest Neighbor Classification Algorithm.
Classifies a random-generated point into 0 or 1 by looking at its 
closest point's observations. Uses data from ckd.csv. This computes
the result of a Nearest Neighbor Classification and a K-Nearest 
Neighbors Classification, where K is how many points will be
used to classify the test point. K can be edited by the user. 

NN_KNN_functions.py: Contains the functions used by NN_KNN_script.

KMC_script.py: K-Means Clustering Classification Algorithm to
    classify the data in ckd.csv into K clusters. User defines how 
    many clusters/groups by editing the value of K.

KMC_functions.py: Stores functions used by KMC_script. 

README.txt: This file 

# ************************************************************** #
#                          HOW TO USE:                           #
# ************************************************************** #

1. Include your data in the ckd.csv file.
2. To compute the Nearest Neighbor Classification change K (number
of neighbors that will be calculated) to the desirable number and
run NN_KNN_script.py. Both KNN and NN computations will be printed
to the user. A graph showing the data and generated case will be
plotted. 
3. To compute the KMC Classification change K (number of clusters)
to the desirable number and run KMC_script.py. A graph will be 
plotted showing the data classified into clusters. The locations of
the centroids separating the clusters will be printed to the user. 


# ************************************************************** #
#                       KMC FUNCTIONS:                           #
# ************************************************************** #

normalizeData(array)
    PURPOSE: normalize a NumPy array (glucose and hemoglobin both
    will need to be normalized), calculating min and max values
    PARAMETERS: NumPy array to be normalized
    RETURNS: the normalized array, the minimum value of the old     
    array, and the maximum value of the old array.

calculateDistForK(k, gs, hs, centroids)
    PURPOSE: helper function for calculateDistArr. Calculates the 
    distance between gs and hs and provided centroids. 
    PARAMETERS: which centroid (k), glucose array, hemoglobin array, 
    and centroid arrays
    RETURNS: calculated distance

calculateDistArr(K, gs, hs, centroids)
    PURPOSE: calculate the distances between points and centroids
    PARAMETERS: number of centroids (K), glucose array, hemoglobin 
    array, and centroid arrays
    RETURNS: array with distances from glucose and hemoglobin points 
    to centroid points

calculateAssignment(distanceArr, K, dataLength)
    PURPOSE: uses the distance array to find the centroid closest to
    each point, assigning it a classification (from 0 to K)
    PARAMETERS: array with distances for each point, number of 
    centroids (K), number of points (length of data taken from file)
    RETURNS: array with assignments for all points

updateCentroids(K, gs, hs, assignment, centroids)
    PURPOSE: Updates the location of each centroid by finding the 
    mean of each feature of the observations (data points) currently 
    assigned to that centroid
    PARAMETERS: number of centroids (K), glucose array, hemoglobin 
    array, and centroids arrays
    RETURNS: updated centroid array

denormalize(centroids, h, g, K)
    PURPOSE: "Denormalizes" the centroids' data. Transform values
    ranging from 0 to 1 into values ranging from max and min (of
    glucose and hemoglobins). 
    PARAMETERS: denormalized centroids array, denormalized hemoglobin
    array and K
    RETURNS: denormalized centroids array

