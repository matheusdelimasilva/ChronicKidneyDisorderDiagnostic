# Machine Learning Project - Functions for Step 2 and 3
# *****************************************
# COLLABORATION STATEMENT:
# https://www.w3schools.com/
# https://numpy.org/
# https://www.askpython.com/python/array/python-add-elements-to-an-array
# *****************************************

#Import statements
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

# *******************************************************************
# FUNCTIONS
# *******************************************************************
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

'''
PURPOSE: normalize a NumPy array (glucose and hemoglobin both will need to be normalized)
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
PURPOSE: create a scatter plot of the glucose (Y) and hemoglobin (X)
        with the points graphed colored based on the classification
PARAMETERS: array of glucose, array of hemoglobins, and 
            its classification
RETURNS: nothing, only makes the graph
'''
def graphData(glucose, hemoglobin, classification):
    length = len(classification)

    # zeroGlucose and zeroHemoglobin arrays will values 
    # for patients with no CKD
    zeroGlucose = []
    zeroHemoglobin = []

    # oneGlucose and oneHemoglobin arrays will hold the values 
    # for patients with CKD
    oneGlucose = []
    oneHemoglobin = []

    for i in range(length):
        if (classification[i] == 0):
            zeroGlucose.append(glucose[i])
            zeroHemoglobin.append(hemoglobin[i])
        else:
            oneGlucose.append(glucose[i])
            oneHemoglobin.append(hemoglobin[i]) 

    plt.scatter(oneGlucose, oneHemoglobin, label="Patients with CKD")
    plt.scatter(zeroGlucose, zeroHemoglobin, label="Patients without CKD")
    plt.xlabel("Glucose levels")
    plt.ylabel("Hemoglobin levels")
    plt.legend()
    plt.title("Patients' glucose and hemoglobin levels")
    plt.show()

'''
PURPOSE:creates a random test case that falls within the
        minimum and maximum values of the training hemoglobin and
        glucose data
PARAMETERS: glucose's min and max value, hemoglobin's min and
            max value
RETURNS: normalized and not-normalized test case
'''
def createTestCase(minH, maxH, minG, maxG):
    print("  -Generating a random test case-  ")
    #generate random test cases for hemoglobin and glucose
    testCaseH = random.randint(int(minH), int(maxH)+1)
    testCaseG = random.randint(int(minG), int(maxG)+1)

    print("Hemoglobin level:", testCaseH)
    print("Glucose level:", testCaseG)

    #normalize random test cases
    newHemoglobin = (testCaseH - minH)/(maxH - minH)
    newGlucose = (testCaseG - minG)/(maxG - minG)

    return newHemoglobin, newGlucose, testCaseH, testCaseG

'''
PURPOSE:Calculates the distance from the test case to each
        point
PARAMETERS: test case's glucose and hemoglobin levels, the 
            array with glucose and hemoglobin levels from
            the patients
RETURNS: an array containing all the distances calculated
'''
def calculateDistanceArray(newGlucose, newHemoglobin, glucose, hemoglobin):
    distanceG = (newGlucose - glucose)**2
    distanceH = (newHemoglobin - hemoglobin)**2
    distanceArray = (distanceG + distanceH)*0.5
    return distanceArray 

'''
PURPOSE: Use the Nearest Neighbor Classifier to classify the 
         test case
PARAMETERS: glucose and hemoglobins data, and 
            their classifications, test case's glucose and
            hemoglobin levels (newGlucose, newHemoglobin)
RETURNS: prints test case's classification
'''
def nearestNeighborClassifier(newGlucose, newHemoglobin, glucose, hemoglobin, classification):
    print("\nNearest Neighbor Classifier:")
    #Get the distance array
    distanceArray = calculateDistanceArray(newGlucose, newHemoglobin, glucose, hemoglobin)

    #Find the point that is closest to the test case
    closest = min(distanceArray)

    #Find the index of the closest point
    closestIdx = np.where(distanceArray == closest)
    print("Closest index:", closestIdx)

    #Get the closest point's classification
    if (classification[closestIdx]):
        print("RESULT: Test case probably has CKD (class = 1)")
    else:
        print("RESULT: Test case probably does not have CKD (class = 0)")
    print("\n")

'''
PURPOSE: Graph the patients' and the test case's glucose and     
         hemoglobin levels
PARAMETERS: glucose and hemoglobins data and 
            their classifications, test case's glucose and
            hemoglobin levels (newGlucose, newHemoglobin)
RETURNS: plots a graph
'''
def graphTestCase (newGlucose, newHemoglobin, glucose, hemoglobin, classification):
    length = len(classification)

    # zeroGlucose and zeroHemoglobin arrays will hold values 
    # for patients with no CKD
    zeroGlucose = []
    zeroHemoglobin = []

    # oneGlucose and oneHemoglobin arrays will hold the values 
    # for patients with CKD
    oneGlucose = []
    oneHemoglobin = []

    for i in range(length):
        if (classification[i] == 0):
            zeroGlucose.append(glucose[i])
            zeroHemoglobin.append(hemoglobin[i])
        else:
            oneGlucose.append(glucose[i])
            oneHemoglobin.append(hemoglobin[i]) 

    plt.scatter(oneGlucose, oneHemoglobin, label="Patients with CKD")
    plt.scatter(zeroGlucose, zeroHemoglobin, label="Patients without CKD")
    plt.scatter(newGlucose, newHemoglobin, label="Test case", edgecolors = "r", color = "r")
    plt.xlabel("Glucose levels")
    plt.ylabel("Hemoglobin levels")
    plt.legend()
    plt.title("Patients' glucose and hemoglobin levels")
    plt.show()

'''
PURPOSE: Use the K-Nearest Neighbor Classifier to classify the 
         test case
PARAMETERS: glucose and hemoglobins data, and 
            their classifications, test case's glucose and
            hemoglobin levels (newGlucose, newHemoglobin)
RETURNS: prints test case's classification
'''
def kNearestNeighborClassifier(k, newGlucose, newHemoglobin, glucose, hemoglobin, classification):
    print("K-Nearest Neighbor Classifier:")
    #Get the distance array
    distanceArray = calculateDistanceArray(newGlucose, newHemoglobin, glucose, hemoglobin)

    #Get all the indices in closest order (closest to furthest)
    allClosestIndexes = np.argsort(distanceArray)

    #Get the k-closest points
    kClosestIndexes = allClosestIndexes[0:k]

    print("K-closest indexes:", kClosestIndexes)

    #Get the classification of the closest points
    kClosestClass = classification[kClosestIndexes]

    #Find the mode of the k closest points
    mode = stats.mode(kClosestClass)

    print("K-closest classifications:", kClosestClass)
    

    #Get the test's classification
    if (mode):
        print("RESULT: Test case probably has CKD (class = 1)")
    else:
        print("RESULT: Test case probably does not have CKD (class = 0)")