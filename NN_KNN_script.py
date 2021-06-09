# Machine Learning Project - Step 2 and 3
# *****************************************
# YOUR NAME: Matt de Lima Silva
# NUMBER OF HOURS TO COMPLETE: 3
# YOUR COLLABORATION STATEMENT(s):
# https://www.w3schools.com/
# https://numpy.org/
# https://www.askpython.com/python/array/python-add-elements-to-an-array
# *****************************************

import NN_KNN_functions as myfunctions
import numpy as np

# *******************************************************************
# DRIVER / SCRIPT
# *******************************************************************

# Get data from file
filename = "ckd.csv"
data = np.genfromtxt(filename, delimiter=",", dtype="float", skip_header=1)

#get glucose, hemoglobin and class from file 
glucose = data[:,0]
hemoglobin = data[:,1]
classification = data[:,2]

#graph the non-normalized data  
print("\n Graphing initial data\n")
myfunctions.graphData(glucose, hemoglobin, classification)

#normalize the data and get min and max
scaledG, minG, maxG = myfunctions.normalizeData(glucose)
scaledH, minH, maxH = myfunctions.normalizeData(hemoglobin)

#Create a test case 
#newHemoglobin and newGlucose are normalized 
#testCaseH and testCaseG are not normalized
newHemoglobin, newGlucose, testCaseH, testCaseG = myfunctions.createTestCase(minH, maxH, minG, maxG)


#Use the Nearest Neighbor Classification to classify the test case
myfunctions.nearestNeighborClassifier(newGlucose, newHemoglobin, scaledG, scaledH, classification)

#Set a value for K
K = 3
#Use the K-Nearest Neighbor Classification to classify the test case
myfunctions.kNearestNeighborClassifier(K, newGlucose, newHemoglobin, scaledG, scaledH, classification)

#Graph the test case
print("\n Graphing data with test case\n")
myfunctions.graphTestCase (testCaseG, testCaseH, glucose, hemoglobin, classification)