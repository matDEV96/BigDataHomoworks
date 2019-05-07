from pyspark.mllib.linalg import Vectors  # Functions
import numpy as np
from numpy import random as rn, linalg as lin
import time  # Performance tests


# Needed Functions #


# Function given as a starting point, used to load file
def readVectorsSeq(filename):
	file = open(filename, 'r')
	vector_list = []
	for row in file.readlines():
		vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
	return vector_list


# Sequential KMeans++ function
def kmeansPP(P, WP, k, iter):  # WP must be column vector

	# Set of centers which will be returned on output
	setCenters = []

	# Start by picking first center at random
	randChoice = rn.randint(0, len(P))
	setCenters.append(P[randChoice])

	# Remove such point from initial set and weights set
	P.pop(randChoice)
	WP = np.delete(WP, randChoice)

	# Each iteration we need distance points to closest center
	setMinDist = np.zeros(len(P))

	# Initial distance computation
	for index in range(len(P)):
		# Compute new distance, between point and new center
		setMinDist[index] = lin.norm(P[index] - setCenters[0])

	# Main algorithm loop
	for round in range(0, k-1):  # Only k-1 centers stil have to be chosen

		# Compute the points' probabilities
		setProb = np.multiply(setMinDist, WP)
		# Normalization factor
		setProb = setProb/np.dot(setMinDist, WP)  # Possible errors, in case specify first row

		# Sample random int, use cumsum of pr.ties as tresholds
		randChoice = rn.rand()
		cumProb = np.cumsum(setProb)

		# Check which center to choose, according to tresholds
		luckyIndex = np.argwhere(cumProb > randChoice)[0]
		luckyIndex = luckyIndex[0]
		# Add winner to the centers set
		setCenters.append(P[luckyIndex])
		# Remove it from other sets
		P.pop(luckyIndex)
		WP = np.delete(WP, luckyIndex)
		setMinDist = np.delete(setMinDist, luckyIndex)

		# Update min distance by checking if closer to new center than before
		for index in range(len(P)):
			# Compute new distance, between point and new center
			newDist = lin.norm(P[index] - setCenters[round])
			# Update distance if smaller
			if(newDist < setMinDist[index]):
				setMinDist[index] = newDist


	# Refine centers set using Lloyd's algorithm
	### STILL TO DO ###

	return setCenters


# KMedian cost function, P set points, C set centers
# Assumes both represented as lists of DenseVectors
def kmeansObj(P, C):
	# Scan through all points, set distance as min among the centers
	# Keep just cumulative distance, since output is just average

	# Initialize avge cost
	avgeCost = 0

	# Scan through points
	for point in P:
		# Initialize as cost between point and first center
		pointCost = lin.norm(point - C[0])
		for centerIndex in range(1, len(C)):
			# If better cost, update
			if lin.norm(point - C[centerIndex]) < pointCost:
				pointCost = lin.norm(point - C[centerIndex])

		avgeCost = avgeCost + pointCost

	return avgeCost/len(P)


# Actual program

# Read the file containing the data
# For testing purposed, pre-set the filename
filename = "covtype10K.data"
vectorData = readVectorsSeq(filename)
# Now vectorData is a list of linalg.Vectors

#####################################
# Just testing set of x points in R2
x = 20000
testWeights = np.ones(x)
testPoints = []
for k in range(x):
	# Create point
	testVector = [rn.uniform(), rn.uniform()]
	testPoints.append(Vectors.dense(testVector))

testOutput = kmeansPP(testPoints, testWeights, 3, 3)
#####################################


# Test on the proper dataset 
k = 5
iterations = 10
testOutput = kmeansPP(vectorData, np.ones(len(vectorData)), k, iterations)
testCost = kmeansObj(vectorData, testOutput)
print("Average distance of a point from set of centers: " + str(testCost))

# Check if it works: is it better than randomly picking the points as centers?
# WARNING: Not particularly relevant without refinement, probably

suspiciousTrials = 0
maxTrials = 100

for trials in range(maxTrials):
	# Pick k random centers, compute distance
	randCenters = []
	for center in range(k):
		temp = vectorData[rn.randint(0, len(vectorData))]
		randCenters.append(Vectors.dense(temp))
	if kmeansObj(vectorData, randCenters) < testCost:
		suspiciousTrials = suspiciousTrials + 1

print("We should be " + str(suspiciousTrials/maxTrials*100) + "% worried!")
