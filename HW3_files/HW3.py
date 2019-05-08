from pyspark.mllib.linalg import Vectors
import numpy as np
from numpy import random as rn, linalg as lin
import time
from functools import reduce
from operator import add


# Function given as a starting point, used to load file
def readVectorsSeq(filename):
	file = open(filename, 'r')
	vector_list = []
	for row in file.readlines():
		vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
	return vector_list


# lloyd function
# takes too much time..?
# points: list of points, centers: precomputed centers, iters: number of iterations in lloyd algo
def lloyd(points, centers, iters):
	points_length = len(points)
	center_length = len(centers)

	new_centers = centers.copy()
	clustered_points = np.zeros(points_length)

	for i in range(iters):
		# compute the centers belonging to the points
		for point_index in range(points_length):
			point_cluster_index = -1
			point_min_dist = np.inf
			for center_index in range(center_length):
				curr_dist = lin.norm(points[point_index] - new_centers[center_index])
				if curr_dist < point_min_dist:
					point_cluster_index = center_index
					point_min_dist = curr_dist
			clustered_points[point_index] = point_cluster_index

		# recompute centers
		for center_index in range(center_length):
			curr_cluster_bool = clustered_points == center_index
			curr_cluster_points = [points[i] for i in range(points_length) if curr_cluster_bool[i]]

			vectors_sum = reduce(add, curr_cluster_points)
			new_center = vectors_sum / len(curr_cluster_points)
			new_centers[center_index] = new_center

	return new_centers


# Sequential KMeans++ function
def kmeansPP(points, point_weights, k, iter):  # point_weights must be column vector

	# Set of centers which will be returned on output
	setCenters = []

	# Start by picking first center at random
	rand_choice = rn.randint(0, len(points))
	setCenters.append(points[rand_choice])

	# Remove such point from initial set and weights set
	points.pop(rand_choice)
	point_weights = np.delete(point_weights, rand_choice)

	# Each iteration we need distance points to closest center
	setMinDist = np.zeros(len(points))

	# Initial distance computation
	for index in range(len(points)):
		# Compute new distance, between point and new center
		setMinDist[index] = lin.norm(points[index] - setCenters[0])

	# Main algorithm loop
	for round in range(0, k-1):  # Only k-1 centers stil have to be chosen

		# Compute the points' probabilities
		setProb = np.multiply(setMinDist, point_weights)
		# Normalization factor
		setProb = setProb/np.dot(setMinDist, point_weights)  # Possible errors, in case specify first row

		# Sample random int, use cumsum of pr.ties as tresholds
		rand_choice = rn.rand()
		cumProb = np.cumsum(setProb)

		# Check which center to choose, according to tresholds
		luckyIndex = np.argwhere(cumProb > rand_choice)[0]
		luckyIndex = luckyIndex[0]
		# Add winner to the centers set
		setCenters.append(points[luckyIndex])
		# Remove it from other sets
		points.pop(luckyIndex)
		point_weights = np.delete(point_weights, luckyIndex)
		setMinDist = np.delete(setMinDist, luckyIndex)

		# Update min distance by checking if closer to new center than before
		for index in range(len(points)):
			# Compute new distance, between point and new center
			newDist = lin.norm(points[index] - setCenters[round])
			# Update distance if smaller
			if newDist < setMinDist[index]:
				setMinDist[index] = newDist

	start = time.time()
	new_centers = lloyd(points, setCenters, iter)
	end = time.time()
	print("time in lloyd", end - start)

	# for now: return precomputed and after lloyd to see the difference
	return setCenters, new_centers


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

# I think this test is kind of useless, it was useful just in the beginning, right?
"""
#####################################
# Just testing set of x points in R2
x = 20000
testWeights = np.ones(x)
testPoints = []
for k in range(x):
	# Create point
	testVector = [rn.uniform(), rn.uniform()]
	testPoints.append(Vectors.dense(testVector))

testOutput, testOutputeNewOne = kmeansPP(testPoints, testWeights, 3, 3)
#####################################
"""

# Test on the proper dataset 
k = 5
iterations = 10
testOutput, test_new_output = kmeansPP(vectorData, np.ones(len(vectorData)), k, iterations)
testCost = kmeansObj(vectorData, testOutput)
test_new_cost = kmeansObj(vectorData, test_new_output)
print("Average distance of a point from set of centers: " + str(testCost))
print("Average distance of a point from set of centers: " + str(test_new_cost))

# Check if it works: is it better than randomly picking the points as centers?
# WARNING: Not particularly relevant without refinement, probably


suspiciousTrials = 0
maxTrials = 100

# I don't really understand this :-o
for trials in range(maxTrials):
	# Pick k random centers, compute distance
	randCenters = []
	for center in range(k):
		temp = vectorData[rn.randint(0, len(vectorData))]
		randCenters.append(Vectors.dense(temp))
	if kmeansObj(vectorData, randCenters) < testCost:
		suspiciousTrials = suspiciousTrials + 1

print("We should be " + str(suspiciousTrials/maxTrials*100) + "% worried!")

