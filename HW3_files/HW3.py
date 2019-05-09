from pyspark.mllib.linalg import Vectors
import numpy as np
from numpy import random as rn, linalg as lin
import time
from functools import reduce
from operator import add


# TODO: lloyd must also take into account the weights
"""

For groups of 2 students: The iterations of Lloyd's algorithm that you must apply after kmeans++ in method kmeansPP(P, WP, k, iter) must take into account the weights. I recall that the centroid of a cluster C in this case is defined as:

(1/sum_{p in C} w(p)) * sum_{p in C} p*w(p)

This is explained in Slide 24 of the Slides on Clustering Part 2.
"""
# TODO: in the KMeans++ can't pop the elements, it changes the set


# Function given as a starting point, used to load file
def readVectorsSeq(filename):
	file = open(filename, 'r')
	vector_list = []
	for row in file.readlines():
		vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
	return vector_list


# lloyd function
# points: list of points, centers: default centers, iters: number of iterations in lloyd algo
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
				curr_dist = lin.norm(points[point_index] - new_centers[center_index], ord=1)
				if curr_dist < point_min_dist:
					point_cluster_index = center_index
					point_min_dist = curr_dist
			clustered_points[point_index] = point_cluster_index

		# recompute centers
		for center_index in range(center_length):
			# curr_cluster_bool = clustered_points == center_index
			curr_cluster_points = [points[i] for i in range(points_length) if clustered_points[i] == center_index]

			vectors_sum = reduce(add, curr_cluster_points)
			new_center = vectors_sum / len(curr_cluster_points)
			new_centers[center_index] = new_center

	return new_centers


# Sequential KMeans++ function
def kmeansPP(P, WP, k, iter):  # WP must be column vector

	# Set of centers which will be returned on output
	center_set = []

	# Start by picking first center at random
	rand_choice = rn.randint(0, len(P))
	center_set.append(P[rand_choice])

	# Remove such point from initial set and weights set
	# TODO: wrong, editing point set, parameter by reference
	P.pop(rand_choice)
	WP = np.delete(WP, rand_choice)

	# Each iteration we need distance of P to the closest center
	setMinDist = np.zeros(len(P))

	# Initial distance computation
	for index in range(len(P)):
		# Compute new distance, between point and new center
		setMinDist[index] = lin.norm(P[index] - center_set[0], ord=1)

	# Main algorithm loop
	for round in range(0, k-1):  # Only k-1 centers still have to be chosen

		# Compute the P' probabilities
		prob_set = np.multiply(setMinDist, WP)

		# Normalization factor
		prob_set = prob_set/np.dot(setMinDist, WP)

		# Sample random int, use cumsum of probabilities as thresholds
		rand_choice = rn.rand()
		cumProb = np.cumsum(prob_set)

		# Check which center to choose, according to thresholds
		luckyIndex = np.argwhere(cumProb > rand_choice)[0]
		luckyIndex = luckyIndex[0]

		# Add winner to the centers set
		center_set.append(P[luckyIndex])

		# Remove it from other sets
		P.pop(luckyIndex)
		WP = np.delete(WP, luckyIndex)
		setMinDist = np.delete(setMinDist, luckyIndex)

		# Update min distance by checking if closer to new center than before
		for index in range(len(P)):
			# Compute new distance, between point and new center
			newDist = lin.norm(P[index] - center_set[round], ord=1)
			# Update distance if smaller
			if newDist < setMinDist[index]:
				setMinDist[index] = newDist

	start = time.time()
	new_centers = lloyd(P, center_set, iter)
	end = time.time()
	print("time in lloyd", end - start)

	# for now: return precomputed and after lloyd to see the difference
	return center_set, new_centers


# KMedian cost function, P set points, C set centers
# Assumes both represented as lists of DenseVectors
def kmeansObj(P, C):
	# Scan through all points, set distance as min among the centers
	# Keep just cumulative distance, since output is just average

	sum_of_distances = 0

	# Scan through points
	for point in P:

		# Initialize as cost between point and first center
		point_cost = lin.norm(point - C[0], ord=1)

		for center_index in range(1, len(C)):
			# If better cost, update
			if lin.norm(point - C[center_index], ord=1) < point_cost:
				point_cost = lin.norm(point - C[center_index], ord=1)

		sum_of_distances = sum_of_distances + point_cost

	return sum_of_distances/len(P)


# --------------------------------

filename = "covtype10K.data"  # file with the data
k = 5  # number of centers
iterations = 4  # number of iterations

vectorData = readVectorsSeq(filename)  # read the file containing the data
# vectorData is a list of linalg.Vectors

testOutput, test_new_output = kmeansPP(vectorData, np.ones(len(vectorData)), k, iterations)

# test the result:
testCost = kmeansObj(vectorData, testOutput)
test_new_cost = kmeansObj(vectorData, test_new_output)

print("Average distance of a point from set of centers: " + str(testCost))
print("Average distance of a point from set of centers: " + str(test_new_cost))






# -----------------------
# testing, delete after

errors = 0
maxTrials = 100

for trials in range(maxTrials):
	randCenters = []
	for center_index in range(k):
		temp = vectorData[rn.randint(0, len(vectorData))]
		randCenters.append(Vectors.dense(temp))
	test_shit = lloyd(vectorData, randCenters, iterations)
	test_without_kmeanspp = kmeansObj(vectorData, test_shit)
	print(test_without_kmeanspp)
	if test_without_kmeanspp < test_new_cost:
		errors += 1

print("errors: ", errors, " of: ", maxTrials)
