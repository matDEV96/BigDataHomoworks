from pyspark.mllib.linalg import Vectors
import numpy as np
from numpy import random as rn, linalg as lin
import time
from functools import reduce
from operator import add


# load file to the list of linalg.Vectors
def readVectorsSeq(filename):
	file = open(filename, 'r')
	vector_list = []
	for row in file.readlines():
		vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
	return vector_list


# lloyd function
# can take long time, e.g. 30 seconds for k=50, iter=3, |points|=10000
# points: list of points, centers: default centers, iters: number of iterations in lloyd algo
def lloyd(points, point_weights, centers, iters):
	points_length = len(points)
	center_length = len(centers)

	new_centers = centers.copy()
	clustered_points = np.zeros(points_length)

	for i in range(iters):
		# compute the centers belonging to the points
		for point_index in range(points_length):  # for each point
			distances_from_centers = np.zeros(center_length)
			for center_index in range(center_length):  # for each center compute the dist from the point
				distances_from_centers[center_index] = lin.norm(points[point_index] - new_centers[center_index])
			clustered_points[point_index] = np.argmin(distances_from_centers)  # pick the minimal dist

		# recompute centers
		for center_index in range(center_length):  # for each cluster
			curr_cluster_points = [points[i] for i in range(points_length) if clustered_points[i] == center_index]
			curr_cluster_point_weights = [point_weights[i] for i in range(points_length) if clustered_points[i] == center_index]

			# compute the new center
			vectors_sum = reduce(add, curr_cluster_points)
			new_center = vectors_sum / sum(curr_cluster_point_weights)
			new_centers[center_index] = new_center
	return new_centers


# Sequential KMeans++ function
def kmeansPP(P, WP, k, iter):  # point_weights must be column vector
	print("  started kmeans++ precomputation")

	points = P.copy()  # copy so we can pop
	point_weights = WP.copy()  # copy so we can pop
	center_set = []  # Set of centers that will be returned

	# Start by picking first center at random
	rand_choice = rn.randint(0, len(points))
	center_set.append(points[rand_choice])

	# Remove such point from initial set and weights set
	points.pop(rand_choice)
	point_weights = np.delete(point_weights, rand_choice)

	# Each iteration we need distance of points to the closest center
	setMinDist = np.zeros(len(points))

	# Initial distance computation
	for index in range(len(points)):
		# Compute new distance, between point and new center
		setMinDist[index] = lin.norm(points[index] - center_set[0])

	# Main algorithm loop
	for round in range(0, k-1):  # Only k-1 centers still have to be chosen

		# Compute the points' probabilities
		prob_set = np.multiply(setMinDist, point_weights)

		# Normalization factor
		prob_set = prob_set/np.dot(setMinDist, point_weights)

		# Sample random int, use cumsum of probabilities as thresholds
		rand_choice = rn.rand()
		cumProb = np.cumsum(prob_set)

		# Check which center to choose, according to thresholds
		luckyIndex = np.argwhere(cumProb > rand_choice)[0]
		luckyIndex = luckyIndex[0]

		# Add winner to the centers set
		center_set.append(points[luckyIndex])

		# Remove it from other sets
		points.pop(luckyIndex)
		point_weights = np.delete(point_weights, luckyIndex)
		setMinDist = np.delete(setMinDist, luckyIndex)

		# Update min distance by checking if closer to new center than before
		for index in range(len(points)):
			# Compute new distance, between point and new center
			newDist = lin.norm(points[index] - center_set[round])
			# Update distance if smaller
			if newDist < setMinDist[index]:
				setMinDist[index] = newDist

	print("  finished precomputation, starting lloyd (it might take a while)")
	start = time.time()
	new_centers = lloyd(P, WP, center_set, iter)
	end = time.time()
	print("  lloyd finished, time spent: ", np.round(end - start, 2), "s")

	# for now: return precomputed and after lloyd to see the difference
	return new_centers


# KMedian cost function, P set points, C set centers
# Assumes both represented as lists of DenseVectors
def kmeansObj(P, C):
	# Scan through all points, set distance as min among the centers
	# Keep just cumulative distance, since output is just average

	print("  counting the average distance of the points to the centers")

	sum_of_distances = 0

	# Scan through points
	for point in P:

		# Initialize as cost between point and first center
		point_cost = lin.norm(point - C[0])

		for center_index in range(1, len(C)):
			# If better cost, update
			if lin.norm(point - C[center_index]) < point_cost:
				point_cost = lin.norm(point - C[center_index])

		sum_of_distances = sum_of_distances + point_cost

	return sum_of_distances/len(P)


# ----------------------
# ---- main program ----
# ----------------------

filename = input("Enter the name of the file:")
try:
	vectorData = readVectorsSeq(filename)
except FileNotFoundError:
	print("Unable to open the file.")
	exit(-1)

k_try = input("Enter the number of centers:")
try:
	k = int(k_try)
	assert(k > 0), 'number must be bigger than 0'
except Exception:
	print("Wrong number, insert integer bigger than zero.")
	exit(-1)


iterations_try = input("Enter the number of partitions:")
try:
	iterations = int(iterations_try)
except ValueError:
	print("Wrong number, insert integer bigger than or equal to zero.")
	exit(-1)

result_centers = kmeansPP(vectorData, np.ones(len(vectorData)), k, iterations)
avg_dist_results = kmeansObj(vectorData, result_centers)
print("Avg dist for: iterations =", iterations, "; k =", k, "    -> ", avg_dist_results)
