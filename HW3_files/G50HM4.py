import sys
import math
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from functools import partial
from random import randint
from numpy import random as rn, linalg as lin
import time
from functools import reduce
from operator import add


# --------------------------------------------------------------
# ------- our methods ------------------------------------------
# --------------------------------------------------------------

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
			if len(curr_cluster_points) > 0:
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


# ----------------------------------
# ---- prepared code ---------------
# ----------------------------------

def compute_weights(points, centers):
	weights = np.zeros(len(centers))
	for point in points:
		mycenter = 0
		mindist = math.sqrt(point.squared_distance(centers[0]))
		for i in range(1, len(centers)):
			if math.sqrt(point.squared_distance(centers[i])) < mindist:
				mindist = math.sqrt(point.squared_distance(centers[i]))
				mycenter = i
		weights[mycenter] = weights[mycenter] + 1
	return weights


def f2(k, L, iterations, partition):
	points = [vector for vector in iter(partition)]
	weights = np.ones(len(points))
	centers = kmeansPP(points, weights, k, iterations)
	final_weights = compute_weights(points, centers)
	return [(vect, weight) for vect, weight in zip(centers, final_weights)]


def f3(centers, partition):
	points = [vector for vector in iter(partition)]
	partition_sum = 0
	num_centers = len(centers)
	for vec in points:
		partition_sum += min([lin.norm(vec - centers[i]) for i in range(0, num_centers)])
	return [partition_sum]


def MR_kmedian(pointset, k, L, iterations):
	# ---------- ROUND 1 ---------------
	coreset = pointset.mapPartitions(partial(f2, k, L, iterations))
	# ---------- ROUND 2 ---------------
	centersR1 = []
	weightsR1 = []
	for pair in coreset.collect():
		centersR1.append(pair[0])
		weightsR1.append(pair[1])
	centers = kmeansPP(centersR1, weightsR1, k, iterations)

	# ---------- ROUND 3 --------------------------
	#  ---------- ADDED OUR CODE HERE -------------
	#  (check it please)
	sum_of_distances = pointset.mapPartitions(partial(f3, centers)).sum()
	objective = sum_of_distances / pointset.count()
	return objective



def f1(line):
	return Vectors.dense([float(coord) for coord in line.split(" ") if len(coord) > 0])


def main(argv):
	# Avoided controls on input..
	dataset = "covtype10K.data"  # argv[1]
	k = 20  # int(argv[2])
	L = 8  # int(argv[3])
	iterations = 10  # int(argv[4])
	conf = SparkConf().setAppName('HM4 python Template')
	sc = SparkContext(conf=conf)
	pointset = sc.textFile(dataset).map(f1).repartition(L).cache()
	N = pointset.count()
	print("Number of points is : " + str(N))
	print("Number of clusters is : " + str(k))
	print("Number of parts is : " + str(L))
	print("Number of iterations is : " + str(iterations))

	obj = MR_kmedian(pointset, k, L, iterations)
	print("Objective function is : " + str(obj))


if __name__ == '__main__':
	# for testing purposes no params
	"""if len(sys.argv) != 5:
		print("Usage: <pathToFile> k L iter")
		sys.exit(0)"""
	main(sys.argv)
