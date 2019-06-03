import sys
import math
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from functools import partial
from random import randint


# --------------------------------------------------------------
# --------------------------------------------------------------

def compute_weights(points, centers):
	weights = np.zeros(len(centers))
	for point in points:
		mycenter = 0
		mindist = math.sqrt(point.squared_distance(centers[0]))
		for i in range(1, len(centers)):
			if (math.sqrt(point.squared_distance(centers[i])) < mindist):
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
# ---------- ADD YOUR CODE HERE ---------------

def f1(line):
	return Vectors.dense([float(coord) for coord in line.split(" ")])


def main(argv):
	# Avoided controls on input..
	dataset = argv[1]
	k = int(argv[2])
	L = int(argv[3])
	iterations = int(argv[4])
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
	if (len(sys.argv) != 5):
		print("Usage: <pathToFile> k L iter")
		sys.exit(0)
	main(sys.argv)