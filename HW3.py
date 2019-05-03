# Import needed linear algebra libraries
from pyspark.mllib.linalg import Vectors # Functions
from numpy import random as rn


## Needed Functions ##

# Function given as a starting point, used to load file
def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list

# Sequential KMeans++ function
def kmeansPP(P,WP,k,iter):

	# Set of centers which will be returned on output
	setCenters = []
	# Each iter we need distance points to closest center
	setMinDist = np.full((len(P)), np.inf) # Initially distances all +inf

	# Start by picking first center at random
	randChoice = rn.randint(1, len(P) + 1)
	setCenters.append(P[randChoice])
	# Remove such point from initial set
	P.pop(randChoice)

	# Main algorithm loop
	for round in range(1, k-1): # Only k-1 centers still have to be chosen

		# Each iter we need distance points to closest center
		# keep track of the min one, update it only if such value decreases
		for index in range(1, len(P)):
			# Compute new distance, between point and new center
			newDist = np.norm(P[index] - setCenters[round])
			# Update distance if smaller
			if(newDist < setMinDist[index]):
				setMinDist[index] = newDist


		# Initialize probabilities bucket
		setProb = np.zeros(len(P), 1)
		# Compute the points' probabilities
		setProb = 
		# Normalization factor




	return setCenters


# Read the file containing the data
# For testing purposed, pre-set the filename
filename = "covtype10K.data"
vectorData = readVectorsSeq(filename)
# Now vectorData is a list of linalg.Vectors


# test = kmeansPP(vectorData, np.)