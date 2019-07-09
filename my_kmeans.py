# K-MEANS
import random
import numpy as np


# SUPPORT FUNCTIONS

#Function to pick a random value based on probabilities
def randomPick(my_list, probabilities):
    # create a random value between 0 and 1
    x = random.uniform(0, 1)
    # calculate cumulative probabilities
    cumulative_probability = 0.0
    for item, item_probability in zip(my_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: 
            break
    return item

def find_initial_center_plusplus(X, n_clusters):
    cluster_centers = {}
    # Pick the first cluster center randomly from the existing points
    first_centre = X[random.sample(range(X.shape[0]), 1)][0]
    cluster_centers[0] = first_centre
    
    # Pick the rest cluster centers by chance propoertional to its distance to the
    # closest existing cluster
    #
    for i in range(1, n_clusters):
        distances = []
        item_list = []
        for row in range(X.shape[0]):
            # for each sample in the X find the center that gives the lowest distance to
            # that sample and store the distance and the sample item information
            lowest_distance = 100000
            for c in cluster_centers.keys():
                dist = np.sqrt(sum((X[row] - cluster_centers[c])**2))
                if dist < lowest_distance:
                    lowest_distance = dist

            distances.append(lowest_distance)
            item_list.append(row)
        #
        # Pick the next cluster center in proportion to the item's distances to the 
        # current cluster centers
        # which means the items that are the futhers away have higher distance
        # and therefore higher chance of being picked up
        # to be a next cluster center
        #
        distances = np.array(distances)
        probabilities = distances.astype(float)  / sum(distances)
        pick_cluster_index = randomPick(item_list, probabilities)    
        cluster_centers[i] = X[pick_cluster_index]
    return cluster_centers



class MyKmeans():
    def __init__(self, n_clusters, iterations = 100):
        self.n_clusters = n_clusters
        self.iterations = iterations 
      
    def fit_predict(self, X):
        self.X = X
        cluster_points = find_initial_center_plusplus(X, self.n_clusters)
        
        for i in range(self.iterations):
        
            # Update each row in X with closest cluster   
            target = np.zeros(self.X.shape[0])
            for row in range(self.X.shape[0]):
                cluster_keys = list(cluster_points.keys())
                cluster_distances = []
                for cl in cluster_keys:
                    distance = np.sqrt(sum((self.X[row] - cluster_points[cl])**2))
                    cluster_distances.append(distance)
                target[row] = cluster_keys[np.argmin(cluster_distances)]

                #Update clusters with the average of the samples belonging to that cluster
                for c in cluster_points.keys():
                    cluster_points[c] = self.X[target == c].mean(axis = 0)
        return target

    


# CODE TO CHECK
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the data
# iris = load_iris()
# X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
# y = iris.target

# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c = y.astype(np.float))
# plt.xlabel(iris.feature_names[1], fontsize = 10)
# plt.ylabel(iris.feature_names[3], fontsize = 10)
# plt.title("Actual")

# #MY CODE
# km = MyKmeans(X, 3)
# plt.subplot(1, 2, 2)
# plt.scatter(X[:, 0], X[:, 1], c = km.astype(np.float))
# plt.xlabel(iris.feature_names[1], fontsize = 10)
# plt.ylabel(iris.feature_names[3], fontsize = 10)
# plt.title("Predited")

# plt.show()