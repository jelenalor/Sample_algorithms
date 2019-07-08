
# -- Author -- 
# JLor
#

import numpy as np
import matplotlib.pyplot as plt

class myKNN():
    def __init__(self, neighbours):
        self.neighbours = neighbours  
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, test):
        prediction = np.zeros(test.shape[0])
        for i in range(test.shape[0]):
            targets = []
            #collect distances for each test point
            distances = []                            
            for j in range(self.X_train.shape[0]):   
                target = self.y_train[j]                     
                distance = np.sqrt(np.sum((test[i] - self.X_train[j])**2))   
                targets.append(target)
                distances.append(distance)
                
            distances, targets = zip(*sorted(zip(distances, targets)))
            # Extract closest neigbours
            n_targets = list(targets)[:self.neighbours]
             
            #Find the one appearing more often
            most_frequent = np.unique(n_targets, return_counts=True)[0][np.argmax(np.unique(n_targets, return_counts=True)[1])]
            prediction[i] = most_frequent
        if len(prediction) == 1:
            prediction = prediction[0]
            
        return prediction
    
    
# CHECK
#Example code for running the model

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 0)
# k = myKNN(5)
# k.fit(X_train, y_train)
# pred = k.predict(X_test)