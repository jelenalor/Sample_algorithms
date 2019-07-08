
# -- Author -- 
# JLor
#

import numpy as np
import matplotlib.pyplot as plt


class myRegression():
    def __init__(self, alpha, iterations = 1000):
        self.alpha = alpha
        self.iterations = iterations
        self.errors = []
        # Set random weight between 0, 1 vector for each feature in X_Train plus bias
        self.weights = np.insert(np.random.rand(X_train.shape[1]), 0, 1)
        
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # normalize X_train (add 1 for each bias weight)
        X_norm = np.insert(self.X_train, 0, 1, axis=1)
        #1 For each iters go line by line in X_train and estimate 
        
        j = 0
        for i in range(self.iterations):
            y = np.sum(X_norm[j] * self.weights)  
            error = self.y_train[j] - y                 
            self.errors.append(error)
            
            #3 Adjust the weights
            self.weights = self.weights + self.alpha*error*X_norm[j]
            
            #4 Update j if needed
            if j == (self.X_train.shape[0] - 1):
                j = 0
            else:
                j += 1 
        print("The model is fit with alpha %s" %self.alpha)
        
        
    def predict(self, X_test):
        self.X_test = X_test
        # normalize X_test (add 1 for each bias weight)
        X_norm = np.insert(X_test, 0, 1, axis=1)
        self.prediction = np.zeros(self.X_test.shape[0])
        for i in range(X_test.shape[0]):          
            y_pred = np.sum(X_norm[i] * self.weights) 
            self.prediction[i] = y_pred
        return self.prediction
    
    
    def plotErrors(self):
        plt.figure(figsize = (15, 5))
        plt.plot(np.arange(self.iterations), np.abs(self.errors))
        plt.show()
        
        
    def MeanSquaredError(self, y_test):
        self.y_test = y_test
        mean_square_error = np.mean((self.prediction - y_test)**2)
        print("Mean squared error is %.2f" %mean_square_error )
        

# CHECK
# Example code for checking the regression
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split

# # Make dataset
# X, y = make_regression(n_samples=100, n_features=3, noise = 3)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0)

# # Check model
# model = myRegression(alpha = 0.05)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# model.plotErrors()
# model.MeanSquaredError(y_test)
