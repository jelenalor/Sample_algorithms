
#
# Author - JLor
#

import numpy as np

# Input -> categorical variables
# Out put the most likely class as per Naive Bayes theory


class MyNB():
    def __init__(self):
        print("Naive Bayes model")
        
    # Support Functions
    def priors(self, y):
        value, counts = np.unique(y, return_counts = True)
        probabilities = counts / counts.sum()
        probabilities_dict = {}
        counts_dict = {}
        for i, j, k in zip(value, probabilities, counts):
            probabilities_dict[i] = j
            counts_dict[i] = k
        return counts_dict, probabilities_dict
    
    def posteri_counts(self, X, y):
        y_unique = np.unique(y)
        posteri_dict = {}
        # i -> class, j-> col index, k-> value in col j
        for i in y_unique:
            posteri_dict[i] = {}
            # Iterate over columns
            for j in range(X.shape[1]):
                posteri_dict[i][j] = {}
                # Find the list of unique values in the column j corresponding to class i
                unique_values = np.unique(X[y == i][:, j])
                for k in unique_values:
                    posteri_dict[i][j][k] = np.float(X[:, j][y == i][X[:, j][y == i] == k].shape[0])
        return posteri_dict
    
    def fit(self, X, y):
        self.probabilities = {}
        self.y = y
        self.X = X
        self.y_unique = np.unique(y)
        
        self.value_counts = self.posteri_counts(X, y)
        self.priors_counts, self.priors_probabilities = self.priors(y)
    
        
    def predict(self, X_test):
        prediction = np.zeros(X_test.shape[0])
        
        # iterate each sample in test data
        for i in range(X_test.shape[0]):
            # iterate each y class
            y_unique_probabilities = np.zeros(len(self.y_unique))
            
            for j in self.y_unique:
                
                # iterate each column in test data
                posteri_counts_allcols = np.zeros(len(X_test[i]))
                for c in range(len(X_test[i])):
                    try:
                        posteri_counts_allcols[c] = self.value_counts[j][c][X_test[i][c]]
                    except KeyError as e:
                        posteri_counts_allcols[c] = 0
                        
                posteri_probs_allcols = round(np.prod(posteri_counts_allcols / self.priors_counts[j]), 4)
                nb_probability = round(posteri_probs_allcols * self.priors_probabilities[j], 4)
                y_unique_probabilities[j] = nb_probability
                        
            y_pred = self.y_unique[np.argmax(y_unique_probabilities)] 
            prediction[i] = y_pred
            
        return prediction
            