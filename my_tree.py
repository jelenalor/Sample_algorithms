
# 
# Author - JLor
#

import numpy as np

# Input works best for categorical values
# WIP
# Ideas for improvement:
# Adjust the code to work for continuous numeric features
# 

# SUPPORT FUNCTIONS

# Returns items from the list in proportion to their frequency in the list
def get_result_proportionally(my_list):
    values, counts = np.unique(my_list, return_counts = True)
    proportions = counts / np.float(np.sum(counts))

    # Find cumulative proportion of items appearing in the list
    cumulative_proportions = []
    for index in range(len(proportions)):
        if index > 0:
            (cumulative_proportions.append(proportions[index] + 
                                        + cumulative_proportions[index-1]))
        else:
            cumulative_proportions.append(proportions[index])

    random_number = random.random()
    for index in range(len(proportions)):
        # If random generated number between 0-1 is less than first item in 
        # cumulative proportions list, than choose first item in my_list
        if index == 0 and random_number < cumulative_proportions[index]:
            return values[index]

        elif index > 0:  
            # Check where random numbers falls to and choose the item accordingly. 
            # For example, if cumulative proportions are 0.50,0.30,0.20 for items 
            # A,B,C -> there are 50% chance that random number is less than 0.5 
            # and item A is chosen

            if (random_number < cumulative_proportions[index] and 
                random_number > cumulative_proportions[index-1]):
                return values[index]


# Chooses the best feature based on the lowest Gini
def choose_feature(data, target):
    gini_results = []
    for feature in range(data.shape[1]):
        gini_results.append(get_gini_score(data, feature, target))
    best_feature = np.argmin(gini_results)
    return best_feature


# Calculates the proportions of each unique category within the specified feature
def get_proportions(data, feature):
    proportions = {}
    values, counts = np.unique(data[:, feature], return_counts = True)
    for value, count in zip(values, counts):
        proportions[value] = count / np.float(np.sum(counts))
    return proportions


# Counts number of instances for each combination of category and target values in 
# specificed feature, and stores in the dictionary of dictionaries
def get_feature_counts(data, feature, target):
    feature_counts = {}
    for category in np.unique(data[:, feature]):
        target_subset = target[data[:, feature] == category]
        feature_counts[category] = {}

        for result in np.unique(target_subset):
            feature_counts[category][result] = np.sum(target_subset == result)
    return feature_counts


# Get impurity of specified category within a specified feature
def get_impurity(data, feature, category, target):
    feature_counts = get_feature_counts(data, feature, target)
    impurity = 0
    for result in feature_counts[category].keys():
        #Impurity Formula
#         print("Feat counts", feature_counts)
#         print("Category", category)
#         print("Feat counts", np.sum(np.array(feature_counts[category].values())))
        impurity += (feature_counts[category][result] / np.float(np.sum(list(feature_counts[category].values()))))**2
    impurity = 1 - impurity
    return impurity


# Get impurity of a specified feature
def get_gini_score(data, feature, target):
    # Get the dictionary with all counts for each combination of category and target 
    # for specified feature
    feature_counts = get_feature_counts(data, feature, target)
    # Get proportions of all categories within a specified feature
    proportions = get_proportions(data, feature)
    gini_score = 0

    for category in feature_counts.keys():
        gini = get_impurity(data, feature, category, target)
        # Gini_score is a weighted average of impurities for all categories 
        # within the feature
        gini_score+= gini * proportions[category]
    return gini_score


#  Find category with lowest impurity
def lowest_impurity_category(data, feature, target):
    # Deliberetly initiate lowest_impurity with very high value 
    # (As we try to find the lowest)
    lowest_impurity = 100
    lowest_impurity_category = " "
    for category in np.unique(data[:, feature]):
        impurty = get_impurity(data, feature, category, target)
        if impurty < lowest_impurity:
            # If found lower impurity -> update lowest_impurity and 
            # lowest_impurity_category
            lowest_impurity = impurty
            lowest_impurity_category = category
    return lowest_impurity_category


# Split data into Left (if categories agree) and Right (if categories do not agree)
def split_data(data, target, feature, category, answer):
    if answer == "yes":
        left_data = data[data[:, feature] == category]
        left_target = target[data[:, feature] == category]
        return left_data, left_target

    elif answer == "no":
        right_data = data[data[:, feature] != category]
        right_target = target[data[:, feature] != category] 
        return right_data, right_target


# CREATE TREE
class MyTree():
    def __init__(self, depth, leaf_split = 4):
        # Depth parametor sets the maximum number of recursions when creating 
        # the tree (or number of times we split the data)
        self.depth = depth
        # Leaf_split defines minimum items in the list when leaf is created
        self.leaf_split = leaf_split


    def fit(self, X_train, y_train, iteration = 0):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # If all results are equal, return the result
        if len(np.unique(y_train)) == 1:
            return y_train[0]

        # Creates a leaf when the minimum number of items in the list reached
        # Helps to not overfit the tree
        elif len(y_train) < self.leaf_split:
            # Return list of results so that at a 'predict' stage we can 
            # return prediction proportionally
            # Since the size of training data is small, we can not be certain 
            # that in this scenario the majority feature is the best prediction
            # Therefore, it makes the model more flexible by allowing it to 
            # choose any of the results in the list, proportionally
            return list(y_train)

        # Creates leaf when the maximum depth of the tree is reached. 
        # This helps not to overfit the tree
        elif iteration == self.depth:
            # The same motivation as described above
            return list(y_train)

        # If none of the above are satisfied, the tree is created recursively
        elif iteration < self.depth:
            best_feature = choose_feature(X_train, y_train)
            category = lowest_impurity_category(X_train, best_feature, y_train)
            split = (best_feature, category)
            # Create dictionary sub_tree to store the result of next recursion
            sub_tree = {tuple(split): []}

            # Split the data --> all data with chosen feature's category 
            # goes to "left" dataset, the rest go to "right" dataset
            left_data, left_target = split_data(X_train, y_train, best_feature, category, "yes")
            right_data, right_target = split_data(X_train, y_train, best_feature, category, "no")

            if len(left_target) != 0:
                sub_tree[split].append(self.fit(left_data, left_target, iteration+1))
            else:
                sub_tree[split].append(get_result_proportionally(y_train))

            if len(right_target) != 0:
                sub_tree[split].append(self.fit(right_data, right_target, iteration+1))
            else:
                sub_tree[split].append(get_result_proportionally(y_train))
            return sub_tree


    # Define a predict method
    def predict_one(self, X_test, tree):
        # Recursively look through the dictionary of dictionaries/lists until reach a 
        # base case -> a list of possible results or the final result
        #BASE CASES
        # Check if reached the leaf of the tree and the result is 
        # definitive (not a list), return the result
        
        if type(tree) != dict and type(tree) != list:
            return tree    

        # Check if reached the leaf of the tree and if the result is not definitive 
        # (list of possible results) choose the result proportionally
        if type(tree) != dict and type(tree) == list:
            return get_result_proportionally(tree)

        # if base case not reached -> Recurse through the tree
        else:
            root = list(tree.keys())[0]
            feature = root[0]
            category = root[1]
            if X_test[feature] == category:
                # If categories agree, reach the next dictionary by 
                # following the first item in the list
                prediction = self.predict_one(X_test, tree[root][0])
            else:
                # If categories do not agree, reach the next dictionary by 
                # following the second item in the list
                prediction = self.predict_one(X_test, tree[root][1])
            return prediction
        
        
    def predict(self, X_test, tree):
        prediction = []
        for i in range(X_test.shape[0]):
            prediction.append(self.predict_one(X_test[i], tree))
        return prediction


# Tree 'depth' and 'leaf_split' parametors allows a user to control 
# how generalised the tree is.
# To make the Tree more generalised decrease a 'depth' parametor 
# and/or increase a 'leaf_split' parametor. 


# To run
# Input -> currently works only for categorical variable input and not continues numerical
#

# tree = MyTree(depth = 5, leaf_split = 4)
# model = tree.fit(data, target)
# pred = tree.predict(data, model)