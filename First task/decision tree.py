# Import necessary libraries
import numpy as np
import pandas as pd
import requests as rq
import joblib as jb
import pickle as pk
from numpy import savetxt, loadtxt
from sklearn import svm
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

## Download the data, split it into training and testing data, and save it as a file 
git = "https://raw.githubusercontent.com/SergUSProject/IntelligentSystemsAndTechnologies/main/Practice/datasets/mnist.csv"
data = rq.get(git)
with open("mnist.csv", "wb") as file:
    file.write(data.content)
dataset = loadtxt(open('mnist.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)
training_dataset, testing_data = train_test_split(dataset, test_size=0.35)
jb.dump(training_dataset, 'training_set.pkl')
training_data = jb.load('training_set.pkl')

## Extract the target variable and the rest of the data from the training set and Save the testing data (without the target variable) to a file test_set.pkl
target_variable = [x[0] for x in training_data]
training_features = [x[1:] for x in training_data]
testing_features_no = np.delete(testing_data, 0, axis=1)
jb.dump(testing_features_no, 'test_set.pkl')
testing_features = jb.load('test_set.pkl')

## Train the decision tree classifier and write the classification result on the saved testing set (test_set.pkl) to a file called answer.csv, also evaluate the classification quality using the accuracy_score metric
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(training_features, target_variable)
testing_features = jb.load('test_set.pkl')
tree_predictions = tree_classifier.predict(testing_features)
savetxt('answer.csv', tree_predictions, delimiter=',', fmt='%d')
testing_target = [x[0] for x in testing_features]
accuracy = accuracy_score(testing_target, tree_predictions)
print("Accuracy:", accuracy)
