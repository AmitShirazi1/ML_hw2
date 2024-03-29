import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """

        self.ids = (319044434, 314779166)
       
       
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        print("Fitting PerceptronClassifier...")

        self.num_classes = len(np.unique(y))
        X_with_bias = np.column_stack((np.ones(len(X)), X))  # Add bias term
        # self.max_iterations = len(X)
        self.num_features_with_bias = X_with_bias.shape[1]
        self.weights = np.zeros((self.num_classes, self.num_features_with_bias), dtype=np.float32) # The W from the pseudo code is initialized to zeros.
        y_pred = np.zeros(len(y), dtype=np.uint8) # We initialize the array of predictions with zeros.

        t = 0
        while not np.array_equal(y_pred, y): # We iterate over the dataset until we reach the maximum number of iterations.    
            y_pred[t] = np.argmax(np.dot(self.weights, X_with_bias[t]))  # We get the index of the max value of dot product which is the predicted class.
            
            if y_pred[t] != y[t]: # If the prediction is wrong, we update the weights according to pseudo code provided.
                self.weights[y[t]] += X_with_bias[t] 
                self.weights[y_pred[t]] -= X_with_bias[t]

            t = (t + 1) % len(X) # We iterate over the dataset in a loop.


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        # Preprocess data
        X_with_bias = np.column_stack((np.ones(len(X)), X))  # Add bias term
        y_pred = np.zeros(len(X), dtype=np.uint8) # We initialize the array of predictions with zeros.

        for i, x in enumerate(X_with_bias):
            y_pred[i] = np.argmax(np.dot(self.weights, x)) # We get the index of the max value of dot product which is the predicted class.
         
        return y_pred


if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
