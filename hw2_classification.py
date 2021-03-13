"""
This code implements a k-class Classifier per week 6 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.X for running on Vocareum

Execute as follows:
$ python3 hw2_classification.py X_train.csv y_train.csv X_test.csv
"""

# builtin modules
from __future__ import division
import os
#import psutil
import requests
import sys
import math
from random import randrange
import functools
import operator

# 3rd party modules
import pandas as pd
import numpy as np


def separate_by_class(X_train, y_train,  k_classes:int = 10):
    """
    Separates our training data by class, from the following inputs:
    
        X_train : training dataset features excluding the label (multiple columns)
        y_train : correspoding labels of the training dataset (single column)
        k_classes: number of k classes for the classifier, (10 classes fixed in assignment)

    It returns a dictionary where each key is the class value.

    """
    keys = list(range(k_classes))
    separated = dict([(key, []) for key in keys])

    dataset_train = pd.concat([X_train, y_train], axis = 1)
    for i in range(len(dataset_train)):
        vector = np.array(dataset_train.iloc[i])
        class_value = y_train.iloc[i, 0]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def summarize_dataframe(dataframe, class_value, n_features):

    """
    Calculate the mean, standard deviation and count for each column in the dataframe from the following inputs:
        dataframe : dataset to summarise as a DataFrame
        class_value : the value (label from 0 to 9) of the class being summarised
        n_features : number of features (columns) in the training dataset (X_train + y_train)
    
    It returns a DataFrame of mean, std and count for each column/feature in the dataset. Note that it is prepared to handle an empty dataframe to deal with classes unseen in the training dataset.
    
    """
    # dealing with emtpy dataframes as a result of including for classes unseen in dataset
    if dataframe.shape == (0,0):
        mean = np.append(np.zeros(n_features), [class_value])
        sigma = np.zeros(n_features + 1)
        count = np.zeros(n_features + 1)
    else:
        mean = dataframe.mean(axis=0)
        sigma = dataframe.std(axis=0, ddof=1)  #ddof = 0 to have same behaviour as numpy.std, std takes the absolute value before squaring
        count = dataframe.count(axis=0)
    
    frame = {'mean': mean, 'std': sigma, 'count': count}

    summaries = pd.DataFrame(frame)

    return summaries

def summarize_by_class(X_train, y_train):
    """
    Calculate statistics (mean, stdv, count) for each class subset from the following inputs:
        X_train : training dataset features excluding the label (multiple columns)
        y_train : corresponding labels of the training dataset (single column)

    It first calls the function 'separate_by_class' to split the dataset by class. It then calls the function 'summarize_dataframe' to calculate the statistics for each row. 
    
    It returns a dictionary object where each key is the class value and then a list of all the records as the value in the dictionary.
    """
    
    separated = separate_by_class(X_train, y_train, 10)
    summaries = dict()
    
    for class_value, rows in separated.items():
        # convert class subset lists to a dataframe before passing on to summarize_dataframe
        class_subset = pd.DataFrame(separated[class_value])
        # obtain summary statistics per class subset, note we specify the number of features in the dataframe to be summarised
        summaries[class_value] = summarize_dataframe(class_subset, class_value, len(X_train.columns))

    return summaries

def calculate_probability(x, mean, stdev):
    """
    Calculate the Gaussian probability distribution function for x from inputs:
    
    x: the variable we are calculating the probability for
    mean: the mean of the distribution
    stdev: the standard deviation of the distribution (sigma before squaring)

    It returns the Gaussian probability of a given value based on:
    f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))

    """

    if (mean or stdev) == float(0.0):
        probability = float(0.0)
    else:
        probability = (1 / (math.sqrt(2 * math.pi) * stdev)) * (math.exp(-((x-mean)**2 / (2 * stdev**2 ))))

    return probability

def calculate_class_probabilities(summaries, row):
    """
    Calculate the probability of a value using the Gaussian Probability Density Function from inputs:
    
    summaries: prepared summaries of dataset
    row: a new row

    This function uses the statistics calculated from training data to calculate probabilities for the testing dataset (new data). Probabilities are calculated separately for each class. First, we calculate the probability that a new X vector from the testing dataset belongs to the first class. Then, we calculate the probabilities that it belongs to the second class, and so on for all the classes identified in the training dataset.

    The probability that a new X vector from the testing dataset belongs to a class is calculated as follows:
    P(class|data) = P(X|class) * P(class)
    Note we have simplified the Bayes theorem by removing the division as we do not strictly need a number between 0 and 1 to predict the class the new data belongs to as we will be simply taking the maximum result from the above equation for each class.

    It returns a dictionary where each key is the class label and the values are the probabibilities of that row belonging to each class on the dataset.

    """
    # total number of training records calculated from the counts stored in the summary statistics
    # note that the count column has the same value for all rows, and hence picking up item [0] will suffice
    total_rows = sum([summaries[label]['count'][0] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value]['count'][0]/float(total_rows)
        for i in range(len(class_summaries)-1):
            mean, stdev, _ = class_summaries.iloc[i]
            # probabilities are multiplied together as they accumulate.
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

# normalize probabilities so that they sum 1

    max_prob = probabilities[max(probabilities, key=probabilities.get)]
    min_prob = probabilities[min(probabilities, key=probabilities.get)]

    for class_value, probability in probabilities.items():
        if (max_prob - min_prob) > 0:
            probabilities[class_value] = (probability - min_prob) / (max_prob - min_prob)
        else:
            probabilities[class_value] = float(0.0)

    # divide by the sum of the probabilities to ensure sum of probabilities for all classes is equal to 1.
    sum_prob = sum(probabilities.values())

    for class_value, probability in probabilities.items():
        if sum_prob > 0:
            probabilities[class_value] = probability / sum_prob

    return probabilities


def predict(summaries, row):
    """
    Predict the most likely class from inputs:

    summaries: prepared summaries of dataset
    row: a row in the dataset for predicting its label (a row of X_test)

    This function uses the probabilities calculated from each class via the function 'calculate_class_probabilities' and returns the label with the highest probability.
    
    It returns the maximum likelihood estimate for a given row.

    """
    
    probabilities = calculate_class_probabilities(summaries, row)

    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        print(class_value, probability)
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label, best_prob, probabilities

def write_csv(filename, a, **kwargs):
        # write the outputs csv file
        if 'header' in kwargs:
            header = kwargs['header']
        else:
            header = False
        if 'path' in kwargs:
            filepath = kwargs['path']
        else:
            filepath = os.path.join(os.getcwd(),'datasets','out', filename)

        df = pd.DataFrame(a)
        df1 = df.iloc[0:len(df)-1]
        df2 = df.iloc[[len(df)-1]]
        df1.to_csv(filepath, sep=',', index = False, header = header)
        df2.to_csv(filepath, sep=',', index = False, header = False, mode = 'a', line_terminator = "")

    #print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def pluginClassifier(X_train, y_train, X_test):
    """
    Implements a Bayes Naive Classifier from inputs:
    X_train : training dataset features excluding the label (multiple columns)
    y_train : corresponding labels of the training dataset (single column)
    X_test : testing dataset features excluding the label (multiple columns)

    This function consists of the following main steps:
    Step 1. Get a summary of the statistics of the training dataset (X_train, y_train) pairs combined
    Step 2. Declare empty lists for predictions and probabilities for each row in X_test of belonging to each of the classes present on the dataset
    Step 3. Get the maximum likelihood estimate for each row of belonging to each of the classes, from a given 'summaries' and 

    It returns two lists:
    prediction_outputs: a list of the predicted labels for each row (class of highest probability)
    probabilities_output: a dictionary where each key is the class label and the values are the probabibilities of that row belonging to each class on the dataset.
        
    """

    # Step 1. Get statistics summary on the training dataset
    summaries = summarize_by_class(X_train, y_train)

    # Step 2. create an empty list to store predictions
    predictions = list()
    probabilities_output = list()

    # Step 3. Go through each row in the testing dataset to get the maximum likelihood estimate of the probability of a row to belong to each of the classes
    for i in range(len(X_test)):
        row = X_test.iloc[i] #note how row does not include the label value 'y'
        out_prediction, _, out_probability = predict(summaries, row)
        predictions.append(out_prediction)
        probabilities_output.append(out_probability)

    prediction_outputs = (predictions)
    probabilities_output = (probabilities_output)

    return probabilities_output



def main():

    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    X_train = pd.DataFrame(data=X_train)
    y_train = pd.DataFrame(data=y_train).astype('int32')
    X_test = pd.DataFrame(data=X_test)

    final_outputs = pluginClassifier(X_train, y_train, X_test) # get final outputs
    # write the probability of predicting the class right to a csv
    # note it is important not to write the header into the output csv as Vocareum will throw error
    write_csv("probs_test.csv", final_outputs, header = False, path = os.path.join(os.getcwd(), "probs_test.csv"))

    #np.savetxt("probs_test.csv", final_outputs, fmt='%1.2f', delimiter="\n") # write output to file, note values for fmt and delimiter


if __name__ == '__main__':
    main()

