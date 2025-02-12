﻿"""
This code implements a k-class Classifier per week 6 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.7
"""

# builtin modules
import os
#import psutil
import requests
import sys
import math
from random import randrange
import functools

# 3rd party modules
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

from matplotlib import pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def separate_by_class(X_train, y_train, k_classes:int = 10) -> dict():
    """
    Separates our training data by class, from the following inputs:
    
        X_train : training dataset features excluding the label (multiple columns)
        y_train : correspoding labels of the training dataset (single column)
        k_classes: number of k classes for the classifier, (10 classes fixed in assignment)
    It returns a dictionary where each key is the class value.

    """
    keys = list(range(k_classes))
    separated = dict([(key, []) for key in keys])

    dataset_train = X_train.join(y_train)
    for i in range(len(dataset_train)):
        vector = dataset_train.iloc[i].to_numpy()
        class_value = y_train.iloc[i]
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
    
    It returns a DataFrame of mean, std and count for each column/feature in the dataset. The number of features is used to populate the mean, stdv and coun figures for the unseen classes in the training dataset for the number of classes specify in k_classes.

    """
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

def summarize_by_class(X_train, y_train) -> dict():
    """
    Calculate statistics (mean, stdv, count) for each class subset from the following inputs:
        X_train : training dataset features excluding the label (multiple columns)
        y_train : corresponding labels of the training dataset (single column)

    It first calls the function 'separate_by_class' to split the dataset by class. It then calls the function 'summarize_dataframe' to calculate the statistics for each row. 
    
    It returns a dictionary object where each key is the class value and then a list of all the records as the value in the dictionary.
    """
    
    separated = separate_by_class(X_train, y_train)
    summaries = dict()
    
    for class_value, rows in separated.items():
        # convert class subset lists to a dataframe before passing on to summarize_dataframe
        class_subset = pd.DataFrame(separated[class_value])
        # obtain summary statistics per class subset, note we specify the number of features in the dataframe to be summarised
        summaries[class_value] = summarize_dataframe(class_subset, class_value, len(X_train.columns))

    return summaries

def calculate_probability(x, mean, stdev) -> float:
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

def calculate_class_probabilities(summaries, row) -> dict():
    """
    Calculate the probability of a value using the Gaussian Probability Density Function from inputs:
    
    summaries: prepared summaries of dataset
    row: a row in the dataset for predicting its label (a row of X_test)

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
        if (max_val - min_val) > 0:
            probabilities[class_value] = (probability - min_val) / (max_val - min_val)
        else:
            probabilities[class_value] = float(0.0)

    sum_prob = sum(probabilities.values())

    for class_value, probability in probabilities.items():
        if sum_prob > 0:
            probabilities[class_value] = probability / sum_prob

    return probabilities

def predict(summaries, row) -> tuple:
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

def pluginClassifier(X_train, X_test, y_train):
    """
    Implements a Bayes Naive Classifier from inputs:
    X_train : training dataset features excluding the label (multiple columns)
    X_test : testing dataset features excluding the label (multiple columns)
    y_train : corresponding labels of the training dataset (single column)

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

    return prediction_outputs, probabilities_output

def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right


def corrplot(data, size_scale:int = 500, marker:str = 's'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )


def get_data(source_file, **kwargs):
    """
    Read data from a file given its name. Option to provide the path to the file if different from: [./datasets/in]

    """
    # Define input and output filepaths
    input_path = os.path.join(os.getcwd(),'datasets','in', source_file)

    if 'col_titles' in kwargs:
        # Read input data
        df = pd.read_csv(input_path, names = kwargs['col_titles'])
    else:
        # Read input data
        df = pd.read_csv(input_path)
       
    return df


def split_data(df, ratio:float = 0.7):

    """
    Splits the data set into the training and testing datasets from the following inputs:

    df: dataframe of the dataset to split
    ratio : percentage by which to split the dataset; ratio = training data / all data;
    e.g. a ratio of 0.70 is equivalent to 70% of the dataset being dedicated to the training data and the remainder (30%) to testing.

    Outputs: X_train, y_train, X_test, y_test
   
    X_train: Each row corresponds to a single vector  xi.
    y_train: Each row has a single number and the i-th row of this file combined with the i-th row of "X_train" constitutes the training pair (yi,xi).
    X_test: The remainder of the dataset not included already in the X_train dataframe. Same format as "X_train".
    y_test: The remainder of the dataset not included already in teh y_train dataframe. Same format as "y_train".

    """

    rows, cols = df.shape
    rows_split = int(ratio * rows)

    # split the dataset into train and test sets
    # drop last column of X which will become 'y' vector
    df_X_train = df[df.columns[:-1]].loc[0 : rows_split]
    df_X_test = df[df.columns[:-1]].loc[rows_split : rows]

    # get the last column of X as the 'y' vector and split it into train and test sets
    df_y_train = df[df.columns[cols - 1]].loc[0 : rows_split] 
    df_y_test = df[df.columns[cols - 1]].loc[rows_split : rows]

    return df_X_train, df_X_test, df_y_train, df_y_test


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
        df.to_csv(filepath, index = False, header = header)
        return print("New Outputs file saved to: <<", filename, ">>", sep='', end='\n')


def check_results(df_y_test, final_outputs):
    score = 0
    check = []
    
    for i in range(len(final_outputs)):
        if final_outputs[i] == df_y_test.to_numpy()[i]:
            score += 1
            check.append('Correct')
        else:
            check.append('Incorrect')

    n = len(final_outputs)


    message = f"The k-class classifier has predicted a total of {score} classes out of {n} correctly. This means it is only correct a {score/n:.0%} of the time"

    return score, check, n, message


def cross_validation_split(X_train, y_train, n_folds):

    """
    Splits a given dataset into k folds
    """
    
    dataset_split = list()
    dataset_train = X_train.join(y_train)
    dataset_copy = list(dataset_train.values.tolist())
    fold_size = int(len(dataset_train.values.tolist()) / n_folds)

    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split


def accuracy_metric(actual, predicted):
    """
    Calculate accuracy percentage
    """
    
    correct = 0
	
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
	
    return correct / float(len(actual)) * 100.0



def evaluate_algorithm(algorithm, n_folds, X_train, X_test, y_train):
    """
    Evaluate an algorithm using a cross validation split
    """

    #if 'y_train' in kwargs:
    #    y_train = kwargs['y_train']
    folds = cross_validation_split(X_train, y_train, n_folds)

    scores = list()
    
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        #transform train_set and test_set to dataframes as X_train and X_test instead of lists
        df_train_set =  pd.DataFrame(data=train_set)
        df_test_set =  pd.DataFrame(data=test_set)
        
        
        #split the df_train_set onto: X_train and y_train to match structure of pluginClassifier function
        rows, cols = df_train_set.shape
        df_y_train = df_train_set[df_train_set.columns[cols - 1]]
        df_X_train = df_train_set[df_train_set.columns[:-1]]
       
        predicted, out_probabilities = algorithm(df_X_train, df_test_set, df_y_train)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    return scores, (predicted), (out_probabilities)


def main():
      
    # get data
    df = get_data('iris.data.csv', col_titles=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species_class']) #use for irisdata set

    # Not needed in Vocareum
    # shuffle dataframe in-place and reset the index to ensure split function does not exclude any classes in the train dataset
    # the frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order)
    #df = df.sample(frac=1).reset_index(drop=True).drop(['Id'], axis = 1).fillna(0) #for when using 'glasstypes.csv'
    #df = df.sample(frac=1).reset_index(drop=True).fillna(0) #for when using 'iris.data.csv'

    # to ensure label is numerical, convert the last column of the dataframe to numerical instead of categorical
    _, cols = df.shape
    df[df.columns[cols - 1]] = df[df.columns[cols - 1]].astype('category')
    col_class = df.select_dtypes(['category']).columns
    df[col_class] = df[col_class].apply(lambda x: x.cat.codes)

    # Not needed in Vocareum
    # split the dataset
    df_X_train, df_X_test, df_y_train, df_y_test = split_data(df, ratio = 0.85)

    # Not needed in Vocareum
    write_csv('X_train.csv', df_X_train)
    write_csv('y_train.csv', df_y_train)
    write_csv('X_test.csv', df_X_test)
    write_csv('y_test.csv', df_y_test)

    # Not needed in Vocareum
    dataset_train = df_X_train.join(df_y_train)
    dataset_test = df_X_test.join(df_y_test)

    # Not needed in Vocareum
    corr = df.corr()
    plt.figure(figsize=(10,10))
    corrplot(corr)
    
    # Not needed in Vocareum
    n_folds = 5
    # evaluate the algorithm and get scores of its accuracy
    scores, class_predicted, prob_output = evaluate_algorithm(pluginClassifier, n_folds, df_X_train, df_X_test, df_y_train)

    # run the classifier to predict the class of each item in the y_train dataset
    class_predicted, prob_output  = pluginClassifier(df_X_train, df_X_test, y_train = df_y_train)

    ## write the results of the prediction to a csv
    np.savetxt("y_validate.csv", class_predicted, fmt='%1i', delimiter="\n") # write output to file, note values for fmt and delimiter

    ## write the probability of predicting the class right to a csv
    write_csv("probs_test.csv", prob_output, header = False, path = os.path.join(os.getcwd(), "probs_test.csv"))

    ## compare the results of the prediction against the y_test dataset to calculate the total prediction error
    score, check, n, message = check_results(df_y_test, class_predicted)
    # write the results from the check ('Correct' / 'Incorrect') to a csv file
    np.savetxt("check_results.csv", check, fmt='%s', delimiter="\n") # write output to file, note values for fmt and delimiter


if __name__ == '__main__':
    main()

