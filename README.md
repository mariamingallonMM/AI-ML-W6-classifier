﻿# AI-ML-W6-classifier

ColumbiaX CSMM.102x Machine Learning Course. Week 6 Assignment on Classifiers.

## Instructions

In this assignment, we will implement a K-class Bayes classifier. Assume the following generative model: 

- We are given labeled data  (x1,y1),…,(xN,yN) , where  x∈Rd  and  y∈{1,…,K} .
- For the i-th data point, assume that:

![equation_1: yi∼iidDiscrete(π),xi|yi∼Normal(μyi,Σyi),i=1,…,N.](./ref/eq1.JPG?raw=true)

For this model, we will **derive the maximum likelihood updates** for the class prior probability vector  πˆ  and the class-specific Gaussian parameters (μˆk,Σˆk) for each class k=1,…,K , where ⋅ˆ indicates *maximum likelihood estimate*. While we are not required to turn in these derivations, we will need to implement them in our code, as well as the prediction for a new point y0 given x0 and these estimates:

![equation_2: Prob(y0=y|x0,πˆ,(μˆ1,Σˆ1),…,(μˆK,ΣˆK))](./ref/eq2.JPG?raw=true)

## Execute the program

The following command will execute your program:
> $ python3 hw2_classification.py X_train.csv y_train.csv X_test.csv

Note the following:
- The name of the train and testing X and y datasets. 
- The main .py file shall be named 'hw2_regression.py'.

The input csv files will need to be formatted as follows:

- X_train.csv: A comma separated file containing the covariates. Each row corresponds to a single vector  xi.
- y_train.csv: A file containing the classes. Each row has a single number and the i-th row of this file combined with the i-th row of "X_train.csv" constitutes the labeled pair  (yi,xi) . There are 10 classes having index values 0,1,2,...,9.
- X_test.csv: This file follows exactly the same format as "X_train.csv". 

No class file is given for the testing data. This assignment uses supervised learning, the main idea being to be able to generalize to samples that we haven't seen before.


## Expected Outputs from the program

When executed, the code writes the output to the file listed below following the formatting requirements specified also below.

**probs_test.csv**: This is a comma separated file containing the posterior probabilities of the label of each row in **X_test.csv**. Since there are 10 classes, the i-th row of this file should contain 10 numbers, where the j-th number is the probability that the i-th testing point belongs to class j-1 (since classes are indexed 0 to 9 here).

## Plotting results

Although not required in this assignment, I have created a jupyter notebook on Kaggle.com to better understand :
1. the initial dataset by plotting its correlations
2. the results from the k-class Bayes classifier by plotting the outputs

[AI-ColumbiaX-ML-W6-K-class Bayes Classifier](https://www.kaggle.com/mariamingallon/ai-columbiax-ml-w6-k-class-bayes-classifier)


## How we develop the code

We use Bayes' Theorem to calculate the probability of a piece of data belonging to a given class, give our prior knowledge. Bayes' Theorem is stated as follows:

P(class|data) = (P(data|class) * P(class)) / P(data)

where 
P(class|data) is the probability of class given the provided data.

Naive Bayes is called 'naive' because the calculations of the probabilities for each class are simplified. Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent given the class value. Although the latter is not true is most real life cases, the assumption still produces accurate results while enormously simplifying calculations.

1. Split the dataset by class values, return a dictionary.
2. Calculate the mean, stdev and count for each column in the dataset.
3. Split dataset by class and calculate statistics for each row.
4. Calculate the Gaussian probability distribution function for x.
5. Calculate the probabilities of predicting each class for a given row.
6. Predict the class for a given row.
7. Implement Naive Bayes Algorithm.


## Note on Correctness

Please note that for both of these problems, there is one and only one correct solution. Therefore, we will grade your output based on how close your results are to the correct answer. We strongly suggest that you test out your code on your own computer before submitting. The UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/) has a good selection of datasets for classification.

## Notes on data repositories

The following datasets have been selected from the UCI Machine Learning Repository for use and testing of the code written for this assignment:

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires). This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data (see details [here](http://www.dsi.uminho.pt/~pcortez/forestfires)).
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality). Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009](http://www3.dsi.uminho.pt/pcortez/wine/)).
- [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris). This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (see [Duda & Hart](http://rexa.info/paper/e6b7a3a8c46efef785a6ab735be07dafa0713ff3), for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.


## Citations & References

- [Forest Fires Data Set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires) by P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.
- [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset has been used for testing the Naives Bayes classifier as it consists of colour images in 10 different classes. Chapter 3 of the following report describes the dataset and the methodology followed when collecting the CIFAR-10 dataset in much greater detail. Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
- [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris) by Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.