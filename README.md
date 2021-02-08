# AI-ML-W6-classifier
ColumbiaX CSMM.102x Machine Learning Course. Week 6 Assignment on Classifiers.


## INSTRUCTIONS
Assume you are given labeled data  (x1,y1),…,(xN,yN) , where  x∈Rd  and  y∈{1,…,K} . In this assignment, you will implement a K-class Bayes classifier. In the specific classifier that you will implement, assume the following generative model: For the i-th data point, assume that

https://github.com/mariamingallonMM/AI-ML-W6-classifier/tree/main/ref
![equation_1: yi∼iidDiscrete(π),xi|yi∼Normal(μyi,Σyi),i=1,…,N.](./ref/eq1.JPG?raw=true)

For this model, you will need to derive the maximum likelihood updates for the class prior probability vector  πˆ  and the class-specific Gaussian parameters  (μˆk,Σˆk)  for each class  k=1,…,K , where  ⋅ˆ  indicates "maximum likelihood estimate". While you will not turn in these derivations, you will need to implement them in your code, as well as the prediction for a new point  y0  given  x0  and these estimates:

![equation_2: Prob(y0=y|x0,πˆ,(μˆ1,Σˆ1),…,(μˆK,ΣˆK))](./ref/eq2.JPG?raw=true)

More details about the inputs we provide and the expected outputs are given below.

Sample starter code to read the inputs and write the outputs:  Download hw2_classification.py

## WHAT YOU NEED TO SUBMIT
You can use either Python (3.6.4) or Octave coding languages to complete this assignment. Octave is a free version of Matlab. Your Matlab code should be able to directly run in Octave, but you should not assume that advanced built-in functions will be available to you in Octave. Unfortunately we will not be supporting other languages in this course.


Execute the program by running:

$ python3 hw2_classification.py X_train.csv y_train.csv X_test.csv


The csv files that will be input as default are formatted as follows:

- X_train.csv: A comma separated file containing the covariates. Each row corresponds to a single vector  xi .
- y_train.csv: A file containing the classes. Each row has a single number and the i-th row of this file combined with the i-th row of "X_train.csv" constitutes the labeled pair  (yi,xi) . There are 10 classes having index values 0,1,2,...,9.
- X_test.csv: This file follows exactly the same format as "X_train.csv". No class file is given for the testing data.

## WHAT YOUR PROGRAM OUTPUTS

When executed, you should have your code write the output to the file listed below. It is required that you follow the formatting instructions given below.

probs_test.csv: This is a comma separated file containing the posterior probabilities of the label of each row in "X_test.csv". Since there are 10 classes, the i-th row of this file should contain 10 numbers, where the j-th number is the probability that the i-th testing point belongs to class j-1 (since classes are indexed 0 to 9 here).


## Note on Correctness

Please note that for both of these problems, there is one and only one correct solution. Therefore, we will grade your output based on how close your results are to the correct answer. We strongly suggest that you test out your code on your own computer before submitting. The UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/) has a good selection of datasets for classification.