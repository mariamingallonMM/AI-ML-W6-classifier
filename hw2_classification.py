"""
This code implements a k-class Classifier per week 6 assignment of the machine learning module part of Columbia University Micromaster programme in AI. 
Written using Python 3.X for running on Vocareum
"""

# builtin modules
from __future__ import division
import os
#import psutil
import requests
import sys
import math

# 3rd party modules
import pandas as pd
import numpy as np




def pluginClassifier(X_train, y_train, X_test):    
  # this function returns the required output 

  final_outputs=

  return final_outputs


def main():

    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function
    np.savetxt("probs_test.csv", final_outputs, fmt='%1.2f', delimiter="\n") # write output to file, note values for fmt and delimiter


if __name__ == '__main__':
    main()


 


