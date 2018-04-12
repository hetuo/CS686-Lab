#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:27:06 2018

@author: yuntuotuo
"""

from svm_basic import svm_basic
from svmMLiA import loadDataSet

def plot_fit(fit_line, datamatrix, labelmatrix):
    import matplotlib.pyplot as plt
    import numpy as np

    weights = fit_line
    print(len(weights))
    dataarray = np.asarray(datamatrix)
    n = dataarray.shape[0]

    # Keep track of the two classes in different arrays so they can be plotted later...
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmatrix[i]) == 1:
            xcord1.append(dataarray[i, 1])
            ycord1.append(dataarray[i, 2])
        else:
            xcord2.append(dataarray[i, 1])
            ycord2.append(dataarray[i, 2])
    fig = plt.figure()

    # Plot the data as points with different colours
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot the best-fit line
    x = np.arange(-1.0, 6.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(y)
    ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def get_data(filename):
    datamatrix = []
    labelmatrix = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        datamatrix.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelmatrix.append(int(lineArr[2]))
    return datamatrix, labelmatrix


clf = svm_basic(5000)
dataArr, labelArr = loadDataSet('linearly_separable.csv')
weightArray = clf.fit(dataArr, labelArr)
datamatrix, labelmatrix = get_data('linearly_separable.csv')
plot_fit(weightArray, datamatrix, labelmatrix)

