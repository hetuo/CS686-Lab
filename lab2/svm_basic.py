#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:40:30 2018

@author: yuntuotuo
"""

from classifier import classifier
import numpy as np
from svmMLiA import smoPK
from svmMLiA import calcWs
from svmMLiA import loadDataSet

class svm_basic(classifier):

    def __init__(self, cycles):
        self.alpha = 0.001
        self.maxcycles = cycles
        self.weights = None  # Placeholder for later...


    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def fit(self, Xin, Yin):
        b, alphas = smoPK(Xin, Yin, 0.6, 0.001, 40)
        weights = calcWs(alphas, Xin, Yin)
        weightArray = [b.getA()[0][0], weights[0][0], weights[1][0]]
        self.Weights = weightArray
        return weightArray
        
        
 

    def predict(self, X):
        hypotheses = []
        for x in X:
            prob = self.sigmoid(sum(x*self.weights))
            if prob > 0.5:
                hypotheses.append(1)
            else:
                hypotheses.append(0)
        return hypotheses