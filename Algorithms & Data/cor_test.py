#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 11:27:57 2018

@author: DANDING
"""

import pandas as pd

import numpy as np

from sklearn.naive_bayes import GaussianNB

data0 = pd.read_csv('WeatherData_Averages1.csv', sep=',')

data = pd.DataFrame(data0)

features_train = data.iloc[:, 6:]

correlations = features_train.corr()

correlations.to_csv('/Users/apple/desktop/CorrTest.csv')
