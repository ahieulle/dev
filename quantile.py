#!/usr/bin/env python
# -*- coding: utf8 -*-
import time
import re
import numpy as np
from pandas import read_csv, Series, DataFrame, concat, value_counts
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor, KernelDensity


params = {  'loss':'lad',
            'max_depth': 4,
            'n_estimators': 10000,
            'learning_rate': 0.01,
            'verbose': 1,
            'subsample': 0.5 }

model = GradientBoostingRegressor(**params)
weights = 1.0/ y_train
model.fit(x_train, y_train, sample_weight=weights)
y_pred = model.predict(x_test)
score = mape(y_test, y_pred)
print "score", score



# params = {  'loss':'lad',
#             'max_depth': 4,
#             'n_estimators': 10000,
#             'learning_rate': 0.01,
#             'verbose': 1,
#             'subsample': 0.5 }
# rm outliers 10 testing= 0.3
