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
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, KernelDensity

train_data = read_csv('./boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./boites_medicaments_test.csv', encoding='utf-8', sep=";")

num_cols = [   "libelle_plaquette",
                "libelle_ampoule",
                "libelle_flacon",
                "libelle_tube",
                "libelle_stylo",
                "libelle_seringue",
                "libelle_pilulier",
                "libelle_sachet",
                "libelle_comprime",
                "libelle_gelule",
                "libelle_film",
                "libelle_poche",
                "libelle_capsule",
                "nb_plaquette",
                "nb_ampoule",
                "nb_flacon",
                "nb_tube",
                "nb_stylo",
                "nb_seringue",
                "nb_pilulier",
                "nb_sachet",
                "nb_comprime",
                "nb_gelule",
                "nb_film",
                "nb_poche",
                "nb_capsule",
                "nb_ml",
                "date_diff",
                "nb_years_amm",
                "nb_years_declar"]

to_vectorize = ["statut",
                "etat commerc",
                "agrement col",
                "tx rembours",
                "forme pharma",
                "statut admin",
                "type proc",
                "date declar annee",
                "date amm annee"]

to_binarize = ["titulaires", "substances", "voies admin"]

train_data["date_diff"] = train_data["date declar annee"] - train_data["date amm annee"]
train_data["nb_years_declar"] = train_data["date declar annee"].apply(lambda x: 2016 - x)
train_data["nb_years_amm"] = train_data["date amm annee"].apply(lambda x: 2016 - x)

y = train_data["prix"]
to_drop = ["prix"]
train_data = train_data.drop(to_drop, axis=1)

def my_tokenizer(s):
    return s.split(',')

def vectorize(training, testing, col_name, stop_words):
    vec = CountVectorizer(tokenizer=my_tokenizer, stop_words=stop_words, strip_accents="unicode")
    x_train = vec.fit_transform(training[col_name])
    x_test = vec.transform(testing[col_name])
    return x_train, x_test

def vectorize_in_test(col_name):
    v = CountVectorizer(tokenizer=my_tokenizer, stop_words=None, strip_accents="unicode")
    vv = CountVectorizer(tokenizer=my_tokenizer, stop_words=None, strip_accents="unicode")
    v.fit(train_data[col_name])
    vv.fit(test_data[col_name])
    stop = [w for w in v.vocabulary_.keys() if w not in vv.vocabulary_.keys()]
    return stop

stop_words_ti = vectorize_in_test("titulaires")
stop_words_va = vectorize_in_test("voies admin")

def vectorize_substances(training, testing):
    substances = training.substances.apply(lambda x: re.sub(r'\(|\)|,','',x))
    substances_test = testing.substances.apply(lambda x: re.sub(r'\(|\)|,','',x))
    vec = CountVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(3,3), binary=True)
    x = vec.fit_transform(substances)
    xtest = vec.transform(substances_test)
    return x, xtest


def preprocessing(training, testing):
    x_train_ti , x_test_ti = vectorize(training, testing, "titulaires", stop_words_ti)
    x_train_va , x_test_va = vectorize(training, testing, "voies admin", stop_words_va)
    x_train_su , x_test_su = vectorize_substances(training, testing)

    x_train_num = training[num_cols].as_matrix()
    x_test_num = testing[num_cols].as_matrix()
    x_train_num = np.hstack((x_train_num, x_train_va.toarray(), x_train_ti.toarray(), x_train_su.toarray()))
    x_test_num = np.hstack((x_test_num, x_test_va.toarray(), x_test_ti.toarray(), x_test_su.toarray()))

    x_train_cat = training[to_vectorize].T.to_dict().values()
    x_test_cat = testing[to_vectorize].T.to_dict().values()
    vec_dict = DictVectorizer(sparse=False)
    x_train_cat = vec_dict.fit_transform(x_train_cat)
    x_test_cat = vec_dict.transform(x_test_cat)

    x_train = np.hstack((x_train_num, x_train_cat))
    x_test = np.hstack((x_test_num, x_test_cat))

    return x_train, x_test


n_folds = 5
kfold = KFold(y.shape[0], n_folds, shuffle=True, random_state=12345)

k = 500

scores = []
# y_pred_list = []
# y_test_list = []
it = 0
for train_idx, test_idx in kfold:
    print "fold " + str(it +1)
    t0 = time.time()
    ## compute KF train and test sets:
    training = train_data[train_data.index.isin(train_idx)]
    testing = train_data[train_data.index.isin(test_idx)]

    # x_train, x_test = preprocessing(training, testing)
    x_train, x_test = preprocessing(training, testing)
    y_train = y[y.index.isin(train_idx)]
    y_test = y[y.index.isin(test_idx)]
    model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0, n_jobs=10)
    # model = RANSACRegressor(base_model, min_samples=1000, max_trials=1000)
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train, sample_weight=w)
    y_pred = model.predict(x_test)

    scores.append(mape(y_test, y_pred))
    # y_pred_list.append(y_pred)
    # y_test_list.append(y_test)

    print time.time() - t0
    it += 1
