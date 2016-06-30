#!/usr/bin/env python
# -*- coding: utf8 -*-
import time
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

train_data = read_csv('./arthur/concours_pharma/boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./arthur/concours_pharma/boites_medicaments_test.csv', encoding='utf-8', sep=";")

train_data = read_csv('./boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./boites_medicaments_test.csv', encoding='utf-8', sep=";")


num_cols = [ #   "libelle_plaquette",
                # "libelle_ampoule",
                # "libelle_flacon",
                # "libelle_tube",
                # "libelle_stylo",
                # "libelle_seringue",
                # "libelle_pilulier",
                # "libelle_sachet",
                # "libelle_comprime",
                # "libelle_gelule",
                # "libelle_film",
                # "libelle_poche",
                # "libelle_capsule",
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
                "type proc"]
                # "date declar annee",
                # "date amm annee"]

to_binarize = ["titulaires", "substances", "voies admin"]

def my_tokenizer(s):
    return s.split(',')

def mape(y_true, y_pred):
    errors = np.abs((y_true - y_pred) / y_true)
    return np.average(errors)

def inverse_mape(y_true, y_pred):
    return 1. / mape(y_true, y_pred)

def remove_outliers(data, y_label, to_remove=5):
    l = data.shape[0]
    first = np.floor(l * 0.5 * to_remove / 100)
    last = np.floor(l * (100 - 0.5 * to_remove) / 100)
    sorted_index = data[y_label].order().index
    to_keep = sorted_index[first:last]
    return data[data.index.isin(to_keep)]

def vectorize(training, testing, col_name, stop_words):
    vec = CountVectorizer(tokenizer=my_tokenizer, stop_words=stop_words, strip_accents="unicode")
    x_train = vec.fit_transform(training[col_name])
    x_test = vec.transform(testing[col_name])
    return x_train, x_test

def preprocessing(training, testing, libelle=False):
    x_train_ti , x_test_ti = vectorize(training, testing, "titulaires", stop_words_ti)
    x_train_va , x_test_va = vectorize(training, testing, "voies admin", stop_words_va)
    x_train_su , x_test_su = vectorize(training, testing, "substances", stop_words_su)

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

    if libelle:
        v = TfidfVectorizer(sublinear_tf=True,
                            max_df=0.5,
                            ngram_range=(1, 2))

        x_train_lib = v.fit_transform(training["libelle"])
        x_test_lib = v.transform(testing["libelle"])
        x_train = np.hstack((x_train, x_train_lib.toarray()))
        x_test = np.hstack((x_test, x_test_lib.toarray()))

    return x_train, x_test


def vectorize_in_test(col_name):
    v = CountVectorizer(tokenizer=my_tokenizer, stop_words=None, strip_accents="unicode")
    vv = CountVectorizer(tokenizer=my_tokenizer, stop_words=None, strip_accents="unicode")
    v.fit(train_data[col_name])
    vv.fit(test_data[col_name])
    stop = [w for w in v.vocabulary_.keys() if w not in vv.vocabulary_.keys()]
    return stop

stop_words_su = vectorize_in_test("substances")
stop_words_ti = vectorize_in_test("titulaires")
stop_words_va = vectorize_in_test("voies admin")

# v = DictVectorizer(sparse=False)
# vv = DictVectorizer(sparse=False)
# x_cat = train_data[to_vectorize].T.to_dict().values()
# x_test_cat = test_data[to_vectorize].T.to_dict().values()
# v.fit(x_cat)
# vv.fit(x_test_cat)
# stop_words_dict = [w for w in v.vocabulary_.keys() if w not in vv.vocabulary_.keys()]
# del v,vv

#################################################################
#################################################################

train_data = remove_outliers(train_data, "prix", 10)
train_data = train_data.reset_index(drop=True)

y = train_data["prix"]

to_drop = ["libelle" , "prix"]

train_data = train_data.drop(to_drop, axis=1)

train_data["date_diff"] = train_data["date declar annee"] - train_data["date amm annee"]
train_data["nb_years_declar"] = train_data["date declar annee"].apply(lambda x: 2016 - x)
train_data["nb_years_amm"] = train_data["date amm annee"].apply(lambda x: 2016 - x)

# test_data = test_data.drop(["id", "libelle"], axis=1)


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
    x_train, x_test = preprocessing(training, testing, False)
    y_train = y[y.index.isin(train_idx)]
    y_test = y[y.index.isin(test_idx)]

    # x_scaler = StandardScaler()
    # # y_scaler = StandardScaler()
    #
    # x_train = x_scaler.fit_transform(x_train)
    # x_test = x_scaler.transform(x_test)
    #
    # q0 = KernelDensity(kernel="gaussian", bandwidth=0.75)
    # q0.fit(x_train)
    #
    # q1 = KernelDensity(kernel="gaussian", bandwidth=0.75)
    # q1.fit(x_test)
    #
    # w = q1.score_samples(x_train)/q0.score_samples(x_train)
    # y_train = y_scaler.fit_transform(y_train)

    ## feature_selection
    # fs = SelectKBest(f_regression, k=k)
    # x_train = fs.fit_transform(x_train, y_train)
    # x_test = fs.transform(x_test)

    # model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0, n_jobs=10)
    # model = SGDRegressor(loss='huber', penalty="elasticnet", n_iter=100)
    # model = KNeighborsRegressor(n_neighbors=10, n_jobs=10, weights='distance')

    base_model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0, n_jobs=10)
    model = RANSACRegressor(base_model, min_samples=1000, max_trials=1000)

    # model = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=4, learning_rate=0.1)
    # model = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000), scoring=make_scorer(inverse_mape), cv=3)
    # model = SVR(kernel="rbf", C=0.01)
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train, sample_weight=w)

    y_pred = model.predict(x_test)

    # y_pred = y_scaler.inverse_transform(y_pred)

    scores.append(mape(y_test, y_pred))
    # y_pred_list.append(y_pred)
    # y_test_list.append(y_test)

    print time.time() - t0
    it += 1

diff_list = []
for i in range(0, n_folds):
    s = Series(y_pred_list[i])
    s.index = y_test_list[i].index
    diff_list.append(concat([y_test_list[i], s], axis=1))

for i in range(0, n_folds):
    diff_list[i].columns = ['truth', 'pred']

for i in range(0, n_folds):
    diff_list[i]["ape"] = diff_list[i].apply(lambda x: np.abs((x['truth'] - x['pred'])/ x['truth']), axis=1)

idx = []
for i in range(0, n_folds):
    idx.extend(list( diff_list[i].sort('ape', ascending=False).head(20).index))


scores = []
it = 0
for train_idx, test_idx in kfold:
    print "fold " + str(it +1)
    t0 = time.time()
    ## compute KF train and test sets:
    training = train_data[train_data.index.isin(train_idx)]
    testing = train_data[train_data.index.isin(test_idx)]

    # v = TfidfVectorizer(sublinear_tf=True,
    #                     max_df=0.5,
    #                     ngram_range=(1, 2))
    #
    # x_train = v.fit_transform(training["libelle"])
    # x_test = v.transform(testing["libelle"])


    # x_train = training[num_cols].as_matrix()
    # x_test = testing[num_cols].as_matrix()
    # x_train, x_test = vectorize(training, testing, "substances", stop_words_su)
    # x_train = x_train.toarray()
    # x_test = x_test.toarray()

    x_train_cat = training[to_vectorize].T.to_dict().values()
    x_test_cat = testing[to_vectorize].T.to_dict().values()
    vec_dict = DictVectorizer(sparse=False)
    x_train = vec_dict.fit_transform(x_train_cat)
    x_test = vec_dict.transform(x_test_cat)

    print "dim x_train : ", x_train.shape

    y_train = y[y.index.isin(train_idx)]
    y_test = y[y.index.isin(test_idx)]

    model = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0, n_jobs=3)
    # model = SGDRegressor(loss='huber', penalty="elasticnet")
    # model = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=4, learning_rate=0.01)
    # model = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000), scoring=make_scorer(inverse_mape), cv=3)
    # model = SVR(kernel="rbf", C=0.01)
    # model = LinearSVR(C = 0.01)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # y_pred = y_scaler.inverse_transform(y_pred)

    scores.append(mape(y_test, y_pred))
    # y_pred_list.append(y_pred)
    # y_test_list.append(y_test)

    print time.time() - t0
    it += 1


x_train, x_test = preprocessing(train_data, test_data)
model = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, n_jobs=3)
model.fit(x_train, y)

y_pred = model.predict(x_test)

res = concat([test_data["id"], Series(y_pred)], axis=1)
res.columns = ["Id", "Prix"]
res["Prix"] = res["Prix"].round(4)

res.to_csv("./data/res.csv", sep=";", index=False)



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

train_data = read_csv('./boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./boites_medicaments_test.csv', encoding='utf-8', sep=";")


substances = train_data.substances.apply(lambda x: re.sub(r'\(|\)|,','',x))
substances_test = test_data.substances.apply(lambda x: re.sub(r'\(|\)|,','',x))
vec = CountVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(3,3), binary=True)
x = vec.fit_transform(substances)
xtest = vec.transform(substances_test)
cos = cosine_similarity(x)
cos2 = cosine_similarity(x,xtest)
mask = cos > 0.7

for i in range(0,cos2.shape[1]):
    m = max(cos2[:,i])
    if m < 0.8 and m>0.7:
        print i, test_data.substances.loc[i], max(cos2[:,i]), min(cos2[:,i])
        j = np.argmax(cos2[:,i])
        print train_data.loc[j,"substances"]
        print "----------------------------"

def make_clusters(train):
    clusters = []
    done = []
    cluster_index = 0
    for i in range(0, train.shape[0]):
        if i in done:
            continue
        else:
            index = train.index[mask[i]]
            clusters.append({cluster_index: index })
            done.extend(list(index))
            cluster_index += 1
    return clusters



def substances_processing(x):
    s_lower = unicode2ascii(x).lower().replace("(","").replace(")","")
    subs = s_lower.split(", ")
    sorted_subs = [" ".join(sorted(s.split(" "))) for s in subs]
    x = ", ".join(sorted(sorted_subs))
    return x

train_data["substances"] = train_data["substances"].apply(substances_processing)
test_data["substances"] = test_data["substances"].apply(substances_processing)

# train_data["substances"] = train_data["substances"].apply(lambda x: unicode2ascii(x).lower())
# train_data["substances"] = train_data["substances"].apply(lambda x: ", ".join(sorted(x.split(", "))))
#
# test_data["substances"] = test_data["substances"].apply(lambda x: unicode2ascii(x).lower())
# test_data["substances"] = test_data["substances"].apply(lambda x: ", ".join(sorted(x.split(", "))))

num_cols = [    "nb_plaquette",
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

def add_class_stats(data, stats):
    for k in ["mean", "median"]:
        data["prix_"+k] = data["substances"].apply(lambda x: stats.loc[x,k])
    return data


train_data["date_diff"] = train_data["date declar annee"] - train_data["date amm annee"]
train_data["nb_years_declar"] = train_data["date declar annee"].apply(lambda x: 2016 - x)
train_data["nb_years_amm"] = train_data["date amm annee"].apply(lambda x: 2016 - x)

y = train_data["prix"]

training, testing, y_train, y_test = train_test_split(train_data, y, train_size=0.7)

gb = training.groupby("substances")
vc = value_counts(training['substances'])
vc.name = "counts"

stats = concat([vc,
                gb['prix'].agg({"mean":np.mean,
                                "median":np.median})], axis=1)

training = add_class_stats(training, stats)

i1 = testing['substances'].isin(training.substances)
testing1 = testing[i1]
y_test1 = y_test[i1]
testing2 = testing[~i1]
y_test2 = y_test[~i1]

num_cols2 = num_cols + ["prix_median", "prix_mean"]

x_train = training[num_cols2].as_matrix()

model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0, n_jobs=10)
# model = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=4, learning_rate=0.01)
model.fit(x_train, y_train)

testing = testing.drop("prix", axis=1)

testing1 = add_class_stats(testing1, stats)

y_pred1 = model.predict(testing1[num_cols2].as_matrix())

mape(y_test1, y_pred1)

model2 = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0, n_jobs=10)
to_keep = num_cols + to_vectorize + to_binarize
x_train , x_test = preprocessing(training[to_keep], testing2[to_keep])
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
mape(y_test2, y_pred2)
