from __future__ import print_function
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.cross_validation import KFold

def mape(y_true, y_pred):
    errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(errors)

train_data["logprix"] = train_data["prix"].apply(np.log)
# train_data["logprix"] = ((train_data["prix"].apply(np.log)+1).apply(np.log)+1).apply(np.log)

n_folds = 5
kfold = KFold(train_data.shape[0], n_folds, shuffle=True, random_state=12345)
scores = []

params = {"n_estimators":200,
            # "max_depth":None,
            # "min_samples_split":1,
            # "random_state":0,
            "n_jobs":10}

it = 0
for train_idx, test_idx in kfold:
    print ("fold " + str(it +1))
    t0 = time.time()
    ## compute KF train and test sets:
    x_train = train_data.loc[train_idx, variables]
    x_test = train_data.loc[test_idx, variables]

    y_train = train_data.loc[train_idx, "logprix"]
    y_test = train_data.loc[test_idx, "logprix"]

    model = RandomForestRegressor(**params)
    # model = RANSACRegressor(base_model, min_samples=1000, max_trials=1000)
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train, sample_weight=w)
    log_y_pred = model.predict(x_test)
    y_pred = np.exp(log_y_pred)
    score = mape(np.exp(y_test), y_pred)
    # y_pred = np.exp(-1+np.exp(-1+np.exp(log_y_pred)))
    # score = mape(np.exp(-1+np.exp(-1+np.exp(y_test))), y_pred)

    scores.append(score)

    print (time.time() - t0)
    it += 1

print(scores)
print ("*** Random Forest MAPE Error : ", np.mean(scores))
importance = DataFrame({"fname": variables, "importance":model.feature_importances_})
importance = importance.sort_values('importance',ascending=False)
importance.head(15)
