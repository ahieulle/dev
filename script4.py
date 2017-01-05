from __future__ import print_function
import math
import re
import numpy as np
from pandas import read_csv, DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

train_data = read_csv('./boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./boites_medicaments_test.csv', encoding='utf-8', sep=";")

train_data["prix_log"] = train_data["prix"].apply(math.log)
train_data.drop("prix", axis=1, inplace=True)

train_data["date_diff"] = train_data["date declar annee"] - train_data["date amm annee"]
train_data["nb_years_declar"] = train_data["date declar annee"].apply(lambda x: 2016 - x)
train_data["nb_years_amm"] = train_data["date amm annee"].apply(lambda x: 2016 - x)
train_data["nb_comprime_total"] = train_data["nb_comprime"] * train_data["nb_comprime"]


def remove_accents(text):
    return text.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

txt_cols = ["substances" , "titulaires", "statut", "voies admin"]

train_data[txt_cols] = train_data[txt_cols].apply(remove_accents)

def get_titulaires_countries(titulaires):
    reg = re.compile(r"\((.+)\)")
    x = [re.findall(reg, x) for x in titulaires.split(",")][0]
    if len(x) > 0:
        return x[0]
    else:
        return ""

def clean_countries(country):
    countries_mapping = {"LUXMEBOURG" : "LUXEMBOURG",
                         "LYON" : "FRANCE",
                         "IRELAND" : "IRLANDE",
                         "UK) (ROYAUME UNI": "ROYAUME UNI" }
    return countries_mapping[country] if country in countries_mapping else country

train_data["country"] = train_data["titulaires"].apply(get_titulaires_countries)
train_data["country"] = train_data["country"].str.replace("-", " ")
train_data["country"] = train_data["country"].apply(clean_countries)

countries = get_dummies(train_data["country"])
countries.drop("", axis=1, inplace=True)

train_data = train_data.join(countries)

num_cols = [    "libelle_plaquette",
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
                "nb_years_amm",
                "nb_years_declar",
                "date_diff",
                "nb_comprime_total"]

drop_cols = ["prix_log",
            "statut",
            "etat commerc",
            "agrement col",
            "tx rembours",
            "forme pharma",
            "statut admin",
            "type proc",
            "titulaires",
            "substances",
            "voies admin",
            "country",
            "libelle"]

train_data, test_data = train_test_split(train_data, train_size=0.7)

x_train = train_data.drop(drop_cols, axis=1)
y_train = train_data["prix_log"]
x_test = test_data.drop(drop_cols, axis=1)

model = RandomForestRegressor(n_estimators=100)
model.fit(x_train, y_train)
y_log_pred = model.predict(x_test)

y_test = test_data["prix_log"].apply(math.exp).as_matrix()
y_pred = Series(y_log_pred).apply(math.exp).as_matrix()

score = mape(y_test, y_pred)
importance = DataFrame({"feature": x_train.columns, "importance" : model.feature_importances_})
importance.sort_values("importance", ascending=False, inplace=True)
print(importance)
print(score)
