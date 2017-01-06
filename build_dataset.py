from __future__ import print_function

from pandas import read_csv, notnull
from sklearn.preprocessing import LabelEncoder

train_data = read_csv('./boites_medicaments_train.csv', encoding='utf-8', sep=";")
test_data = read_csv('./boites_medicaments_test.csv', encoding='utf-8', sep=";")

originals = [   "statut",
                "etat commerc",
                "agrement col",
                "tx rembours",
                "forme pharma",
                "statut admin",
                "type proc",
                "date declar annee",
                "date amm annee",
                "titulaires",
                "substances",
                "voies admin",
                "libelle",
                "prix"]


cis = read_csv("./external_data/CIS_bdpm.txt", sep="\t", header=None)
cis_cols = ["CIS", "presentation", "forme pharma", "voies admin", "statut admin",
                "type proc", "etat_commerc_bin", "date_amm", "statut_bdm", "num_autorisation_eur",
                "titulaires", "sureveillance_renforcee"]
cis.columns = cis_cols
cis["date amm annee"] = cis["date_amm"].apply(lambda x: x.split("/")[2])

cis_cip = read_csv("./external_data/CIS_CIP_bdpm.txt", sep="\t", header=None)
cis_cip_cols = ["CIS", "CIP7", "libelle", "statut", "etat commerc",
                "date_declar_amm", "CIP13", "agrement col", "tx rembours", "prix",
                "prix2", "diff_prix", "indications_tx_rembours"]
cis_cip.columns = cis_cip_cols
cis_cip = cis_cip[notnull(cis_cip["prix"])]
cis_cip["date declar annee"] = cis_cip["date_declar_amm"].apply(lambda x: x.split("/")[2])

cis_compo = read_csv("./external_data/CIS_COMPO_bdpm.txt", sep="\t", header=None)
cis_compo_cols = ["CIS", "element_pharma", "code_susbtance", "substances", "dosage_substance",
                    "ref_dosage", "nature_composant", "numero_lien_substances", "unknown"]
cis_compo.columns = cis_compo_cols

final = cis.set_index("CIS").join(cis_cip.set_index("CIS")).join(cis_compo.set_index("CIS"))
final = final[notnull(final["prix"])]


def remove_accents(text):
    return text.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

def lower(text):
    return text.str.lower()

txt_cols = [   "statut",
                "etat commerc",
                "agrement col",
                "forme pharma",
                "statut admin",
                "type proc",
                "titulaires",
                "substances",
                "voies admin",
                "libelle"]

train_data[txt_cols] = train_data[txt_cols].apply(remove_accents)
train_data[txt_cols] = train_data[txt_cols].apply(lower)
test_data[txt_cols] = test_data[txt_cols].apply(remove_accents)
test_data[txt_cols] = test_data[txt_cols].apply(lower)
final[txt_cols] = final[txt_cols].apply(remove_accents)
final[txt_cols] = final[txt_cols].apply(lower)


###########
temp = train_data["voies admin"].apply(lambda x: x.split(","))
voies_admin = sorted(set([l for j in temp for l in j]))
for va in voies_admin:
    train_data["va_" + va] = train_data["voies admin"].apply(lambda x : 1 if va in x else 0)
    test_data["va_" + va] = test_data["voies admin"].apply(lambda x : 1 if va in x else 0)
    final["va_" + va] = final["voies admin"].apply(lambda x : 1 if va in x else 0)

##########
temp = train_data["titulaires"].apply(lambda x: x.split(","))
titulaires = sorted(set([l for j in temp for l in j]))
for titu in titulaires:
    train_data["titu_" + titu] = train_data["titulaires"].apply(lambda x : 1 if titu in x else 0)
    test_data["titu_" + titu] = test_data["titulaires"].apply(lambda x : 1 if titu in x else 0)
    final["titu_" + titu] = final["titulaires"].apply(lambda x : 1 if titu in x else 0)

#########
temp = train_data['substances'].apply(lambda x : x.split(','))
train_data['nb_substances'] = train_data['substances'].apply(lambda x : len(x.split(',')))
test_data['nb_substances'] = test_data['substances'].apply(lambda x : len(x.split(',')))
final['nb_substances'] = final['substances'].apply(lambda x : len(x.split(',')))
substances = set([l for j in temp for l in j])
substances = {x[1:] if x[0]==' ' else x for x in substances}
substances = {x[:-1] if x[-1] == ' ' else x for x in substances}
substances = sorted(substances)
for s in substances:
    train_data['sub_' + s] = train_data['substances'].apply(lambda x : 1 if s in x else 0)
    test_data['sub_' + s] = test_data['substances'].apply(lambda x : 1 if s in x else 0)
    final['sub_' + s] = final['substances'].apply(lambda x : 1 if s in x else 0)

#######
var_cat = [     "statut",
                "etat commerc",
                "agrement col",
                "forme pharma",
                "statut admin",
                "type proc"]


for c in var_cat:
    le = LabelEncoder()
    le.fit(train_data[c].append(test_data[c]).append(final[c]))
    train_data[c] = le.transform(train_data[c])
    test_data[c] = le.transform(test_data[c])
    final[c] = le.transform(final[c])

train_data['tx rembours'] = (train_data["tx rembours"].str.rstrip("%")).apply(int)
test_data['tx rembours'] = (test_data["tx rembours"].str.rstrip("%")).apply(int)
final['tx rembours'] = (final["tx rembours"].str.rstrip("%")).apply(int)

var_substances = ["sub_" + s for s in substances] + ["nb_substances"]
var_titulaires = ["titu_" + titu for titu in titulaires]
var_voies_admin = ["va_" + va for va in voies_admin]

var_libelle = ['libelle_plaquette', 'libelle_ampoule', 'libelle_flacon',
            'libelle_tube', 'libelle_stylo', 'libelle_seringue',
            'libelle_pilulier', 'libelle_sachet', 'libelle_comprime',
            'libelle_gelule', 'libelle_film', 'libelle_poche',
            'libelle_capsule']
var_nb = ['nb_plaquette', 'nb_ampoule',
            'nb_flacon', 'nb_tube', 'nb_stylo', 'nb_seringue',
            'nb_pilulier', 'nb_sachet', 'nb_comprime', 'nb_gelule',
            'nb_film', 'nb_poche', 'nb_capsule', 'nb_ml']

var_generique = ["generique"]

var_dates = ['date declar annee', 'date amm annee']

variables = var_libelle + var_nb + var_substances + var_titulaires + var_cat \
                + var_voies_admin + ["tx rembours"] \
                + var_generique

titulaires_idx = list(train_data.columns).index("titulaires")
idx = 0
for row in train_data.itertuples():
    labs = str(row[titulaires_idx + 1])
    value_generique = 1 if ('generiques' in labs or 'generics' in labs or 'eurogenerics' in labs) else 0
    train_data.set_value(idx, 'generique', int(value_generique))
    idx += 1

titulaires_idx = list(test_data.columns).index("titulaires")
idx = 0
for row in test_data.itertuples():
    labs = str(row[titulaires_idx + 1])
    value_generique = 1 if ('generiques' in labs or 'generics' in labs or 'eurogenerics' in labs) else 0
    test_data.set_value(idx, 'generique', int(value_generique))
    idx += 1
