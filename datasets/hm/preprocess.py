# %% [markdown]
# # hm dataset pre-processing
#
# import packages

import datetime
import math

# %%
import os
import pickle as pkl
import re
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# %% [markdown]
# ## Demographic data

# %%
demographic = pd.read_csv(
    "./datasets/hm/raw_data/19_04_2021/COVID_DSL_01.CSV",
    encoding="ISO-8859-1",
    sep="|",
)
print(len(demographic))
demographic.head()

# %% [markdown]
# get rid of patient with missing label

# %%
print(len(demographic))
demographic = demographic.dropna(
    axis=0,
    how="any",
    subset=["IDINGRESO", "F_INGRESO_ING", "F_ALTA_ING", "MOTIVO_ALTA_ING"],
)
print(len(demographic))

# %%
def outcome2num(x):
    if x == "Fallecimiento":
        return 1
    else:
        return 0


def to_one_hot(x, feature):
    if x == feature:
        return 1
    else:
        return 0


# %%
# select necessary columns from demographic
demographic = demographic[
    [
        "IDINGRESO",
        "EDAD",
        "SEX",
        "F_INGRESO_ING",
        "F_ALTA_ING",
        "MOTIVO_ALTA_ING",
        "ESPECIALIDAD_URGENCIA",
        "DIAG_URG",
    ]
]

# rename column
demographic = demographic.rename(
    columns={
        "IDINGRESO": "PATIENT_ID",
        "EDAD": "AGE",
        "SEX": "SEX",
        "F_INGRESO_ING": "ADMISSION_DATE",
        "F_ALTA_ING": "DEPARTURE_DATE",
        "MOTIVO_ALTA_ING": "OUTCOME",
        "ESPECIALIDAD_URGENCIA": "DEPARTMENT_OF_EMERGENCY",
        "DIAG_URG": "DIAGNOSIS_AT_EMERGENCY_VISIT",
    }
)

# SEX: male: 1; female: 0
demographic["SEX"].replace("MALE", 1, inplace=True)
demographic["SEX"].replace("FEMALE", 0, inplace=True)

# outcome: Fallecimiento(dead): 1; others: 0
demographic["OUTCOME"] = demographic["OUTCOME"].map(outcome2num)

# diagnosis at emergency visit (loss rate < 10%)
# demographic['DIFFICULTY_BREATHING'] = demographic['DIAGNOSIS_AT_EMERGENCY_VISIT'].map(lambda x: to_one_hot(x, 'DIFICULTAD RESPIRATORIA')) # 1674
# demographic['SUSPECT_COVID'] = demographic['DIAGNOSIS_AT_EMERGENCY_VISIT'].map(lambda x: to_one_hot(x, 'SOSPECHA COVID-19')) # 960
# demographic['FEVER'] = demographic['DIAGNOSIS_AT_EMERGENCY_VISIT'].map(lambda x: to_one_hot(x, 'FIEBRE')) # 455

# department of emergency (loss rate < 10%)
# demographic['EMERGENCY'] = demographic['DEPARTMENT_OF_EMERGENCY'].map(lambda x: to_one_hot(x, 'Medicina de Urgencias')) # 3914

# %%
# del useless data
demographic = demographic[
    [
        "PATIENT_ID",
        "AGE",
        "SEX",
        "ADMISSION_DATE",
        "DEPARTURE_DATE",
        "OUTCOME",
        # 'DIFFICULTY_BREATHING',
        # 'SUSPECT_COVID',
        # 'FEVER',
        # 'EMERGENCY'
    ]
]

# %%
demographic.describe().to_csv(
    "./datasets/hm/demographic_overview.csv", mode="w", index=False
)
demographic.describe()

# %% [markdown]
# ### Analyze data

# %%
plt.scatter(demographic["PATIENT_ID"], demographic["AGE"], s=1)
plt.xlabel("Patient Id")
plt.ylabel("Age")
plt.title("Patient-Age Scatter Plot")

# %% [markdown]
# ### Normalize data

# %%
# demographic['AGE'] = (demographic['AGE'] - demographic['AGE'].mean()) / (demographic['AGE'] + 1e-12)

r = demographic[
    demographic["AGE"].between(
        demographic["AGE"].quantile(0.05), demographic["AGE"].quantile(0.95)
    )
]
demographic["AGE"] = (demographic["AGE"] - r["AGE"].mean()) / (r["AGE"].std() + 1e-12)

# %%
plt.scatter(demographic["PATIENT_ID"], demographic["AGE"], s=1)
plt.xlabel("Patient Id")
plt.ylabel("Age")
plt.title("Patient-Age Scatter Plot")

# %%
demographic.to_csv("./datasets/hm/demographic.csv", mode="w", index=False)
demographic.head()

# %% [markdown]
# ## Vital Signal

# %%
vital_signs = pd.read_csv(
    "./datasets/hm/raw_data/19_04_2021/COVID_DSL_02.CSV",
    encoding="ISO-8859-1",
    sep="|",
)
print(len(vital_signs))
vital_signs.head()

# %%
vital_signs = vital_signs.rename(
    columns={
        "IDINGRESO": "PATIENT_ID",
        "CONSTANTS_ING_DATE": "RECORD_DATE",
        "CONSTANTS_ING_TIME": "RECORD_TIME",
        "FC_HR_ING": "HEART_RATE",
        "GLU_GLY_ING": "BLOOD_GLUCOSE",
        "SAT_02_ING": "OXYGEN_SATURATION",
        "TA_MAX_ING": "MAX_BLOOD_PRESSURE",
        "TA_MIN_ING": "MIN_BLOOD_PRESSURE",
        "TEMP_ING": "TEMPERATURE",
    }
)
vital_signs["RECORD_TIME"] = (
    vital_signs["RECORD_DATE"] + " " + vital_signs["RECORD_TIME"]
)
vital_signs["RECORD_TIME"] = vital_signs["RECORD_TIME"].map(
    lambda x: str(datetime.datetime.strptime(x, "%Y-%m-%d %M:%S"))
)
vital_signs = vital_signs.drop(
    ["RECORD_DATE", "SAT_02_ING_OBS", "BLOOD_GLUCOSE"], axis=1
)

# %%
vital_signs.describe()

# %%
def format_temperature(x):
    if type(x) == str:
        return float(x.replace(",", "."))
    else:
        return float(x)


def format_oxygen(x):
    x = float(x)
    if x > 100:
        return np.nan
    else:
        return x


def format_heart_rate(x):
    x = int(x)
    if x > 220:
        return np.nan
    else:
        return x


vital_signs["TEMPERATURE"] = vital_signs["TEMPERATURE"].map(
    lambda x: format_temperature(x)
)
vital_signs["OXYGEN_SATURATION"] = vital_signs["OXYGEN_SATURATION"].map(
    lambda x: format_oxygen(x)
)
vital_signs["HEART_RATE"] = vital_signs["HEART_RATE"].map(
    lambda x: format_heart_rate(x)
)

# %%
vital_signs = vital_signs.replace(0, np.NAN)

# %%
vital_signs = vital_signs.groupby(
    ["PATIENT_ID", "RECORD_TIME"], dropna=True, as_index=False
).mean()
vital_signs.head()

# %%
vital_signs.describe()

# %%
vital_signs.describe().to_csv(
    "./datasets/hm/vital_signs_overview.csv", index=False, mode="w"
)
vital_signs.describe()

# %%
# plt.rcParams['figure.figsize'] = [10, 5]
fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="w", edgecolor="k")

plt.subplot(2, 3, 1)
plt.scatter(vital_signs.index, vital_signs["MAX_BLOOD_PRESSURE"], s=1)
plt.xlabel("Index")
plt.ylabel("Max Blood Pressure")
plt.title("Visit-Max Blood Pressure Scatter Plot")

plt.subplot(2, 3, 2)
plt.scatter(vital_signs.index, vital_signs["MIN_BLOOD_PRESSURE"], s=1)
plt.xlabel("Index")
plt.ylabel("Min Blood Pressure")
plt.title("Visit-Min Blood Pressure Scatter Plot")

plt.subplot(2, 3, 3)
plt.scatter(vital_signs.index, vital_signs["TEMPERATURE"], s=1)
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.title("Visit-Temperature Scatter Plot")

plt.subplot(2, 3, 4)
plt.scatter(vital_signs.index, vital_signs["HEART_RATE"], s=1)
plt.xlabel("Index")
plt.ylabel("Heart Rate")
plt.title("Visit-Heart Rate Scatter Plot")

plt.subplot(2, 3, 5)
plt.scatter(vital_signs.index, vital_signs["OXYGEN_SATURATION"], s=1)
plt.xlabel("Index")
plt.ylabel("Oxygen Saturation")
plt.title("Visit-Oxygen Saturation Scatter Plot")

plt.show()

# %%
# plt.rcParams['figure.figsize'] = [10, 5]
fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="w", edgecolor="k")

plt.subplot(2, 3, 1)
plt.hist(vital_signs["MAX_BLOOD_PRESSURE"], bins=30)
plt.xlabel("Index")
plt.ylabel("Max Blood Pressure")
plt.title("Visit-Max Blood Pressure Histogram")

plt.subplot(2, 3, 2)
plt.hist(vital_signs["MIN_BLOOD_PRESSURE"], bins=30)
plt.xlabel("Index")
plt.ylabel("Min Blood Pressure")
plt.title("Visit-Min Blood Pressure Histogram")

plt.subplot(2, 3, 3)
plt.hist(vital_signs["TEMPERATURE"], bins=30)
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.title("Visit-Temperature Histogram")

plt.subplot(2, 3, 4)
plt.hist(vital_signs["HEART_RATE"], bins=30)
plt.xlabel("Index")
plt.ylabel("Heart Rate")
plt.title("Visit-Heart Rate Histogram")

plt.subplot(2, 3, 5)
plt.hist(vital_signs["OXYGEN_SATURATION"], bins=30)
plt.xlabel("Index")
plt.ylabel("Oxygen Saturation")
plt.title("Visit-Oxygen Saturation Histogram")

plt.show()

# %% [markdown]
# ### Missing rate of each visit

# %%
sum(vital_signs.T.isnull().sum()) / ((len(vital_signs.T) - 2) * len(vital_signs))

# %% [markdown]
# ### Normalize data

# %%
"""
for key in vital_signs.keys()[2:]:
    vital_signs[key] = (vital_signs[key] - vital_signs[key].mean()) / (vital_signs[key].std() + 1e-12)

vital_signs.describe()
"""

# %%
for key in vital_signs.keys()[2:]:
    r = vital_signs[
        vital_signs[key].between(
            vital_signs[key].quantile(0.05), vital_signs[key].quantile(0.95)
        )
    ]
    vital_signs[key] = (vital_signs[key] - r[key].mean()) / (r[key].std() + 1e-12)

# %%
vital_signs.to_csv("./datasets/hm/visual_signs.csv", mode="w", index=False)

# %%
len(vital_signs) / len(vital_signs["PATIENT_ID"].unique())

# %% [markdown]
# ## Lab Tests

# %%
lab_tests = pd.read_csv(
    "./datasets/hm/raw_data/19_04_2021/COVID_DSL_06_v2.CSV",
    encoding="ISO-8859-1",
    sep=";",
)
lab_tests = lab_tests.rename(columns={"IDINGRESO": "PATIENT_ID"})
print(len(lab_tests))

# del useless data
lab_tests = lab_tests[
    [
        "PATIENT_ID",
        "LAB_NUMBER",
        "LAB_DATE",
        "TIME_LAB",
        "ITEM_LAB",
        "VAL_RESULT"
        # UD_RESULT: unit
        # REF_VALUES: reference values
    ]
]

lab_tests.head()

# %%
lab_tests = lab_tests.groupby(
    ["PATIENT_ID", "LAB_NUMBER", "LAB_DATE", "TIME_LAB", "ITEM_LAB"],
    dropna=True,
    as_index=False,
).first()
lab_tests = (
    lab_tests.set_index(
        ["PATIENT_ID", "LAB_NUMBER", "LAB_DATE", "TIME_LAB", "ITEM_LAB"], drop=True
    )
    .unstack("ITEM_LAB")["VAL_RESULT"]
    .reset_index()
)

lab_tests = lab_tests.drop(
    [
        "CFLAG -- ALARMA HEMOGRAMA",
        "CORONA -- PCR CORONAVIRUS 2019nCoV",
        "CRIOGLO -- CRIOGLOBULINAS",
        "EGCOVID -- ESTUDIO GENETICO COVID-19",
        "FRO1 -- ",
        "FRO1 -- FROTIS EN SANGRE PERIFERICA",
        "FRO2 -- ",
        "FRO2 -- FROTIS EN SANGRE PERIFERICA",
        "FRO3 -- ",
        "FRO3 -- FROTIS EN SANGRE PERIFERICA",
        "FRO_COMEN -- ",
        "FRO_COMEN -- FROTIS EN SANGRE PERIFERICA",
        "G-CORONAV (RT-PCR) -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR",
        "G-CORONAV (RT-PCR) -- Tipo de muestra: EXUDADO",
        "GRRH -- GRUPO SANGUÖNEO Y FACTOR Rh",
        "HEML -- RECUENTO CELULAR LIQUIDO",
        "HEML -- Recuento Hemat¡es",
        "IFSUERO -- INMUNOFIJACION EN SUERO",
        "OBS_BIOMOL -- OBSERVACIONES GENETICA MOLECULAR",
        "OBS_BIOO -- Observaciones Bioqu¡mica Orina",
        "OBS_CB -- Observaciones Coagulaci¢n",
        "OBS_GASES -- Observaciones Gasometr¡a Arterial",
        "OBS_GASV -- Observaciones Gasometr¡a Venosa",
        "OBS_GEN2 -- OBSERVACIONES GENETICA",
        "OBS_HOR -- Observaciones Hormonas",
        "OBS_MICRO -- Observaciones Microbiolog¡a",
        "OBS_NULA2 -- Observaciones Bioqu¡mica",
        "OBS_NULA3 -- Observaciones Hematolog¡a",
        "OBS_PESP -- Observaciones Pruebas especiales",
        "OBS_SERO -- Observaciones Serolog¡a",
        "OBS_SIS -- Observaciones Orina",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: BAS",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ESPUTO",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: EXUDADO",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO BRONCOALVEOLAR",
        "PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO NASOFARÖNGEO",
        "PTGOR -- PROTEINOGRAMA ORINA",
        "RESUL_IFT -- ESTUDIO DE INMUNOFENOTIPO",
        "RESUL_IFT -- Resultado",
        "Resultado -- Resultado",
        "SED1 -- ",
        "SED1 -- SEDIMENTO",
        "SED2 -- ",
        "SED2 -- SEDIMENTO",
        "SED3 -- ",
        "SED3 -- SEDIMENTO",
        "TIPOL -- TIPO DE LIQUIDO",
        "Tecnica -- T\x82cnica",
        "TpMues -- Tipo de muestra",
        "VHCBLOT -- INMUNOBLOT VIRUS HEPATITIS C",
        "VIR_TM -- VIRUS TIPO DE MUESTRA",
        "LEGIORI -- AG. LEGIONELA PNEUMOPHILA EN ORINA",
        "NEUMOORI -- AG NEUMOCOCO EN ORINA",
        "VIHAC -- VIH AC",
    ],
    axis=1,
)


lab_tests.head()

# %%
lab_tests = lab_tests.replace("Sin resultado.", np.nan)
lab_tests = lab_tests.replace("Sin resultado", np.nan)
lab_tests = lab_tests.replace("----", np.nan).replace("---", np.nan)
lab_tests = lab_tests.replace("> ", "").replace("< ", "")


def change_format(x):
    if x is None:
        return np.nan
    elif type(x) == str:
        if x.startswith("Negativo ("):
            return x.replace("Negativo (", "-")[:-1]
        elif x.startswith("Positivo ("):
            return x.replace("Positivo (", "")[:-1]
        elif x.startswith("Zona limite ("):
            return x.replace("Zona limite (", "")[:-1]
        elif x.startswith(">"):
            return x.replace("> ", "").replace(">", "")
        elif x.startswith("<"):
            return x.replace("< ", "").replace("<", "")
        elif x.endswith(" mg/dl"):
            return x.replace(" mg/dl", "")
        elif x.endswith("/æl"):
            return x.replace("/æl", "")
        elif x.endswith(" copias/mL"):
            return x.replace(" copias/mL", "")
        elif x == "Numerosos":
            return 1.5
        elif x == "Aislados":
            return 0.5
        elif (
            x == "Se detecta" or x == "Se observan" or x == "Normal" or x == "Positivo"
        ):
            return 1
        elif x == "No se detecta" or x == "No se observan" or x == "Negativo":
            return 0
        elif x == "Indeterminado":
            return np.nan
        else:
            num = re.findall("[-+]?\d+\.\d+", x)
            if len(num) == 0:
                return np.nan
            else:
                return num[0]
    else:
        return x


feature_value_dict = dict()

for k in tqdm(lab_tests.keys()[4:]):
    lab_tests[k] = lab_tests[k].map(lambda x: change_format(change_format(x)))
    feature_value_dict[k] = lab_tests[k].unique()

# %%
def nan_and_not_nan(x):
    if x == x:
        return 1
    else:  # nan
        return 0


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_all_float(x):
    for i in x:
        if i == i and (i != None):
            if not is_float(i):
                return False
    return True


def to_float(x):
    if x != None:
        return float(x)
    else:
        return np.nan


other_feature_dict = dict()

for feature in tqdm(feature_value_dict.keys()):
    values = feature_value_dict[feature]
    if is_all_float(values):
        lab_tests[feature] = lab_tests[feature].map(lambda x: to_float(x))
    elif len(values) == 2:
        lab_tests[feature] = lab_tests[feature].map(lambda x: nan_and_not_nan(x))
    else:
        other_feature_dict[feature] = values

# %%
other_feature_dict

# %%
def format_time(t):
    if "/" in t:
        return str(datetime.datetime.strptime(t, "%d/%m/%Y %M:%S"))
    else:
        return str(datetime.datetime.strptime(t, "%d-%m-%Y %M:%S"))


lab_tests["RECORD_TIME"] = lab_tests["LAB_DATE"] + " " + lab_tests["TIME_LAB"]
lab_tests["RECORD_TIME"] = lab_tests["RECORD_TIME"].map(lambda x: format_time(x))
lab_tests = lab_tests.drop(["LAB_NUMBER", "LAB_DATE", "TIME_LAB"], axis=1)
# lab_tests = lab_tests.drop(['LAB_NUMBER', 'TIME_LAB'], axis=1)
lab_tests.head()

# %%
lab_tests_patient = lab_tests.groupby(
    ["PATIENT_ID"], dropna=True, as_index=False
).mean()
print(len(lab_tests_patient))
count = [i for i in lab_tests_patient.count()[1:]]
plt.hist(count)

# %%
patient_total = len(lab_tests_patient)
threshold = patient_total * 0.1
reserved_keys = []

for key in lab_tests_patient.keys():
    if lab_tests_patient[key].count() > threshold:
        reserved_keys.append(key)

print(len(reserved_keys))
reserved_keys

# %%
reserved_keys.insert(1, "RECORD_TIME")

lab_tests = lab_tests.groupby(
    ["PATIENT_ID", "RECORD_TIME"], dropna=True, as_index=False
).mean()

lab_tests = lab_tests[reserved_keys]
lab_tests.head()

# %% [markdown]
# ### Missing rate of each visit

# %%
sum(lab_tests.T.isnull().sum()) / ((len(lab_tests.T) - 2) * len(lab_tests))

# %% [markdown]
# ### Scatter Plot

# %%
fig = plt.figure(figsize=(16, 200), dpi=100, facecolor="w", edgecolor="k")

i = 1
for key in lab_tests.keys()[2:]:
    plt.subplot(33, 3, i)
    plt.scatter(lab_tests.index, lab_tests[key], s=1)
    plt.ylabel(key)
    i += 1

plt.show()

# %%
fig = plt.figure(figsize=(20, 120), dpi=100, facecolor="w", edgecolor="k")

i = 1
for key in lab_tests.keys()[2:]:
    plt.subplot(23, 4, i)
    plt.hist(lab_tests[key], bins=30)
    q3 = lab_tests[key].quantile(0.75)
    q1 = lab_tests[key].quantile(0.25)
    qh = q3 + 3 * (q3 - q1)
    ql = q1 - 3 * (q3 - q1)
    sigma = 5
    plt.axline(
        [sigma * lab_tests[key].std() + lab_tests[key].mean(), 0],
        [sigma * lab_tests[key].std() + lab_tests[key].mean(), 1],
        color="r",
        linestyle=(0, (5, 5)),
    )
    plt.axline(
        [-sigma * lab_tests[key].std() + lab_tests[key].mean(), 0],
        [-sigma * lab_tests[key].std() + lab_tests[key].mean(), 1],
        color="r",
        linestyle=(0, (5, 5)),
    )
    # plt.axline([lab_tests[key].quantile(0.25), 0], [lab_tests[key].quantile(0.25), 1], color = "k", linestyle=(0, (5, 5)))
    # plt.axline([lab_tests[key].quantile(0.75), 0], [lab_tests[key].quantile(0.75), 1], color = "k", linestyle=(0, (5, 5)))
    plt.axline([qh, 0], [qh, 1], color="k", linestyle=(0, (5, 5)))
    plt.axline([ql, 0], [ql, 1], color="k", linestyle=(0, (5, 5)))
    plt.ylabel(key)
    i += 1

plt.show()

# %% [markdown]
# ### Normalize data

# %%
"""
for key in lab_tests.keys()[2:]:
    lab_tests[key] = (lab_tests[key] - lab_tests[key].mean()) / (lab_tests[key].std() + 1e-12)

lab_tests.describe()
"""

# %%
for key in lab_tests.keys()[2:]:
    r = lab_tests[
        lab_tests[key].between(
            lab_tests[key].quantile(0.05), lab_tests[key].quantile(0.95)
        )
    ]
    lab_tests[key] = (lab_tests[key] - r[key].mean()) / (r[key].std() + 1e-12)

# %%
lab_tests.to_csv("./datasets/hm/lab_test.csv", mode="w", index=False)

# %% [markdown]
# # Concat data

# %%
demographic["PATIENT_ID"] = demographic["PATIENT_ID"].map(lambda x: str(int(x)))
vital_signs["PATIENT_ID"] = vital_signs["PATIENT_ID"].map(lambda x: str(int(x)))
lab_tests["PATIENT_ID"] = lab_tests["PATIENT_ID"].map(lambda x: str(int(x)))

# %%
len(demographic["PATIENT_ID"].unique()), len(vital_signs["PATIENT_ID"].unique()), len(
    lab_tests["PATIENT_ID"].unique()
)

# %%
train_df = pd.merge(
    vital_signs, lab_tests, on=["PATIENT_ID", "RECORD_TIME"], how="outer"
)

train_df = train_df.groupby(
    ["PATIENT_ID", "RECORD_TIME"], dropna=True, as_index=False
).mean()

train_df = pd.merge(demographic, train_df, on=["PATIENT_ID"], how="left")

train_df.head()

# %%
# del rows without patient_id, admission_date, record_time, or outcome
train_df = train_df.dropna(
    axis=0, how="any", subset=["PATIENT_ID", "ADMISSION_DATE", "RECORD_TIME", "OUTCOME"]
)

# %%
train_df.to_csv("./datasets/hm/train.csv", mode="w", index=False)
train_df.describe()

# %% [markdown]
# ## Missing rate of each visit

# %%
sum(train_df.T.isnull().sum()) / ((len(train_df.T) - 2) * len(train_df))

# %% [markdown]
# # Split and save data

# %% [markdown]
# * demo: demographic data
# * x: lab test & vital signs
# * y: outcome & length of stay

# %%
patient_ids = train_df["PATIENT_ID"].unique()

demo_cols = [
    "AGE",
    "SEX",
]  # , 'DIFFICULTY_BREATHING', 'FEVER', 'SUSPECT_COVID', 'EMERGENCY'
test_cols = []

for k in train_df.keys():
    if not k in demographic.keys():
        if not k == "RECORD_TIME":
            test_cols.append(k)

test_median = train_df[test_cols].median()

# %%
train_df["RECORD_TIME"] = train_df["RECORD_TIME"].map(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H")
)

# %%
train_df = train_df.groupby(
    ["PATIENT_ID", "ADMISSION_DATE", "DEPARTURE_DATE", "RECORD_TIME"],
    dropna=True,
    as_index=False,
).mean()

train_df.head()

# %%
def fill_nan(cur, prev):
    l = len(prev)
    miss = []
    for idx in range(l):
        # print(cur[idx])
        if np.isnan(cur[idx]):
            cur[idx] = prev[idx]
            miss.append(1)
        else:
            miss.append(0)
    return miss


def init_prev(prev):
    miss = []
    l = len(prev)
    for idx in range(l):
        # print(prev[idx])
        # print(type(prev[idx]))
        if np.isnan(prev[idx]):
            prev[idx] = test_median[idx]
            miss.append(1)
        else:
            miss.append(0)
    return miss


# %%
# max_visit = 300
x, y, demo, x_lab_len, missing_mask = [], [], [], [], []

for pat in tqdm(patient_ids):
    info = train_df[train_df["PATIENT_ID"] == pat]
    info = info[max(0, len(info) - 24) :]
    indexes = info.index
    visit = info.loc[indexes[0]]

    # demographic data
    demo.append([visit[k] for k in demo_cols])

    # label
    outcome = visit["OUTCOME"]
    los = []

    # lab test & vital signs
    tests = []
    prev = visit[test_cols]
    miss = []
    miss.append(init_prev(prev))
    leave = datetime.datetime.strptime(visit["DEPARTURE_DATE"], "%Y-%m-%d %H:%M:%S")
    first = True
    for i in indexes:
        visit = info.loc[i]
        now = datetime.datetime.strptime(visit["RECORD_TIME"], "%Y-%m-%d %H")
        cur = visit[test_cols]
        tmp = fill_nan(cur, prev)
        if not first:
            miss.append(tmp)
        tests.append(cur)
        los_visit = (leave - now).days
        if los_visit < 0:
            los_visit = 0
        los.append(los_visit)
        prev = cur
        first = False

    valid_visit = len(los)
    # outcome = [outcome] * valid_visit
    x_lab_len.append(valid_visit)
    missing_mask.append(miss)

    # tests = np.pad(tests, ((0, max_visit - valid_visit), (0, 0)))
    # outcome = np.pad(outcome, (0, max_visit - valid_visit))
    # los = np.pad(los, (0, max_visit - valid_visit))

    y.append([outcome, los])
    x.append(tests)

# %%
all_x = x
all_x_demo = demo
all_y = y
all_missing_mask = missing_mask

# %%
all_x_labtest = np.array(all_x, dtype=object)
x_lab_length = [len(_) for _ in all_x_labtest]
x_lab_length = torch.tensor(x_lab_length, dtype=torch.int)
max_length = int(x_lab_length.max())
all_x_labtest = [torch.tensor(_) for _ in all_x_labtest]
all_x_labtest = torch.nn.utils.rnn.pad_sequence((all_x_labtest), batch_first=True)
all_x_demographic = torch.tensor(all_x_demo)
batch_size, demo_dim = all_x_demographic.shape
all_x_demographic = torch.reshape(
    all_x_demographic.repeat(1, max_length), (batch_size, max_length, demo_dim)
)
all_x = torch.cat((all_x_demographic, all_x_labtest), 2)

all_y = np.array(all_y, dtype=object)
patient_list = []
for pat in all_y:
    visits = []
    for i in pat[1]:
        visits.append([pat[0], i])
    patient_list.append(visits)
new_all_y = np.array(patient_list, dtype=object)
output_all_y = [torch.Tensor(_) for _ in new_all_y]
output_all_y = torch.nn.utils.rnn.pad_sequence((output_all_y), batch_first=True)

# %%
all_missing_mask = np.array(all_missing_mask, dtype=object)
all_missing_mask = [torch.tensor(_) for _ in all_missing_mask]
all_missing_mask = torch.nn.utils.rnn.pad_sequence((all_missing_mask), batch_first=True)

# %%
all_x.shape

# %%
all_missing_mask.shape

# %%
# save pickle format dataset (torch)
pd.to_pickle(all_x, f"./datasets/hm/processed_data/x.pkl")
pd.to_pickle(all_missing_mask, f"./datasets/hm/processed_data/missing_mask.pkl")
pd.to_pickle(output_all_y, f"./datasets/hm/processed_data/y.pkl")
pd.to_pickle(x_lab_length, f"./datasets/hm/processed_data/visits_length.pkl")

# %%
# Calculate patients' outcome statistics (patients-wise)
outcome_list = []
y_outcome = output_all_y[:, :, 0]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    outcome_list.append(y_outcome[i][0].item())
outcome_list = np.array(outcome_list)
print(len(outcome_list))
unique, count = np.unique(outcome_list, return_counts=True)
data_count = dict(zip(unique, count))
print(data_count)

# %%
# Calculate patients' outcome statistics (records-wise)
outcome_records_list = []
y_outcome = output_all_y[:, :, 0]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    outcome_records_list.extend(y_outcome[i][0 : x_lab_length[i]].tolist())
outcome_records_list = np.array(outcome_records_list)
print(len(outcome_records_list))
unique, count = np.unique(outcome_records_list, return_counts=True)
data_count = dict(zip(unique, count))
print(data_count)

# %%
# Calculate patients' mean los and 95% percentile los
los_list = []
y_los = output_all_y[:, :, 1]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    # los_list.extend(y_los[i][: x_lab_length[i].long()].tolist())
    los_list.append(y_los[i][0].item())
los_list = np.array(los_list)
print(los_list.mean() * 0.5)
print(np.median(los_list) * 0.5)
print(np.percentile(los_list, 95))

print("median:", np.median(los_list))
print("Q1:", np.percentile(los_list, 25))
print("Q3:", np.percentile(los_list, 75))

# %%
los_alive_list = np.array(
    [los_list[i] for i in range(len(los_list)) if outcome_list[i] == 0]
)
los_dead_list = np.array(
    [los_list[i] for i in range(len(los_list)) if outcome_list[i] == 1]
)
print(len(los_alive_list))
print(len(los_dead_list))

print("[Alive]")
print("median:", np.median(los_alive_list))
print("Q1:", np.percentile(los_alive_list, 25))
print("Q3:", np.percentile(los_alive_list, 75))

print("[Dead]")
print("median:", np.median(los_dead_list))
print("Q1:", np.percentile(los_dead_list, 25))
print("Q3:", np.percentile(los_dead_list, 75))

# %%
cdsl_los_statistics = {
    "overall": los_list,
    "alive": los_alive_list,
    "dead": los_dead_list,
}
pd.to_pickle(cdsl_los_statistics, "cdsl_los_statistics.pkl")

# %%
# calculate visits length Median [Q1, Q3]
visits_list = np.array(x_lab_length)
visits_alive_list = np.array(
    [x_lab_length[i] for i in range(len(x_lab_length)) if outcome_list[i] == 0]
)
visits_dead_list = np.array(
    [x_lab_length[i] for i in range(len(x_lab_length)) if outcome_list[i] == 1]
)
print(len(visits_alive_list))
print(len(visits_dead_list))

print("[Total]")
print("median:", np.median(visits_list))
print("Q1:", np.percentile(visits_list, 25))
print("Q3:", np.percentile(visits_list, 75))

print("[Alive]")
print("median:", np.median(visits_alive_list))
print("Q1:", np.percentile(visits_alive_list, 25))
print("Q3:", np.percentile(visits_alive_list, 75))

print("[Dead]")
print("median:", np.median(visits_dead_list))
print("Q1:", np.percentile(visits_dead_list, 25))
print("Q3:", np.percentile(visits_dead_list, 75))

# %%
def check_nan(x):
    if np.isnan(np.sum(x.cpu().numpy())):
        print("some values from input are nan")
    else:
        print("no nan")


# %%
check_nan(all_x)
