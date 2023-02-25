# %% [markdown]
# # hm dataset pre-processing
# 
# import packages

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import math
import datetime
from tqdm import tqdm
import datetime
import re
from functools import reduce

# %% [markdown]
# ## Demographic data

# %%
demographic = pd.read_csv('./raw_data/19_04_2021/COVID_DSL_01.CSV', encoding='ISO-8859-1', sep='|')
print(len(demographic))
demographic.head()

# %%
med = pd.read_csv('./raw_data/19_04_2021/COVID_DSL_04.CSV', encoding='ISO-8859-1', sep='|')
print(len(med))
med.head()

# %%
len(med['ID_ATC7'].unique())

# %% [markdown]
# get rid of patient with missing label

# %%
print(len(demographic))
demographic = demographic.dropna(axis=0, how='any', subset=['IDINGRESO', 'F_INGRESO_ING', 'F_ALTA_ING', 'MOTIVO_ALTA_ING'])
print(len(demographic))

# %%
def outcome2num(x):
    if x == 'Fallecimiento':
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
            'IDINGRESO', 
            'EDAD',
            'SEX',
            'F_INGRESO_ING', 
            'F_ALTA_ING', 
            'MOTIVO_ALTA_ING', 
            'ESPECIALIDAD_URGENCIA', 
            'DIAG_URG'
        ]
    ]

# rename column
demographic = demographic.rename(columns={
    'IDINGRESO': 'PATIENT_ID',
    'EDAD': 'AGE',
    'SEX': 'SEX',
    'F_INGRESO_ING': 'ADMISSION_DATE',
    'F_ALTA_ING': 'DEPARTURE_DATE',
    'MOTIVO_ALTA_ING': 'OUTCOME',
    'ESPECIALIDAD_URGENCIA': 'DEPARTMENT_OF_EMERGENCY',
    'DIAG_URG': 'DIAGNOSIS_AT_EMERGENCY_VISIT'
})

# SEX: male: 1; female: 0
demographic['SEX'].replace('MALE', 1, inplace=True)
demographic['SEX'].replace('FEMALE', 0, inplace=True)

# outcome: Fallecimiento(dead): 1; others: 0
demographic['OUTCOME'] = demographic['OUTCOME'].map(outcome2num)

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
            'PATIENT_ID',
            'AGE',
            'SEX',
            'ADMISSION_DATE',
            'DEPARTURE_DATE',
            'OUTCOME',
            # 'DIFFICULTY_BREATHING',
            # 'SUSPECT_COVID',
            # 'FEVER',
            # 'EMERGENCY'
        ]
    ]

# %%
demographic.describe().to_csv('demographic_overview.csv', mode='w', index=False)
demographic.describe()

# %% [markdown]
# ### Analyze data

# %%
plt.scatter(demographic['PATIENT_ID'], demographic['AGE'], s=1)
plt.xlabel('Patient Id')
plt.ylabel('Age')
plt.title('Patient-Age Scatter Plot')

# %%
plt.scatter(demographic['PATIENT_ID'], demographic['AGE'], s=1)
plt.xlabel('Patient Id')
plt.ylabel('Age')
plt.title('Patient-Age Scatter Plot')

# %%
demographic.to_csv('demographic.csv', mode='w', index=False)
demographic.head()

# %% [markdown]
# ## Vital Signal

# %%
vital_signs = pd.read_csv('./raw_data/19_04_2021/COVID_DSL_02.CSV', encoding='ISO-8859-1', sep='|')
print(len(vital_signs))
vital_signs.head()

# %%
vital_signs = vital_signs.rename(columns={
    'IDINGRESO': 'PATIENT_ID',
    'CONSTANTS_ING_DATE': 'RECORD_DATE',
    'CONSTANTS_ING_TIME': 'RECORD_TIME',
    'FC_HR_ING': 'HEART_RATE',
    'GLU_GLY_ING': 'BLOOD_GLUCOSE',
    'SAT_02_ING': 'OXYGEN_SATURATION',
    'TA_MAX_ING': 'MAX_BLOOD_PRESSURE',
    'TA_MIN_ING': 'MIN_BLOOD_PRESSURE',
    'TEMP_ING': 'TEMPERATURE'
})
vital_signs['RECORD_TIME'] = vital_signs['RECORD_DATE'] + ' ' + vital_signs['RECORD_TIME']
vital_signs['RECORD_TIME'] = vital_signs['RECORD_TIME'].map(lambda x: str(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')))
vital_signs = vital_signs.drop(['RECORD_DATE', 'SAT_02_ING_OBS', 'BLOOD_GLUCOSE'], axis=1)

# %%
vital_signs.describe()

# %%
vital_signs.head()

# %%
def format_temperature(x):
    if type(x) == str:
        return float(x.replace(',', '.'))
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

vital_signs['TEMPERATURE'] = vital_signs['TEMPERATURE'].map(lambda x: format_temperature(x))
vital_signs['OXYGEN_SATURATION'] = vital_signs['OXYGEN_SATURATION'].map(lambda x: format_oxygen(x))
vital_signs['HEART_RATE'] = vital_signs['HEART_RATE'].map(lambda x: format_heart_rate(x))

# %%
vital_signs = vital_signs.replace(0, np.NAN)

# %%
vital_signs = vital_signs.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()
vital_signs.head()

# %%
vital_signs.describe()

# %%
vital_signs.describe().to_csv('vital_signs_overview.csv', index=False, mode='w')
vital_signs.describe()

# %%
"""
#plt.rcParams['figure.figsize'] = [10, 5]
fig=plt.figure(figsize=(16,10), dpi= 100, facecolor='w', edgecolor='k')

plt.subplot(2, 3, 1)
plt.scatter(vital_signs.index, vital_signs['MAX_BLOOD_PRESSURE'], s=1)
plt.xlabel('Index')
plt.ylabel('Max Blood Pressure')
plt.title('Visit-Max Blood Pressure Scatter Plot')

plt.subplot(2, 3, 2)
plt.scatter(vital_signs.index, vital_signs['MIN_BLOOD_PRESSURE'], s=1)
plt.xlabel('Index')
plt.ylabel('Min Blood Pressure')
plt.title('Visit-Min Blood Pressure Scatter Plot')

plt.subplot(2, 3, 3)
plt.scatter(vital_signs.index, vital_signs['TEMPERATURE'], s=1)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.title('Visit-Temperature Scatter Plot')

plt.subplot(2, 3, 4)
plt.scatter(vital_signs.index, vital_signs['HEART_RATE'], s=1)
plt.xlabel('Index')
plt.ylabel('Heart Rate')
plt.title('Visit-Heart Rate Scatter Plot')

plt.subplot(2, 3, 5)
plt.scatter(vital_signs.index, vital_signs['OXYGEN_SATURATION'], s=1)
plt.xlabel('Index')
plt.ylabel('Oxygen Saturation')
plt.title('Visit-Oxygen Saturation Scatter Plot')

plt.show()
"""
# %%
"""
#plt.rcParams['figure.figsize'] = [10, 5]
fig=plt.figure(figsize=(16,10), dpi= 100, facecolor='w', edgecolor='k')

plt.subplot(2, 3, 1)
plt.hist(vital_signs['MAX_BLOOD_PRESSURE'], bins=30)
plt.xlabel('Index')
plt.ylabel('Max Blood Pressure')
plt.title('Visit-Max Blood Pressure Histogram')

plt.subplot(2, 3, 2)
plt.hist(vital_signs['MIN_BLOOD_PRESSURE'], bins=30)
plt.xlabel('Index')
plt.ylabel('Min Blood Pressure')
plt.title('Visit-Min Blood Pressure Histogram')

plt.subplot(2, 3, 3)
plt.hist(vital_signs['TEMPERATURE'], bins=30)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.title('Visit-Temperature Histogram')

plt.subplot(2, 3, 4)
plt.hist(vital_signs['HEART_RATE'], bins=30)
plt.xlabel('Index')
plt.ylabel('Heart Rate')
plt.title('Visit-Heart Rate Histogram')

plt.subplot(2, 3, 5)
plt.hist(vital_signs['OXYGEN_SATURATION'], bins=30)
plt.xlabel('Index')
plt.ylabel('Oxygen Saturation')
plt.title('Visit-Oxygen Saturation Histogram')

plt.show()
"""
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
vital_signs.to_csv('visual_signs.csv', mode='w', index=False)

# %%
len(vital_signs) / len(vital_signs['PATIENT_ID'].unique())

# %% [markdown]
# ## Lab Tests

# %%
lab_tests = pd.read_csv('./raw_data/19_04_2021/COVID_DSL_06_v2.CSV', encoding='ISO-8859-1', sep=';')
lab_tests = lab_tests.rename(columns={'IDINGRESO': 'PATIENT_ID'})
print(len(lab_tests))

# del useless data
lab_tests = lab_tests[
        [
            'PATIENT_ID',
            'LAB_NUMBER',
            'LAB_DATE',
            'TIME_LAB',
            'ITEM_LAB',
            'VAL_RESULT'
            # UD_RESULT: unit
            # REF_VALUES: reference values
        ]
    ]

lab_tests.head()

# %%
lab_tests = lab_tests.groupby(['PATIENT_ID', 'LAB_NUMBER', 'LAB_DATE', 'TIME_LAB', 'ITEM_LAB'], dropna=True, as_index = False).first()
lab_tests = lab_tests.set_index(['PATIENT_ID', 'LAB_NUMBER', 'LAB_DATE', 'TIME_LAB', 'ITEM_LAB'], drop = True).unstack('ITEM_LAB')['VAL_RESULT'].reset_index()

lab_tests = lab_tests.drop([
    'CFLAG -- ALARMA HEMOGRAMA', 
    'CORONA -- PCR CORONAVIRUS 2019nCoV', 
    'CRIOGLO -- CRIOGLOBULINAS',
    'EGCOVID -- ESTUDIO GENETICO COVID-19',
    'FRO1 -- ',
    'FRO1 -- FROTIS EN SANGRE PERIFERICA',
    'FRO2 -- ',
    'FRO2 -- FROTIS EN SANGRE PERIFERICA',
    'FRO3 -- ',
    'FRO3 -- FROTIS EN SANGRE PERIFERICA',
    'FRO_COMEN -- ',
    'FRO_COMEN -- FROTIS EN SANGRE PERIFERICA',
    'G-CORONAV (RT-PCR) -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR',
    'G-CORONAV (RT-PCR) -- Tipo de muestra: EXUDADO',
    'GRRH -- GRUPO SANGUÖNEO Y FACTOR Rh',
    'HEML -- RECUENTO CELULAR LIQUIDO',
    'HEML -- Recuento Hemat¡es',
    'IFSUERO -- INMUNOFIJACION EN SUERO',
    'OBS_BIOMOL -- OBSERVACIONES GENETICA MOLECULAR',
    'OBS_BIOO -- Observaciones Bioqu¡mica Orina',
    'OBS_CB -- Observaciones Coagulaci¢n',
    'OBS_GASES -- Observaciones Gasometr¡a Arterial',
    'OBS_GASV -- Observaciones Gasometr¡a Venosa',
    'OBS_GEN2 -- OBSERVACIONES GENETICA',
    'OBS_HOR -- Observaciones Hormonas',
    'OBS_MICRO -- Observaciones Microbiolog¡a',
    'OBS_NULA2 -- Observaciones Bioqu¡mica',
    'OBS_NULA3 -- Observaciones Hematolog¡a',
    'OBS_PESP -- Observaciones Pruebas especiales',
    'OBS_SERO -- Observaciones Serolog¡a',
    'OBS_SIS -- Observaciones Orina',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ASPIRADO BRONCOALVEOLAR',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: BAS',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: ESPUTO',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: EXUDADO',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO BRONCOALVEOLAR',
    'PCR VIRUS RESPIRATORIOS -- Tipo de muestra: LAVADO NASOFARÖNGEO',
    'PTGOR -- PROTEINOGRAMA ORINA',
    'RESUL_IFT -- ESTUDIO DE INMUNOFENOTIPO',
    'RESUL_IFT -- Resultado',
    'Resultado -- Resultado',
    'SED1 -- ',
    'SED1 -- SEDIMENTO',
    'SED2 -- ',
    'SED2 -- SEDIMENTO',
    'SED3 -- ',
    'SED3 -- SEDIMENTO',
    'TIPOL -- TIPO DE LIQUIDO',
    'Tecnica -- T\x82cnica',
    'TpMues -- Tipo de muestra',
    'VHCBLOT -- INMUNOBLOT VIRUS HEPATITIS C',
    'VIR_TM -- VIRUS TIPO DE MUESTRA',
    'LEGIORI -- AG. LEGIONELA PNEUMOPHILA EN ORINA',
    'NEUMOORI -- AG NEUMOCOCO EN ORINA',
    'VIHAC -- VIH AC'
    ], axis=1)

    
lab_tests.head()

# %%
lab_tests = lab_tests.replace('Sin resultado.', np.nan)
lab_tests = lab_tests.replace('Sin resultado', np.nan)
lab_tests = lab_tests.replace('----', np.nan).replace('---', np.nan)
lab_tests = lab_tests.replace('> ', '').replace('< ', '')

def change_format(x):
    if x is None:
        return np.nan
    elif type(x) == str:
        if x.startswith('Negativo ('):
            return x.replace('Negativo (', '-')[:-1]
        elif x.startswith('Positivo ('):
            return x.replace('Positivo (', '')[:-1]
        elif x.startswith('Zona limite ('):
            return x.replace('Zona limite (', '')[:-1]
        elif x.startswith('>'):
            return x.replace('> ', '').replace('>', '')
        elif x.startswith('<'):
            return x.replace('< ', '').replace('<', '')
        elif x.endswith(' mg/dl'):
            return x.replace(' mg/dl', '')
        elif x.endswith('/æl'):
            return x.replace('/æl', '')
        elif x.endswith(' copias/mL'):
            return x.replace(' copias/mL', '')
        elif x == 'Numerosos':
            return 1.5
        elif x == 'Aislados':
            return 0.5
        elif x == 'Se detecta' or x == 'Se observan' or x == 'Normal' or x == 'Positivo':
            return 1
        elif x == 'No se detecta' or x == 'No se observan' or x == 'Negativo':
            return 0
        elif x == 'Indeterminado':
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
    else: # nan
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
    if '/' in t:
        return str(datetime.datetime.strptime(t, '%d/%m/%Y %H:%M'))
    else:
        return str(datetime.datetime.strptime(t, '%d-%m-%Y %H:%M'))

lab_tests['RECORD_TIME'] = lab_tests['LAB_DATE'] + ' ' + lab_tests['TIME_LAB']
lab_tests['RECORD_TIME'] = lab_tests['RECORD_TIME'].map(lambda x: format_time(x))
lab_tests = lab_tests.drop(['LAB_NUMBER', 'LAB_DATE', 'TIME_LAB'], axis=1)
# lab_tests = lab_tests.drop(['LAB_NUMBER', 'TIME_LAB'], axis=1)
lab_tests.head()

# %%
lab_tests_patient = lab_tests.groupby(['PATIENT_ID'], dropna=True, as_index = False).mean()
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
reserved_keys.insert(1, 'RECORD_TIME')

lab_tests = lab_tests.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()

lab_tests = lab_tests[reserved_keys]
lab_tests.head()
"""
# %% [markdown]
# ### Missing rate of each visit

# %%
sum(lab_tests.T.isnull().sum()) / ((len(lab_tests.T) - 2) * len(lab_tests))

# %% [markdown]
# ### Scatter Plot

# %%
fig=plt.figure(figsize=(16,200), dpi= 100, facecolor='w', edgecolor='k')

i = 1
for key in lab_tests.keys()[2:]:
    plt.subplot(33, 3, i)
    plt.scatter(lab_tests.index, lab_tests[key], s=1)
    plt.ylabel(key)
    i += 1

plt.show()

# %%
fig=plt.figure(figsize=(20,120), dpi= 100, facecolor='w', edgecolor='k')

i = 1
for key in lab_tests.keys()[2:]:
    plt.subplot(23, 4, i)
    plt.hist(lab_tests[key], bins=30)
    q3 = lab_tests[key].quantile(0.75)
    q1 = lab_tests[key].quantile(0.25)
    qh = q3 + 3 * (q3 - q1)
    ql = q1 - 3 * (q3 - q1)
    sigma = 5
    plt.axline([sigma*lab_tests[key].std() + lab_tests[key].mean(), 0], [sigma*lab_tests[key].std() + lab_tests[key].mean(), 1], color = "r", linestyle=(0, (5, 5)))
    plt.axline([-sigma*lab_tests[key].std() + lab_tests[key].mean(), 0], [-sigma*lab_tests[key].std() + lab_tests[key].mean(), 1], color = "r", linestyle=(0, (5, 5)))
    #plt.axline([lab_tests[key].quantile(0.25), 0], [lab_tests[key].quantile(0.25), 1], color = "k", linestyle=(0, (5, 5)))
    #plt.axline([lab_tests[key].quantile(0.75), 0], [lab_tests[key].quantile(0.75), 1], color = "k", linestyle=(0, (5, 5)))
    plt.axline([qh, 0], [qh, 1], color='k', linestyle=(0, (5, 5)))
    plt.axline([ql, 0], [ql, 1], color='k', linestyle=(0, (5, 5)))
    plt.ylabel(key)
    i += 1

plt.show()
"""
# %% [markdown]
# ### Normalize data

# %%
"""
for key in lab_tests.keys()[2:]:
    lab_tests[key] = (lab_tests[key] - lab_tests[key].mean()) / (lab_tests[key].std() + 1e-12)

lab_tests.describe()
"""

# %%
# 【del normalization】
# for key in lab_tests.keys()[2:]:
#     r = lab_tests[lab_tests[key].between(lab_tests[key].quantile(0.05), lab_tests[key].quantile(0.95))]
#     lab_tests[key] = (lab_tests[key] - r[key].mean()) / (r[key].std() + 1e-12)

# %%
lab_tests.to_csv('lab_test.csv', mode='w', index=False)

# %% [markdown]
# # Concat data

# %%
demographic['PATIENT_ID'] = demographic['PATIENT_ID'].map(lambda x: str(int(x)))
vital_signs['PATIENT_ID'] = vital_signs['PATIENT_ID'].map(lambda x: str(int(x)))
lab_tests['PATIENT_ID'] = lab_tests['PATIENT_ID'].map(lambda x: str(int(x)))

# %%
len(demographic['PATIENT_ID'].unique()), len(vital_signs['PATIENT_ID'].unique()), len(lab_tests['PATIENT_ID'].unique())

# %%
train_df = pd.merge(vital_signs, lab_tests, on=['PATIENT_ID', 'RECORD_TIME'], how='outer')

train_df = train_df.groupby(['PATIENT_ID', 'RECORD_TIME'], dropna=True, as_index = False).mean()

train_df = pd.merge(demographic, train_df, on=['PATIENT_ID'], how='left')

train_df.head()

# %%
# del rows without patient_id, admission_date, record_time, or outcome
train_df = train_df.dropna(axis=0, how='any', subset=['PATIENT_ID', 'ADMISSION_DATE', 'RECORD_TIME', 'OUTCOME'])

# %%
train_df.to_csv('train.csv', mode='w', index=False)
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
patient_ids = train_df['PATIENT_ID'].unique()

demo_cols = ['AGE', 'SEX'] # , 'DIFFICULTY_BREATHING', 'FEVER', 'SUSPECT_COVID', 'EMERGENCY'
test_cols = []

# get column names
for k in train_df.keys():
    if not k in demographic.keys():
        if not k == 'RECORD_TIME':
            test_cols.append(k)

test_median = train_df[test_cols].median()

# %%
test_cols

# %%
train_df['RECORD_TIME_DAY'] = train_df['RECORD_TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
train_df['RECORD_TIME_HOUR'] = train_df['RECORD_TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H'))
train_df.head()

# %%
train_df_day = train_df.groupby(['PATIENT_ID', 'ADMISSION_DATE', 'DEPARTURE_DATE', 'RECORD_TIME_DAY'], dropna=True, as_index = False).mean()
train_df_hour = train_df.groupby(['PATIENT_ID', 'ADMISSION_DATE', 'DEPARTURE_DATE', 'RECORD_TIME_HOUR'], dropna=True, as_index = False).mean()

len(train_df), len(train_df_day), len(train_df_hour)

# %% [markdown]
# 
# ```
# number of visits (total)
# - Original data: 168777
# - Merge by hour: 130141
# - Merge by day:  42204
# ```

# %%
len(train_df['PATIENT_ID'].unique())

# %%
def get_visit_intervals(df):
    ls = []
    for pat in df['PATIENT_ID'].unique():
        ls.append(len(df[df['PATIENT_ID'] == pat]))
    return ls

# %%
ls_org = get_visit_intervals(train_df)
ls_hour = get_visit_intervals(train_df_hour)
ls_day = get_visit_intervals(train_df_day)

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
"""
csfont = {'fontname':'Times New Roman', 'fontsize': 18}
font = 'Times New Roman'
fig=plt.figure(figsize=(18,4), dpi= 100, facecolor='w', edgecolor='k')
plt.style.use('seaborn-whitegrid')
color = 'cornflowerblue'
ec = 'None'
alpha=0.5

ax = plt.subplot(1, 3, 1)
ax.hist(ls_org, bins=20, weights=np.ones(len(ls_org)) / len(ls_org), color=color, ec=ec, alpha=alpha, label='overall')
plt.xlabel('Num of visits (org)',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(**csfont)
plt.yticks(**csfont)

ax = plt.subplot(1, 3, 2)
ax.hist(ls_hour, bins=20, weights=np.ones(len(ls_hour)) / len(ls_hour), color=color, ec=ec, alpha=alpha, label='overall')
plt.xlabel('Num of visits (hour)',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(**csfont)
plt.yticks(**csfont)

ax = plt.subplot(1, 3, 3)
ax.hist(ls_day, bins=20, weights=np.ones(len(ls_day)) / len(ls_day), color=color, ec=ec, alpha=alpha, label='overall')
plt.xlabel('Num of visits (day)',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(**csfont)
plt.yticks(**csfont)

plt.show()
"""
# %%
def get_statistic(lst, name):
    print(f'[{name}]\tMax:\t{max(lst)}, Min:\t{min(lst)}, Median:\t{np.median(lst)}, Mean:\t{np.mean(lst)}, 80%:\t{np.quantile(lst, 0.8)}, 90%:\t{np.quantile(lst, 0.9)}, 95%:\t{np.quantile(lst, 0.95)}')

# %%
get_statistic(ls_org, 'ls_org')
get_statistic(ls_hour, 'ls_hour')
get_statistic(ls_day, 'ls_day')

# %%
train_df_hour['LOS'] = train_df_hour['ADMISSION_DATE']
train_df_hour['LOS_HOUR'] = train_df_hour['ADMISSION_DATE']

# %%
train_df_hour = train_df_hour.reset_index()

# %%
for idx in tqdm(range(len(train_df_hour))):
    info = train_df_hour.loc[idx]
    admission = datetime.datetime.strptime(info['ADMISSION_DATE'], '%Y-%m-%d %H:%M:%S')
    departure = datetime.datetime.strptime(info['DEPARTURE_DATE'], '%Y-%m-%d %H:%M:%S')
    visit_hour = datetime.datetime.strptime(info['RECORD_TIME_HOUR'], '%Y-%m-%d %H')
    hour = (departure - visit_hour).seconds / (24 * 60 * 60) + (departure - visit_hour).days
    los = (departure - admission).seconds / (24 * 60 * 60) + (departure - admission).days
    train_df_hour.at[idx, 'LOS'] = float(los)
    train_df_hour.at[idx, 'LOS_HOUR'] = float(hour)

# %%
train_df_hour['LOS']

# %%
los = []
for pat in tqdm(train_df_hour['PATIENT_ID'].unique()):
    los.append(float(train_df_hour[train_df_hour['PATIENT_ID'] == pat]['LOS'].head(1)))

# %%
get_statistic(los, 'los')

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
"""
csfont = {'fontname':'Times New Roman', 'fontsize': 18}
font = 'Times New Roman'
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
plt.style.use('seaborn-whitegrid')
color = 'cornflowerblue'
ec = 'None'
alpha=0.5

ax = plt.subplot(1, 1, 1)
ax.hist(los, bins=20, weights=np.ones(len(los)) / len(los), color=color, ec=ec, alpha=alpha, label='overall')
plt.xlabel('Length of stay',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xticks(**csfont)
plt.yticks(**csfont)

plt.show()
"""
# %%
train_df_hour_idx = train_df_hour.reset_index()

# %%
train_df_hour_idx['LOS'] = train_df_hour_idx['ADMISSION_DATE']

for idx in tqdm(range(len(train_df_hour_idx))):
    info = train_df_hour_idx.loc[idx]
    # admission = datetime.datetime.strptime(info['ADMISSION_DATE'], '%Y-%m-%d %H:%M:%S')
    departure = datetime.datetime.strptime(info['DEPARTURE_DATE'], '%Y-%m-%d %H:%M:%S')
    visit_hour = datetime.datetime.strptime(info['RECORD_TIME_HOUR'], '%Y-%m-%d %H')
    hour = (departure - visit_hour).seconds / (24 * 60 * 60) + (departure - visit_hour).days
    train_df_hour_idx.at[idx, 'LOS'] = float(hour)

# %%
train_df_hour['LOS'] = train_df_hour['LOS_HOUR']
train_df_hour.drop(columns=['LOS_HOUR'])

# %%
# los_threshold = 13.0

# visit_num_hour = []

# for pat in tqdm(train_df_hour_idx['PATIENT_ID'].unique()):
#     pat_records = train_df_hour_idx[train_df_hour_idx['PATIENT_ID'] == pat]
#     hour = 0
#     for vis in pat_records.index:
#         pat_visit = pat_records.loc[vis]
#         if pat_visit['LOS_HOUR'] <= los_threshold:
#             hour += 1
#     visit_num_hour.append(hour)
#     if hour == 0:
#         print(pat)

# %%
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# import matplotlib.font_manager as font_manager
# import pandas as pd
# import numpy as np
# csfont = {'fontname':'Times New Roman', 'fontsize': 18}
# font = 'Times New Roman'
# fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
# plt.style.use('seaborn-whitegrid')
# color = 'cornflowerblue'
# ec = 'None'
# alpha=0.5

# ax = plt.subplot(1, 1, 1)
# ax.hist(visit_num_hour, bins=20, weights=np.ones(len(visit_num_hour)) / len(visit_num_hour), color=color, ec=ec, alpha=alpha, label='overall')
# plt.xlabel('Visit num (80% los)',**csfont)
# plt.ylabel('Percentage',**csfont)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.xticks(**csfont)
# plt.yticks(**csfont)

# plt.show()

# %%
train_df = train_df_hour
train_df.head()

# %%
train_df.describe()

# %%
get_statistic(train_df['LOS'], 'los')

# %%
train_df[train_df['PATIENT_ID'] == '1']['HEART_RATE'].count()

# %%
cols = train_df.columns[5:]
pats = train_df['PATIENT_ID'].unique()
all_pat_cnt = len(pats)
missing_rate = dict()
# for col in tqdm(cols):
#     miss = 0
#     for pat in pats:
#         if train_df[train_df['PATIENT_ID'] == pat][col].count() == 0:
#             miss += 1
#     missing_rate[col] = miss / all_pat_cnt
    
for col in cols:
    missing_rate[col] = 0
for pat in tqdm(pats):
    p = train_df[train_df['PATIENT_ID'] == pat]
    for col in cols:
        if p[col].count() == 0:
            missing_rate[col] += 1
for col in cols:
    missing_rate[col] = missing_rate[col] / all_pat_cnt
    
missing_rate

# %%
with open('missing_rate.csv', mode='w', encoding='utf-8') as file:
    for col in cols:
        file.write(f'"{col}", {100 * missing_rate[col]}\n')

# %%
train_df['LOS'] = train_df['LOS'].clip(lower=0)

# %%
get_statistic(train_df['LOS'], 'los')

# %%
# the first visit of each person
def init_prev(prev):
    miss = []
    l = len(prev)
    for idx in range(l):
        #print(prev[idx])
        #print(type(prev[idx]))
        if np.isnan(prev[idx]): # there is no previous record
            prev[idx] = test_median[idx] # replace nan to median
            miss.append(1) # mark miss as 1
        else: # there is a previous record
            miss.append(0)
    return miss

# the rest of the visits
def fill_nan(cur, prev):
    l = len(prev)
    miss = []
    for idx in range(l):
        #print(cur[idx])
        if np.isnan(cur[idx]): # there is no record in current timestep
            cur[idx] = prev[idx] # cur <- prev
            miss.append(1)
        else: # there is a record in current timestep
            miss.append(0)
    return miss

# %%
x, y, demo, x_lab_len, missing_mask = [], [], [], [], []

for pat in tqdm(patient_ids): # for all patients
    # get visits for pat.id == PATIENT_ID
    info = train_df[train_df['PATIENT_ID'] == pat]
    info = info[max(0, len(info) - 76):]
    indexes = info.index
    visit = info.loc[indexes[0]] # get the first visit

    # demographic data
    demo.append([visit[k] for k in demo_cols])
    
    # label
    outcome = visit['OUTCOME']
    los = []

    # lab test & vital signs
    tests = []
    prev = visit[test_cols]
    miss = [] # missing matrix
    miss.append(init_prev(prev)) # fill nan for the first visit for every patient and add missing status to missing matrix
    # leave = datetime.datetime.strptime(visit['DEPARTURE_DATE'], '%Y-%m-%d %H:%M:%S')
    
    first = True
    for i in indexes:
        visit = info.loc[i]
        # now = datetime.datetime.strptime(visit['RECORD_TIME'], '%Y-%m-%d %H')
        cur = visit[test_cols]
        tmp = fill_nan(cur, prev) # fill nan for the rest of the visits
        if not first:
            miss.append(tmp) # add missing status to missing matrix
        tests.append(cur)
        # los_visit = (leave - now).days
        # if los_visit < 0:
        #     los_visit = 0
        los.append(visit['LOS'])
        prev = cur
        first = False

    valid_visit = len(los)
    # outcome = [outcome] * valid_visit
    x_lab_len.append(valid_visit)
    missing_mask.append(miss) # append the patient's missing matrix to the total missing matrix

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
all_x_demographic = torch.reshape(all_x_demographic.repeat(1, max_length), (batch_size, max_length, demo_dim))
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
pd.to_pickle(all_x,f'./processed_data/x.pkl' )
pd.to_pickle(all_missing_mask,f'./processed_data/missing_mask.pkl' )
pd.to_pickle(output_all_y,f'./processed_data/y.pkl' )
pd.to_pickle(x_lab_length,f'./processed_data/visits_length.pkl' )

# %%
# Calculate patients' outcome statistics (patients-wise)
outcome_list = []
y_outcome = output_all_y[:, :, 0]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    outcome_list.append(y_outcome[i][0].item())
outcome_list = np.array(outcome_list)
print(len(outcome_list))
unique, count=np.unique(outcome_list,return_counts=True)
data_count=dict(zip(unique,count))
print(data_count)

# %%
# Calculate patients' outcome statistics (records-wise)
outcome_records_list = []
y_outcome = output_all_y[:, :, 0]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    outcome_records_list.extend(y_outcome[i][0:x_lab_length[i]].tolist())
outcome_records_list = np.array(outcome_records_list)
print(len(outcome_records_list))
unique, count=np.unique(outcome_records_list,return_counts=True)
data_count=dict(zip(unique,count))
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

print('median:', np.median(los_list))
print('Q1:', np.percentile(los_list, 25))
print('Q3:', np.percentile(los_list, 75))

# %%
los_alive_list = np.array([los_list[i] for i in range(len(los_list)) if outcome_list[i] == 0])
los_dead_list = np.array([los_list[i] for i in range(len(los_list)) if outcome_list[i] == 1])
print(len(los_alive_list))
print(len(los_dead_list))

print('[Alive]')
print('median:', np.median(los_alive_list))
print('Q1:', np.percentile(los_alive_list, 25))
print('Q3:', np.percentile(los_alive_list, 75))

print('[Dead]')
print('median:', np.median(los_dead_list))
print('Q1:', np.percentile(los_dead_list, 25))
print('Q3:', np.percentile(los_dead_list, 75))

# %%
cdsl_los_statistics = {
    'overall': los_list,
    'alive': los_alive_list,
    'dead': los_dead_list
}
pd.to_pickle(cdsl_los_statistics, 'cdsl_los_statistics.pkl')

# %%
# calculate visits length Median [Q1, Q3]
visits_list = np.array(x_lab_length)
visits_alive_list = np.array([x_lab_length[i] for i in range(len(x_lab_length)) if outcome_list[i] == 0])
visits_dead_list = np.array([x_lab_length[i] for i in range(len(x_lab_length)) if outcome_list[i] == 1])
print(len(visits_alive_list))
print(len(visits_dead_list))

print('[Total]')
print('median:', np.median(visits_list))
print('Q1:', np.percentile(visits_list, 25))
print('Q3:', np.percentile(visits_list, 75))

print('[Alive]')
print('median:', np.median(visits_alive_list))
print('Q1:', np.percentile(visits_alive_list, 25))
print('Q3:', np.percentile(visits_alive_list, 75))

print('[Dead]')
print('median:', np.median(visits_dead_list))
print('Q1:', np.percentile(visits_dead_list, 25))
print('Q3:', np.percentile(visits_dead_list, 75))

# %%
def check_nan(x):
    if np.isnan(np.sum(x.cpu().numpy())):
        print("some values from input are nan")
    else:
        print("no nan")

# %%
check_nan(all_x)

# %% [markdown]
# # Draw Charts

# %% [markdown]
# ## Import packages

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np

plt.style.use('seaborn-whitegrid')
color = 'cornflowerblue'
ec = 'None'
alpha=0.5
alive_color = 'olivedrab'
dead_color = 'orchid'

# %% [markdown]
# ## Read data

# %%
demographic.head()

# %%
train = pd.read_csv('./train.csv')
train['PATIENT_ID']=train['PATIENT_ID'].astype(str)
demographic['PATIENT_ID']=demographic['PATIENT_ID'].astype(str)
pat = {
    'PATIENT_ID': train['PATIENT_ID'].unique()
}
pat = pd.DataFrame(pat)
demo = pd.merge(demographic, pat, on='PATIENT_ID', how='inner')

demo_alive = demo.loc[demo['OUTCOME'] == 0]
demo_dead = demo.loc[demo['OUTCOME'] == 1]
demo_overall = demo

# %%
demo.to_csv('demo_overall.csv', index=False)
demo_alive.to_csv('demo_alive.csv', index=False)
demo_dead.to_csv('demo_dead.csv', index=False)

# %%
patient = pd.DataFrame({"PATIENT_ID": (demo_alive['PATIENT_ID'].unique())})
lab_tests_alive = pd.merge(lab_tests, patient, how='inner', on='PATIENT_ID')
print(len(lab_tests_alive['PATIENT_ID'].unique()))

patient = pd.DataFrame({"PATIENT_ID": (demo_dead['PATIENT_ID'].unique())})
lab_tests_dead = pd.merge(lab_tests, patient, how='inner', on='PATIENT_ID')
print(len(lab_tests_dead['PATIENT_ID'].unique()))

patient = pd.DataFrame({"PATIENT_ID": (demo_overall['PATIENT_ID'].unique())})
lab_tests_overall = pd.merge(lab_tests, patient, how='inner', on='PATIENT_ID')
print(len(lab_tests_overall['PATIENT_ID'].unique()))

# %%
patient = pd.DataFrame({"PATIENT_ID": (demo_alive['PATIENT_ID'].unique())})
vital_signs_alive = pd.merge(vital_signs, patient, how='inner', on='PATIENT_ID')
len(vital_signs_alive['PATIENT_ID'].unique())

# %%
patient = pd.DataFrame({"PATIENT_ID": (demo_dead['PATIENT_ID'].unique())})
vital_signs_dead = pd.merge(vital_signs, patient, how='inner', on='PATIENT_ID')
len(vital_signs_dead['PATIENT_ID'].unique())

# %%
patient = pd.DataFrame({"PATIENT_ID": (demo_overall['PATIENT_ID'].unique())})
vital_signs_overall = pd.merge(vital_signs, patient, how='inner', on='PATIENT_ID')
len(vital_signs_overall['PATIENT_ID'].unique())

# %%
"""
limit = 0.05

csfont = {'fontname':'Times New Roman', 'fontsize': 18}
font = 'Times New Roman'
fig=plt.figure(figsize=(16,12), dpi= 100, facecolor='w', edgecolor='k')

idx = 1

key = 'AGE'
low = demo_overall[key].quantile(limit)
high = demo_overall[key].quantile(1 - limit)
demo_AGE_overall = demo_overall[demo_overall[key].between(low, high)]
demo_AGE_dead = demo_dead[demo_dead[key].between(low, high)]
demo_AGE_alive = demo_alive[demo_alive[key].between(low, high)]
ax = plt.subplot(4, 4, idx)
ax.hist(demo_AGE_overall[key], bins=20, weights=np.ones(len(demo_AGE_overall[key])) / len(demo_AGE_overall), color=color, ec=ec, alpha=alpha, label='overall')
plt.xlabel('Age',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# ax.title('Age Histogram', **csfont)
ax.hist(demo_AGE_alive[key], bins=20, weights=np.ones(len(demo_AGE_alive[key])) / len(demo_AGE_alive), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2, label='alive')
ax.hist(demo_AGE_dead[key], bins=20, weights=np.ones(len(demo_AGE_dead[key])) / len(demo_AGE_dead), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2, label='dead')
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'TEMPERATURE'
low = vital_signs_overall[key].quantile(limit)
high = vital_signs_overall[key].quantile(1 - limit)
vs_TEMPERATURE_overall = vital_signs_overall[vital_signs_overall[key].between(low, high)]
vs_TEMPERATURE_dead = vital_signs_dead[vital_signs_dead[key].between(low, high)]
vs_TEMPERATURE_alive = vital_signs_alive[vital_signs_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(vs_TEMPERATURE_overall['TEMPERATURE'], bins=20, weights=np.ones(len(vs_TEMPERATURE_overall)) / len(vs_TEMPERATURE_overall), color=color, ec=ec, alpha=alpha)
plt.xlabel('Temperature',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(vs_TEMPERATURE_alive['TEMPERATURE'], bins=20, weights=np.ones(len(vs_TEMPERATURE_alive)) / len(vs_TEMPERATURE_alive), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(vs_TEMPERATURE_dead['TEMPERATURE'], bins=20, weights=np.ones(len(vs_TEMPERATURE_dead)) / len(vs_TEMPERATURE_dead), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

# plt.subplot(4, 4, 3)
# plt.hist(lab_tests_overall['CREA -- CREATININA'], bins=20, density=True, color=color, ec=ec, alpha=alpha)
# plt.xlabel('CREA -- CREATININA',**csfont)
# plt.ylabel('Percentage',**csfont)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# # plt.title('Temperature Histogram', **csfont)
# plt.hist(lab_tests_alive['CREA -- CREATININA'], bins=20, density=True, color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
# plt.hist(lab_tests_dead['CREA -- CREATININA'], bins=20, density=True, color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
# plt.xticks(**csfont)
# plt.yticks(**csfont)

key = 'CREA -- CREATININA'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('CREA -- CREATININA',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'HEM -- Hemat¡es'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('HEM -- Hemat¡es',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'LEUC -- Leucocitos'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('LEUC -- Leucocitos',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'PLAQ -- Recuento de plaquetas'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('PLAQ -- Recuento de plaquetas',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'CHCM -- Conc. Hemoglobina Corpuscular Media'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('CHCM',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'HCTO -- Hematocrito'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('HCTO -- Hematocrito',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'VCM -- Volumen Corpuscular Medio'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('VCM -- Volumen Corpuscular Medio',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'HGB -- Hemoglobina'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('HGB -- Hemoglobina',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'HCM -- Hemoglobina Corpuscular Media'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('HCM -- Hemoglobina Corpuscular Media',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'NEU -- Neutr¢filos'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('NEU -- Neutr¢filos',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'NEU% -- Neutr¢filos %'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('NEU% -- Neutr¢filos%',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'LIN -- Linfocitos'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('LIN -- Linfocitos',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'LIN% -- Linfocitos %'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('LIN% -- Linfocitos%',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

key = 'ADW -- Coeficiente de anisocitosis'
low = lab_tests_overall[key].quantile(limit)
high = lab_tests_overall[key].quantile(1 - limit)
lt_key_overall = lab_tests_overall[lab_tests_overall[key].between(low, high)]
lt_key_dead = lab_tests_dead[lab_tests_dead[key].between(low, high)]
lt_key_alive = lab_tests_alive[lab_tests_alive[key].between(low, high)]
plt.subplot(4, 4, idx)
plt.hist(lt_key_overall[key], bins=20, weights=np.ones(len(lt_key_overall[key])) / len(lt_key_overall[key]), color=color, ec=ec, alpha=alpha)
plt.xlabel('ADW -- Coeficiente de anisocitosis',**csfont)
plt.ylabel('Percentage',**csfont)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Temperature Histogram', **csfont)
plt.hist(lt_key_alive[key], bins=20, weights=np.ones(len(lt_key_alive[key])) / len(lt_key_alive[key]), color='green', ec=alive_color, alpha=1, histtype="step", linewidth=2)
plt.hist(lt_key_dead[key], bins=20, weights=np.ones(len(lt_key_dead[key])) / len(lt_key_dead[key]), color='green', ec=dead_color, alpha=1, histtype="step", linewidth=2)
plt.xticks(**csfont)
plt.yticks(**csfont)
idx += 1

handles, labels = ax.get_legend_handles_labels()
print(handles, labels)
# fig.legend(handles, labels, loc='upper center')
plt.figlegend(handles, labels, loc='upper center', ncol=5, fontsize=18, bbox_to_anchor=(0.5, 1.05), prop=font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=18))
# fig.legend(, [], loc='upper center')

fig.tight_layout()
plt.show()
"""

