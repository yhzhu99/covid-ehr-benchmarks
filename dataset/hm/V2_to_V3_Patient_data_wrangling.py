import pandas as pd
import datetime as dt

from pathlib import Path

######################################### WRANGLING PATIENT TABLE DATA #########################################
# V3_data_folder = Path("C:/Users/Jesus/Desktop/Master/TFM/Datos/19_04_2021/")
V2_data_folder = Path("C:/Users/Jesus/Desktop/Master/TFM/Datos/20_07_2020/")

# V3_patient = V3_data_folder / "COVID_DSL_01.csv"
V2_patient = V2_data_folder / "CDSL_01.csv"

# ###### V3 COVID_DSL_01
# # Load csv data into dataframe
# df_V3 = pd.read_csv(V3_patient, delimiter="|", encoding="ANSI")

# # Display dataframe head 
# #print(df_V3.head()) 

# # Get columns name
# df_V3_columns = df_V3.columns.tolist()

###### V2 CDSL_01
# Load csv data into dataframe
df_V2 = pd.read_csv("dataset/hm/raw_data/01_基线.csv", delimiter=";", encoding='unicode_escape', decimal = ",")

# Display dataframe head 
#print(df_V2.head()) 

# Get columns name
df_V2_columns = df_V2.columns.tolist()

###### V2 CDSL_01.csv Data wrangling
### Dictionary containing V2 CDSL_01 <--> V3 COVID_DSL_01 column names equivalence (Ordered as in V3 COVID_DSL_01)
col_names_eq_dict = {'EDAD/AGE':'EDAD',
                     'SEXO/SEX':'SEX',
                     'PATIENT ID':'IDINGRESO',
                     'F_INGRESO/ADMISSION_D_ING/INPAT':'F_INGRESO_ING',
                     'F_ALTA/DISCHARGE_DATE_ING':'F_ALTA_ING',
                     'MOTIVO_ALTA/DESTINY_DISCHARGE_ING':'MOTIVO_ALTA_ING',
                     'DIAG ING/INPAT':'DIAGNOSTICO_ING',
                     'F_INGRESO/ADMISSION_DATE_URG/EMERG':'F_INGRESO_URG',
                     'HORA/TIME_ADMISION/ADMISSION_URG/EMERG':'HORA_URG',
                     'DIAG_URG/EMERG':'DIAG_URG',
                     'ESPECIALIDAD/DEPARTMENT_URG/EMERG':'ESPECIALIDAD_URGENCIA',
                     'HORA/TIME_CONSTANT_PRIMERA/FIRST_URG/EMERG':'HORA_CONSTANTES_PRIMERA_URG',
                     'TA_MAX_PRIMERA/FIRST/EMERG_URG':'TA_MAX_PRIMERA_URG',
                     'TA_MIN_PRIMERA/FIRST_URG/EMERG':'TA_MIN_PRIMERA_URG',
                     'TEMP_PRIMERA/FIRST_URG/EMERG':'TEMP_PRIMERA_URG',
                     'FC/HR_PRIMERA/FIRST_URG/EMERG':'FC_PRIMERA_URG',
                     'SAT_02_PRIMERA/FIRST_URG/EMERG':'SAT_02_PRIMERA_URG',
                     'GLU_PRIMERA/FIRST_URG/EMERG':'GLU_PRIMERA_URG',
                     'DIURESIS_PRIMERA_URG':'DIURESIS_PRIMERA_URG', #Column not existing in V2
                     'HORA/TIME_CONSTANT_ULTIMA/LAST_URG/EMERG':'HORA_CONSTANTES_ULTIMA_URG',
                     'TA_MAX_ULTIMA/LAST_URGEMERG':'TA_MAX_ULTIMA_URG',
                     'TA_MIN_ULTIMA/LAST_URG/EMERG':'TA_MIN_ULTIMA_URG',
                     'TEMP_ULTIMA/LAST_URG/EMERG':'TEMP_ULTIMA_URG',
                     'FC/HR_ULTIMA/LAST_URG/EMERG':'FC_ULTIMA_URG',
                     'SAT_02_ULTIMA/LAST_URG/EMERG':'SAT_02_ULTIMA_URG',
                     'GLU_ULTIMA/LAST_URG/EMERG':'GLU_ULTIMA_URG',
                     'DESTINO/DESTINY_URG/EMERG':'DESTINO_URG',
                     'IDCDSL':'IDCDSL', #Column not existing in V2
                     'F_ING_ANT':'F_ING_ANT', #Column not existing in V2
                     'DIAG_ANT':'DIAG_ANT', #Column not existing in V2
                     'RESPIRADOR/MECH.VENT.':'RESPIRADOR',
                     'F_ENTRADA_UC/ICU_DATE_IN':'F_UCI_IN',
                     'F_SALIDA_UCI/ICU_DATE_OUT':'F_UCI_OUT',
                     'UCI_DIAS/ICU_DAYS':'UCI_DAYS',
                     'UCI_N_ING':'UCI_N_ING'} #Column not existing in V2

### V2 CDSL_01 columns reordering according to V3 columns sequence
df_V2_columns_wrangled = []
for dict_keys, dict_values in col_names_eq_dict.items():
  df_V2_columns_wrangled.append(dict_keys) # V2 CDSL_01 column names dictionary re-ordered

df_V2 = df_V2.reindex(columns=df_V2_columns_wrangled) # V2 CDSL_01 dataframe columns re-ordered 

### Rename V2 CDSL_01.csv columns acc. to V3 column names
df_V2.rename(columns = col_names_eq_dict, inplace = True)

### Format adaptation CDSL_01 acc.to V3 COVID_DSL_01.csv format

# Date/Time formatting
# col_aux = ['F_INGRESO_ING', 'F_ALTA_ING', 'F_UCI_IN', 'F_UCI_OUT']
# for col in col_aux:
#   df_V2[col] = pd.to_datetime(df_V2[col]).dt.strftime('%d-%m-%Y %H:%M:%S')

col_aux = ['HORA_URG', 'HORA_CONSTANTES_PRIMERA_URG', 'HORA_CONSTANTES_ULTIMA_URG']
for col in col_aux:
  df_V2[col] = pd.to_datetime(df_V2[col]).dt.strftime('%H:%M:%S')

df_V2['F_ING_ANT'] = pd.to_datetime(df_V2['F_ING_ANT']).dt.strftime('%d-%m-%Y')

# Change NaN/NaT by "empty"
col_aux = ['RESPIRADOR', 'UCI_DAYS', 'UCI_N_ING', 'DIAG_ANT']
for col in col_aux:
  df_V2[col] = df_V2[col].fillna("")

col_aux = ['F_ALTA_ING', 'HORA_URG', 'F_ING_ANT', 'F_UCI_IN', 'F_UCI_OUT', 'HORA_CONSTANTES_PRIMERA_URG', 'HORA_CONSTANTES_ULTIMA_URG']
for col in col_aux:
  df_V2[col] = df_V2[col].replace('NaT', '')

# Change "Non-numeric" to "numeric" (int)
df_V2["UCI_DAYS"] = pd.to_numeric(df_V2['UCI_DAYS'])
df_V2["UCI_DAYS"] = df_V2["UCI_DAYS"].fillna(0).astype(int)

## Dataframe to CSV
df_V2.to_csv("dataset/hm/raw_data/hm_demo.csv", index = False, encoding="utf-8", sep = ",")