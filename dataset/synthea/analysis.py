import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime

def outcome_table(inpatient_ids, survivor_ids, deceased_patients, conditions):
  outcomes = {'Sepsis': 770349000, 'Respiratory Failure': 65710008, 'ARDS': 67782005, 'Heart Failure': 84114007,
            'Septic Shock': 76571007, 'Coagulopathy': 234466008, 'Acute Cardiac Injury': 86175003,
            'Acute Kidney Injury': 40095003}

  outcome_counts = []
  survivor_count = np.intersect1d(inpatient_ids, survivor_ids).size
  non_survivor_count = np.intersect1d(inpatient_ids, deceased_patients).size
  total_complete =  survivor_count + non_survivor_count

  for outcome, outcome_code in outcomes.items():
    patients_with_outcome = conditions[conditions.CODE == outcome_code].PATIENT.unique()
    inpatient_with_outcome = np.intersect1d(inpatient_ids, patients_with_outcome)
    survivors_with_outcome = np.intersect1d(inpatient_with_outcome, survivor_ids).size
    non_survivors_with_outcome = np.intersect1d(inpatient_with_outcome, deceased_patients).size
    total_complete_outcome = survivors_with_outcome + non_survivors_with_outcome
    row = {'outcome': outcome, 'total': total_complete_outcome,
           'percent of inpatient': (total_complete_outcome / total_complete),
           'survivors': survivors_with_outcome, 'percent survivors': (survivors_with_outcome / survivor_count),
           'non survivors': non_survivors_with_outcome,
           'percent non survivors': (non_survivors_with_outcome / non_survivor_count)}
    outcome_counts.append(row)

  return pd.DataFrame.from_records(outcome_counts)

def select_condition_averages(covid_patient_conditions, filters):
  select_covid_patient_conditions = covid_patient_conditions[((covid_patient_conditions.CODE == 267036007) |
                                                           (covid_patient_conditions.CODE == 386661006) |
                                                           (covid_patient_conditions.CODE == 49727002)) &
                                                           filters
                                                          ]
  return select_covid_patient_conditions.groupby('CODE').mean()

def create_covid_icu(covid_info, encounters):
  covid_icu = covid_info.merge(encounters[encounters.CODE == 305351004], on='PATIENT')
  covid_icu['start_days'] = (pd.to_datetime(covid_icu.START) - pd.to_datetime(covid_icu.covid_start, utc=True)) / np.timedelta64(1, 'D')
  covid_icu['end_days'] = (pd.to_datetime(covid_icu.STOP) - pd.to_datetime(covid_icu.covid_start, utc=True)) / np.timedelta64(1, 'D')
  return covid_icu

def create_covid_vent(covid_info, devices):
  covid_vent = covid_info.merge(devices[devices.CODE == 449071006], on='PATIENT')
  covid_vent['start_days'] = (pd.to_datetime(covid_vent.START, utc=True) - pd.to_datetime(covid_vent.covid_start, utc=True)) / np.timedelta64(1, 'D')
  covid_vent['end_days'] = (pd.to_datetime(covid_vent.STOP, utc=True) - pd.to_datetime(covid_vent.covid_start, utc=True)) / np.timedelta64(1, 'D')
  return covid_vent

def create_covid_hosp(covid_info, encounters, filters):
  covid_hosp = covid_info.merge(encounters[(encounters.REASONCODE == 840539006) & (encounters.CODE != 308646001)], on='PATIENT')
  covid_hosp['STOP'] = pd.to_datetime(covid_hosp.STOP, utc=True)
  covid_hosp['START'] = pd.to_datetime(covid_hosp.START, utc=True)
  covid_hosp['DEATHDATE'] = pd.to_datetime(covid_hosp.DEATHDATE, utc=True)
  covid_hosp['covid_start'] = pd.to_datetime(covid_hosp.covid_start, utc=True)
  df_filters = map(lambda pair: (covid_hosp[pair[0]] == pair[1]), filters.items())
  for f in df_filters:
    covid_hosp = covid_hosp[f]
  return covid_hosp.groupby('PATIENT').agg({'STOP': 'max', 'covid_start': 'min', 'DEATHDATE': 'max', 'START': 'min'})

def survivor_timeline_plot(encounters, devices, averages, covid_patient_conditions, covid_info, icu_admit_required=False):
  covid_icu = create_covid_icu(covid_info, encounters)
  covid_vent = create_covid_vent(covid_info, devices)
  filter = {'recovered': True}
  if icu_admit_required:
    filter['icu_admit'] = True
  covid_hosp = create_covid_hosp(covid_info, encounters, filter)
  plot_conditions = {'Fever': 386661006, 'Cough': 49727002, 'Dyspnoea': 267036007}
  plot_order = ['Dyspnoea', 'Cough', 'Fever']
  colors = ['red', 'blue', 'green']

  covid_icu_survivors = covid_icu[covid_icu.recovered == True]
  covid_vent_survivors = covid_vent[covid_vent.recovered == True]

  fig, ax = plt.subplots()
  y_position = 50
  for cond, color in zip(reversed(plot_order), colors):
      bar_start = averages.to_dict()['start_days'][plot_conditions[cond]]
      bar_end = averages.to_dict()['end_days'][plot_conditions[cond]] - bar_start
      ax.broken_barh([(bar_start, bar_end)], (y_position, 9), facecolors='tab:{}'.format(color))
      y_position -= 10
  ax.broken_barh([(covid_icu_survivors.start_days.mean(),
                  (covid_icu_survivors.end_days.mean() - covid_icu_survivors.start_days.mean()))],
                  (y_position, 9), facecolors='tab:purple')
  y_position -= 10
  ax.broken_barh([(covid_vent_survivors.start_days.mean(),
                  (covid_vent_survivors.end_days.mean() - covid_vent_survivors.start_days.mean()))],
                  (y_position, 9), facecolors='tab:orange')

  ax.set_ylim(5, 65)
  ax.set_xlim(0, 25)
  ax.set_xlabel('Days')
  ax.set_yticks([15, 25, 35, 45, 55])
  labels = ['Invasive Ventilation', 'ICU Admission']
  labels.extend(plot_order)
  ax.set_yticklabels(labels)
  ax.grid(True)
  ax.set_title('Survivors')

  annotation_outcomes = {'Sepsis': 770349000, 'ARDS': 67782005}

  label_position = -10
  for condition_name, condition_code in annotation_outcomes.items():
    filter = (covid_patient_conditions.CODE == condition_code) & (covid_patient_conditions.recovered == True)
    if icu_admit_required:
      filter = filter & (covid_patient_conditions.icu_admit == True)
    condition_df = covid_patient_conditions[filter]
    avg_start = ((pd.to_datetime(condition_df.START) - pd.to_datetime(condition_df.covid_start)) / np.timedelta64(1, 'D')).mean()
    ax.annotate(condition_name, (avg_start, 5), xycoords='data', xytext=(avg_start, label_position), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05))
    label_position -= 5

  dischange_avg = ((covid_hosp.STOP - covid_hosp.covid_start) / np.timedelta64(1, 'D')).mean()

  ax.annotate('Discharge', (dischange_avg, 5), xycoords='data', xytext=(dischange_avg, -10), textcoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05))

  plt.show()

def non_survivor_timeline_plot(encounters, devices, averages, covid_patient_conditions, covid_info, icu_admit_required=False):
  plot_conditions = {'Fever': 386661006, 'Cough': 49727002, 'Dyspnoea': 267036007}
  plot_order = ['Dyspnoea', 'Cough', 'Fever']
  colors = ['red', 'blue', 'green']
  covid_icu = create_covid_icu(covid_info, encounters)
  covid_vent = create_covid_vent(covid_info, devices)
  covid_icu_non_survivors = covid_icu[covid_icu.death == True]
  covid_vent_non_survivors = covid_vent[covid_vent.death == True]
  filter = {'death': True}
  if icu_admit_required:
    filter['icu_admit'] = True
  covid_hosp = create_covid_hosp(covid_info, encounters, filter)

  fig, ax = plt.subplots()
  y_position = 50
  for cond, color in zip(reversed(plot_order), colors):
      bar_start = averages.to_dict()['start_days'][plot_conditions[cond]]
      bar_end = averages.to_dict()['death_days'][plot_conditions[cond]] - bar_start
      ax.broken_barh([(bar_start, bar_end)], (y_position, 9), facecolors='tab:{}'.format(color))
      y_position -= 10
  ax.broken_barh([(covid_icu_non_survivors.start_days.mean(),
                  (covid_icu_non_survivors.end_days.mean() - covid_icu_non_survivors.start_days.mean()))],
                  (y_position, 9), facecolors='tab:purple')
  y_position -= 10
  ax.broken_barh([(covid_vent_non_survivors.start_days.mean(),
                  (covid_vent_non_survivors.end_days.mean() - covid_vent_non_survivors.start_days.mean()))],
                  (y_position, 9), facecolors='tab:orange')

  ax.set_ylim(5, 65)
  ax.set_xlim(0, 15)
  ax.set_xlabel('Days')
  ax.set_yticks([15, 25, 35, 45, 55])
  labels = ['Invasive Ventilation', 'ICU Admission']
  labels.extend(plot_order)
  ax.set_yticklabels(labels)
  ax.grid(True)
  ax.set_title('Non Survivors')

  annotation_outcomes = {'Sepsis': 770349000, 'ARDS': 67782005, 'Acute Cardiac Injury': 86175003,
              'Acute Kidney Injury': 40095003, 'Septic Shock': 76571007}

  label_position = -10
  for condition_name, condition_code in annotation_outcomes.items():
      condition_df = covid_patient_conditions[(covid_patient_conditions.CODE == condition_code) &
                          (covid_patient_conditions.death == True)]
      avg_start = ((pd.to_datetime(condition_df.START) - pd.to_datetime(condition_df.covid_start)) / np.timedelta64(1, 'D')).mean()
      ax.annotate(condition_name, (avg_start, 5), xycoords='data', xytext=(avg_start, label_position), textcoords='data',
              arrowprops=dict(facecolor='black', shrink=0.05))
      label_position -= 5

  death_avg = ((covid_hosp.DEATHDATE - covid_hosp.covid_start) / np.timedelta64(1, 'D')).mean()

  ax.annotate('Death', (death_avg, 5), xycoords='data', xytext=(death_avg, -10), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05))
  plt.show()

def symptom_table(covid_patient_conditions, icu_only = False):
  symptom_map = {'Conjunctival Congestion': 246677007, 'Nasal Congestion': 68235000, 'Headache': 25064002,
               'Cough': 49727002, 'Sore Throat': 267102003, 'Sputum Production': 248595008,
               'Fatigue': 84229001, 'Hemoptysis': 66857006, 'Shortness of Breath': 267036007,
               'Nausea': 422587007, 'Diarrhea': 267060006, 'Muscle Pain': 68962001,
               'Joint Pain': 57676002, 'Chills': 43724002, 'Loss of Taste': 36955009}

  if icu_only:
    covid_patient_conditions = covid_patient_conditions[covid_patient_conditions.icu_admit == True]

  covid_count = covid_patient_conditions.PATIENT.unique().size

  table_rows = []
  for symptom, condition_code in symptom_map.items():
    row = {'Symptoms': symptom}
    row['All Patients Percentage'] = covid_patient_conditions[covid_patient_conditions.CODE == condition_code].PATIENT.size / covid_count
    row['All Patients Count'] = covid_patient_conditions[covid_patient_conditions.CODE == condition_code].PATIENT.size
    row['Survivor Percentage'] = covid_patient_conditions[(covid_patient_conditions.CODE == condition_code) & (covid_patient_conditions.recovered == True)].PATIENT.size / covid_patient_conditions[covid_patient_conditions.recovered == True].PATIENT.unique().size
    row['Survivor Count'] = covid_patient_conditions[(covid_patient_conditions.CODE == condition_code) & (covid_patient_conditions.recovered == True)].PATIENT.size
    row['Non Survivor Percentage'] = covid_patient_conditions[(covid_patient_conditions.CODE == condition_code) & (covid_patient_conditions.death == True)].PATIENT.size / covid_patient_conditions[covid_patient_conditions.death == True].PATIENT.unique().size
    row['Non Survivor Count'] = covid_patient_conditions[(covid_patient_conditions.CODE == condition_code) & (covid_patient_conditions.death == True)].PATIENT.size

    table_rows.append(row)

  return pd.DataFrame.from_records(table_rows)