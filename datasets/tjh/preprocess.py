# %%
# Import necessary packages
import numpy as np
import pandas as pd
import torch

# %%
# Read raw data
df_train: pd.DataFrame = pd.read_excel(
    "./datasets/tongji/raw_data/time_series_375_prerpocess_en.xlsx"
)

# %% [markdown]
# Steps:
#
# - fill `patient_id`
# - only reserve y-m-d for `RE_DATE` column
# - merge lab tests of the same (patient_id, date)
# - calculate and save features' statistics information (demographic and lab test data are calculated separately)
# - normalize data
# - feature selection
# - fill missing data (our filling strategy will be described below)
# - combine above data to time series data (one patient one record)
# - export to python pickle file

# %%
# fill `patient_id` rows
df_train["PATIENT_ID"].fillna(method="ffill", inplace=True)

# gender transformation: 1--male, 0--female
df_train["gender"].replace(2, 0, inplace=True)

# only reserve y-m-d for `RE_DATE` and `Discharge time` columns
df_train["RE_DATE"] = df_train["RE_DATE"].dt.strftime("%Y-%m-%d")
df_train["Discharge time"] = df_train["Discharge time"].dt.strftime("%Y-%m-%d")


# %%
df_train = df_train.dropna(
    subset=["PATIENT_ID", "RE_DATE", "Discharge time"], how="any"
)

# %%
# calculate raw data's los interval
df_grouped = df_train.groupby("PATIENT_ID")

los_interval_list = []
los_interval_alive_list = []
los_interval_dead_list = []

for name, group in df_grouped:
    sorted_group = group.sort_values(by=["RE_DATE"], ascending=True)
    # print(sorted_group['outcome'])
    # print('---')
    # print(type(sorted_group))
    intervals = sorted_group["RE_DATE"].tolist()
    outcome = sorted_group["outcome"].tolist()[0]
    cur_visits_len = len(intervals)
    # print(cur_visits_len)
    if cur_visits_len == 1:
        continue
    for i in range(1, len(intervals)):
        los_interval_list.append(
            (pd.to_datetime(intervals[i]) - pd.to_datetime(intervals[i - 1])).days
        )
        if outcome == 0:
            los_interval_alive_list.append(
                (pd.to_datetime(intervals[i]) - pd.to_datetime(intervals[i - 1])).days
            )
        else:
            los_interval_dead_list.append(
                (pd.to_datetime(intervals[i]) - pd.to_datetime(intervals[i - 1])).days
            )

los_interval_list = np.array(los_interval_list)
los_interval_alive_list = np.array(los_interval_alive_list)
los_interval_dead_list = np.array(los_interval_dead_list)

output = {
    "overall": los_interval_list,
    "alive": los_interval_alive_list,
    "dead": los_interval_dead_list,
}
# pd.to_pickle(output, 'raw_tjh_los_interval_list.pkl')


# %%
# we have 2 types of prediction tasks: 1) predict mortality outcome, 2) length of stay

# below are all lab test features
labtest_features_str = """
Hypersensitive cardiac troponinI	hemoglobin	Serum chloride	Prothrombin time	procalcitonin	eosinophils(%)	Interleukin 2 receptor	Alkaline phosphatase	albumin	basophil(%)	Interleukin 10	Total bilirubin	Platelet count	monocytes(%)	antithrombin	Interleukin 8	indirect bilirubin	Red blood cell distribution width 	neutrophils(%)	total protein	Quantification of Treponema pallidum antibodies	Prothrombin activity	HBsAg	mean corpuscular volume	hematocrit	White blood cell count	Tumor necrosis factorα	mean corpuscular hemoglobin concentration	fibrinogen	Interleukin 1β	Urea	lymphocyte count	PH value	Red blood cell count	Eosinophil count	Corrected calcium	Serum potassium	glucose	neutrophils count	Direct bilirubin	Mean platelet volume	ferritin	RBC distribution width SD	Thrombin time	(%)lymphocyte	HCV antibody quantification	D-D dimer	Total cholesterol	aspartate aminotransferase	Uric acid	HCO3-	calcium	Amino-terminal brain natriuretic peptide precursor(NT-proBNP)	Lactate dehydrogenase	platelet large cell ratio 	Interleukin 6	Fibrin degradation products	monocytes count	PLT distribution width	globulin	γ-glutamyl transpeptidase	International standard ratio	basophil count(#)	2019-nCoV nucleic acid detection	mean corpuscular hemoglobin 	Activation of partial thromboplastin time	Hypersensitive c-reactive protein	HIV antibody quantification	serum sodium	thrombocytocrit	ESR	glutamic-pyruvic transaminase	eGFR	creatinine
"""

# below are 2 demographic features
demographic_features_str = """
age	gender
"""

labtest_features = [f for f in labtest_features_str.strip().split("\t")]
demographic_features = [f for f in demographic_features_str.strip().split("\t")]
target_features = ["outcome", "LOS"]

# from our observation, `2019-nCoV nucleic acid detection` feature (in lab test) are all -1 value
# so we remove this feature here
labtest_features.remove("2019-nCoV nucleic acid detection")

# %%
# if some values are negative, set it as Null
df_train[df_train[demographic_features + labtest_features] < 0] = np.nan

# %%
# merge lab tests of the same (patient_id, date)
df_train = df_train.groupby(
    ["PATIENT_ID", "RE_DATE", "Discharge time"], dropna=True, as_index=False
).mean()

# %%
# calculate length-of-stay lable
df_train["LOS"] = (
    pd.to_datetime(df_train["Discharge time"]) - pd.to_datetime(df_train["RE_DATE"])
).dt.days

# %%
# if los values are negative, set it as 0
df_train["LOS"] = df_train["LOS"].clip(lower=0)

# %%
# save features' statistics information


def calculate_statistic_info(df, features):
    """all values calculated"""
    statistic_info = {}
    len_df = len(df)
    for _, e in enumerate(features):
        h = {}
        h["count"] = int(df[e].count())
        h["missing"] = str(round(float((100 - df[e].count() * 100 / len_df)), 3)) + "%"
        h["mean"] = float(df[e].mean())
        h["max"] = float(df[e].max())
        h["min"] = float(df[e].min())
        h["median"] = float(df[e].median())
        h["std"] = float(df[e].std())
        statistic_info[e] = h
    return statistic_info


def calculate_middle_part_statistic_info(df, features):
    """calculate 5% ~ 95% percentile data"""
    statistic_info = {}
    len_df = len(df)
    # calculate 5% and 95% percentile of dataframe
    middle_part_df_info = df.quantile([0.05, 0.95])

    for _, e in enumerate(features):
        low_value = middle_part_df_info[e][0.05]
        high_value = middle_part_df_info[e][0.95]
        middle_part_df_element = df.loc[(df[e] >= low_value) & (df[e] <= high_value)][e]
        h = {}
        h["count"] = int(middle_part_df_element.count())
        h["missing"] = (
            str(round(float((100 - middle_part_df_element.count() * 100 / len_df)), 3))
            + "%"
        )
        h["mean"] = float(middle_part_df_element.mean())
        h["max"] = float(middle_part_df_element.max())
        h["min"] = float(middle_part_df_element.min())
        h["median"] = float(middle_part_df_element.median())
        h["std"] = float(middle_part_df_element.std())
        statistic_info[e] = h
    return statistic_info


# labtest_statistic_info = calculate_statistic_info(df_train, labtest_features)


# group by patient_id, then calculate lab test/demographic features' statistics information
groupby_patientid_df = df_train.groupby(
    ["PATIENT_ID"], dropna=True, as_index=False
).mean()


# calculate statistic info (all values calculated)
labtest_patientwise_statistic_info = calculate_statistic_info(
    groupby_patientid_df, labtest_features
)
demographic_statistic_info = calculate_statistic_info(
    groupby_patientid_df, demographic_features
)  # it's also patient-wise

# calculate statistic info (5% ~ 95% only)
demographic_statistic_info_2 = calculate_middle_part_statistic_info(
    groupby_patientid_df, demographic_features
)
labtest_patientwise_statistic_info_2 = calculate_middle_part_statistic_info(
    groupby_patientid_df, labtest_features
)

# take 2 statistics information's union
statistic_info = labtest_patientwise_statistic_info_2 | demographic_statistic_info_2


# %%
# observe features, export to csv file [optional]
to_export_dict = {
    "name": [],
    "missing_rate": [],
    "count": [],
    "mean": [],
    "max": [],
    "min": [],
    "median": [],
    "std": [],
}
for key in statistic_info:
    detail = statistic_info[key]
    to_export_dict["name"].append(key)
    to_export_dict["count"].append(detail["count"])
    to_export_dict["missing_rate"].append(detail["missing"])
    to_export_dict["mean"].append(detail["mean"])
    to_export_dict["max"].append(detail["max"])
    to_export_dict["min"].append(detail["min"])
    to_export_dict["median"].append(detail["median"])
    to_export_dict["std"].append(detail["std"])
to_export_df = pd.DataFrame.from_dict(to_export_dict)
# to_export_df.to_csv('statistic_info.csv')

# %%
# normalize data
def normalize_data(df, features, statistic_info):

    df_features = df[features]
    df_features = df_features.apply(
        lambda x: (x - statistic_info[x.name]["mean"])
        / (statistic_info[x.name]["std"] + 1e-12)
    )
    df = pd.concat(
        [df[["PATIENT_ID", "gender", "RE_DATE", "outcome", "LOS"]], df_features], axis=1
    )
    return df


df_train = normalize_data(
    df_train, ["age"] + labtest_features, statistic_info
)  # gender don't need to be normalized

# %%
# filter outliers
def filter_data(df, features, bar=3):
    for f in features:
        df[f] = df[f].mask(df[f].abs().gt(bar))
    return df


df_train = filter_data(df_train, demographic_features + labtest_features, bar=3)

# %%
# drop rows if all labtest_features are recorded nan
df_train = df_train.dropna(subset=labtest_features, how="all")

# %%
# Calculate data statistics after preprocessing steps (before imputation)

# Step 1: reverse z-score normalization operation
df_reverse = df_train
# reverse normalize data
def reverse_normalize_data(df, features, statistic_info):
    df_features = df[features]
    df_features = df_features.apply(
        lambda x: x * (statistic_info[x.name]["std"] + 1e-12)
        + statistic_info[x.name]["mean"]
    )
    df = pd.concat(
        [df[["PATIENT_ID", "gender", "RE_DATE", "outcome", "LOS"]], df_features], axis=1
    )
    return df


df_reverse = reverse_normalize_data(
    df_reverse, ["age"] + labtest_features, statistic_info
)  # gender don't need to be normalized

statistics = {}

for f in demographic_features + labtest_features:
    statistics[f] = {}


def calculate_quantile_statistic_info(df, features, case):
    """all values calculated"""
    for _, e in enumerate(features):
        # print(e, lo, mi, hi)
        if e == "gender":
            unique, count = np.unique(df[e], return_counts=True)
            data_count = dict(zip(unique, count))  # key = 1 male, 0 female
            print(data_count)
            male_percentage = (
                data_count[1.0] * 100 / (data_count[1.0] + data_count[0.0])
            )
            statistics[e][case] = f"{male_percentage:.2f}% Male"
            print(statistics[e][case])
        else:
            lo = round(np.nanpercentile(df[e], 25), 2)
            mi = round(np.nanpercentile(df[e], 50), 2)
            hi = round(np.nanpercentile(df[e], 75), 2)
            statistics[e][case] = f"{mi:.2f} [{lo:.2f}, {hi:.2f}]"


def calculate_missing_rate(df, features, case="missing_rate"):
    for _, e in enumerate(features):
        missing_rate = round(float(df[e].isnull().sum() * 100 / df[e].shape[0]), 2)
        statistics[e][case] = f"{missing_rate:.2f}%"


tmp_groupby_pid = df_reverse.groupby(["PATIENT_ID"], dropna=True, as_index=False).mean()

calculate_quantile_statistic_info(tmp_groupby_pid, demographic_features, "overall")
calculate_quantile_statistic_info(
    tmp_groupby_pid[tmp_groupby_pid["outcome"] == 0], demographic_features, "alive"
)
calculate_quantile_statistic_info(
    tmp_groupby_pid[tmp_groupby_pid["outcome"] == 1], demographic_features, "dead"
)

calculate_quantile_statistic_info(df_reverse, labtest_features, "overall")
calculate_quantile_statistic_info(
    df_reverse[df_reverse["outcome"] == 0], labtest_features, "alive"
)
calculate_quantile_statistic_info(
    df_reverse[df_reverse["outcome"] == 1], labtest_features, "dead"
)

calculate_missing_rate(
    df_reverse, demographic_features + labtest_features, "missing_rate"
)

export_quantile_statistics = {
    "Characteristics": [],
    "Overall": [],
    "Alive": [],
    "Dead": [],
    "Missing Rate": [],
}
for f in demographic_features + labtest_features:
    export_quantile_statistics["Characteristics"].append(f)
    export_quantile_statistics["Overall"].append(statistics[f]["overall"])
    export_quantile_statistics["Alive"].append(statistics[f]["alive"])
    export_quantile_statistics["Dead"].append(statistics[f]["dead"])
    export_quantile_statistics["Missing Rate"].append(statistics[f]["missing_rate"])

# pd.DataFrame.from_dict(export_quantile_statistics).to_csv('statistics.csv')

# %%
def calculate_data_existing_length(data):
    res = 0
    for i in data:
        if not pd.isna(i):
            res += 1
    return res


# elements in data are sorted in time ascending order
def fill_missing_value(data, to_fill_value=0):
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)
    if data_len == data_exist_len:
        return data
    elif data_exist_len == 0:
        # data = [to_fill_value for _ in range(data_len)]
        for i in range(data_len):
            data[i] = to_fill_value
        return data
    if pd.isna(data[0]):
        # find the first non-nan value's position
        not_na_pos = 0
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        # fill element before the first non-nan value with median
        for i in range(not_na_pos):
            data[i] = to_fill_value
    # fill element after the first non-nan value
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]
    return data


# %%
# fill missing data using our strategy and convert to time series records
grouped = df_train.groupby("PATIENT_ID")

all_x_demographic = []
all_x_labtest = []
all_y = []
all_missing_mask = []

for name, group in grouped:
    sorted_group = group.sort_values(by=["RE_DATE"], ascending=True)
    patient_demographic = []
    patient_labtest = []
    patient_y = []

    for f in demographic_features + labtest_features:
        to_fill_value = (statistic_info[f]["median"] - statistic_info[f]["mean"]) / (
            statistic_info[f]["std"] + 1e-12
        )
        # take median patient as the default to-fill missing value
        # print(sorted_group[f].values)
        fill_missing_value(sorted_group[f].values, to_fill_value)
        # print(sorted_group[f].values)
        # print('-----------')
    all_missing_mask.append(
        (
            np.isfinite(
                sorted_group[demographic_features + labtest_features].to_numpy()
            )
        ).astype(int)
    )

    for _, v in sorted_group.iterrows():
        patient_y.append([v["outcome"], v["LOS"]])
        demo = []
        lab = []
        for f in demographic_features:
            demo.append(v[f])
        for f in labtest_features:
            lab.append(v[f])
        patient_labtest.append(lab)
        patient_demographic.append(demo)
    all_y.append(patient_y)
    all_x_demographic.append(patient_demographic[-1])
    all_x_labtest.append(patient_labtest)

# all_x_demographic (2 dim, record each patients' demographic features)
# all_x_labtest (3 dim, record each patients' lab test features)
# all_y (3 dim, patients' outcome/los of all visits)

# %%
all_x_labtest = np.array(all_x_labtest, dtype=object)
x_lab_length = [len(_) for _ in all_x_labtest]
x_lab_length = torch.tensor(x_lab_length, dtype=torch.int)
max_length = int(x_lab_length.max())
all_x_labtest = [torch.tensor(_) for _ in all_x_labtest]
# pad lab test sequence to the same shape
all_x_labtest = torch.nn.utils.rnn.pad_sequence((all_x_labtest), batch_first=True)

all_x_demographic = torch.tensor(all_x_demographic)
batch_size, demo_dim = all_x_demographic.shape
# repeat demographic tensor
all_x_demographic = torch.reshape(
    all_x_demographic.repeat(1, max_length), (batch_size, max_length, demo_dim)
)
# demographic tensor concat with lab test tensor
all_x = torch.cat((all_x_demographic, all_x_labtest), 2)

all_y = np.array(all_y, dtype=object)
all_y = [torch.Tensor(_) for _ in all_y]
# pad [outcome/los] sequence as well
all_y = torch.nn.utils.rnn.pad_sequence((all_y), batch_first=True)

all_missing_mask = np.array(all_missing_mask, dtype=object)
all_missing_mask = [torch.tensor(_) for _ in all_missing_mask]
all_missing_mask = torch.nn.utils.rnn.pad_sequence((all_missing_mask), batch_first=True)

# %%
# save pickle format dataset (export torch tensor)
pd.to_pickle(all_x, f"./datasets/tongji/processed_data/x.pkl")
pd.to_pickle(all_y, f"./datasets/tongji/processed_data/y.pkl")
pd.to_pickle(x_lab_length, f"./datasets/tongji/processed_data/visits_length.pkl")
pd.to_pickle(all_missing_mask, f"./datasets/tongji/processed_data/missing_mask.pkl")

# %%
# Calculate patients' outcome statistics (patients-wise)
outcome_list = []
y_outcome = all_y[:, :, 0]
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
y_outcome = all_y[:, :, 0]
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
y_los = all_y[:, :, 1]
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
tjh_los_statistics = {
    "overall": los_list,
    "alive": los_alive_list,
    "dead": los_dead_list,
}
# pd.to_pickle(tjh_los_statistics, 'tjh_los_statistics.pkl')

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
# Length-of-stay interval (overall/alive/dead)
los_interval_list = []
los_interval_alive_list = []
los_interval_dead_list = []

y_los = all_y[:, :, 1]
indices = torch.arange(len(x_lab_length), dtype=torch.int64)
for i in indices:
    cur_visits_len = x_lab_length[i]
    if cur_visits_len == 1:
        continue
    for j in range(1, cur_visits_len):
        los_interval_list.append(y_los[i][j - 1] - y_los[i][j])
        if outcome_list[i] == 0:
            los_interval_alive_list.append(y_los[i][j - 1] - y_los[i][j])
        else:
            los_interval_dead_list.append(y_los[i][j - 1] - y_los[i][j])

los_interval_list = np.array(los_interval_list)
los_interval_alive_list = np.array(los_interval_alive_list)
los_interval_dead_list = np.array(los_interval_dead_list)

output = {
    "overall": los_interval_list,
    "alive": los_interval_alive_list,
    "dead": los_interval_dead_list,
}
# pd.to_pickle(output, 'tjh_los_interval_list.pkl')

# %%
len(los_interval_list), len(los_interval_alive_list), len(los_interval_dead_list)

# %%
def check_nan(x):
    if np.isnan(np.sum(x.cpu().numpy())):
        print("some values from input are nan")
    else:
        print("no nan")
