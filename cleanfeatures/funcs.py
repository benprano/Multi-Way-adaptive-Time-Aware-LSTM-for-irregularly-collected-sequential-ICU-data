import gzip
import os

import numpy as np
import pandas as pd
import math
import re
import itertools
import datetime
from tqdm import tqdm


def read_in_chunks(filename, chunk_size=500000):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    for chunk in pd.read_csv(filename, delimiter=',', iterator=True,
                             chunksize=chunk_size, low_memory=False):
        yield chunk


def recover_icustay(mimic_dir, data):
    corrected_data = data[(data.ICUSTAY_ID.notnull()) & (data.HADM_ID.notnull())]
    uncorrected = data[(data.ICUSTAY_ID.isnull()) | (data.HADM_ID.isnull())]
    # print(f"[INFO]: Recovering stays where there is no ICUSTAY_ID or HADM_ID : {uncorrected.shape[0]}")
    icustay = pd.read_csv(os.path.join(mimic_dir, "ICUSTAYS.csv.gz"))
    for rw in tqdm(uncorrected.itertuples(), total=uncorrected.shape[0],
                   desc="[INFO]: Recovering stays where there is no ICUSTAY_ID or HADM_ID"):
        for rt in icustay.itertuples():
            if rw.ICUSTAY_ID == rt.ICUSTAY_ID:
                if pd.isnull(rw.HADM_ID):
                    uncorrected.at[rw.Index, 'HADM_ID'] = rt.HADM_ID
            elif rw.HADM_ID == rt.HADM_ID:
                if pd.isnull(rw.ICUSTAY_ID):
                    uncorrected.at[rw.Index, 'ICUSTAY_ID'] = rt.ICUSTAY_ID
    data_corrected = uncorrected[(uncorrected.ICUSTAY_ID.notnull()) & (uncorrected.HADM_ID.notnull())]
    preprocessed = pd.concat([corrected_data, data_corrected])
    return preprocessed


def processing_weight_height_patients(patients_weight):
    for row in tqdm(patients_weight.itertuples()):
        if row.ITEMID in (3581, 226531):
            patients_weight.at[row.Index, 'VALUENUM'] = np.round((row.VALUENUM * 0.45359237), 4)  # -- lb en kg
        if row.ITEMID == 3582:
            patients_weight.at[row.Index, 'VALUENUM'] = np.round((row.VALUENUM * 0.0283495231), 4)  # -- once en kg
        if row.ITEMID in (920, 1394, 4187, 3486, 226707):
            patients_weight.at[row.Index, 'VALUENUM'] = np.round((row.VALUENUM * 2.54), 4)  # -- HEIGHT
    return patients_weight


def convert_temperature(x):
    if x.ITEMID in (678, 679, 223761):
        return np.round(((x.VALUENUM - 32) * 5 / 9), 4)
    else:
        return x.VALUENUM


def processed_output_urine(x):
    if x.ITEMID == 227488:
        if x.VALUENUM > 0:
            return -1 * x.VALUENUM
    else:
        return x.VALUENUM


def clean_func(item_id, value, variable_func=None):
    for var_id, clean_range in variable_func.items():
        idx = (item_id == var_id)
        if idx:
            return np.where((variable_func[item_id][0] <= value <= variable_func[item_id][1]), value, np.nan)


def remove_outliers_for_variable(events, colname="ITEMID", variable_func=None):
    if variable_func:
        selected_var = events.loc[events[colname].isin(variable_func.keys())]
        no_selected_var = events.loc[~events[colname].isin(variable_func.keys())]
        try:
            selected_var["VALUENUM"] = np.vectorize(clean_func)(selected_var["ITEMID"].values,
                                                                selected_var["VALUENUM"].values, variable_func)
            events_lab = pd.concat([selected_var, no_selected_var])
            return events_lab
        except ValueError as e:
            return events
    else:
        return events


def preprocess_data_fio2_data(data):
    for row in tqdm(data.itertuples()):
        if row.ITEMID in (223835, 727):
            if row.VALUENUM <= 1:
                data.at[row.Index, 'VALUENUM'] = row.VALUENUM * 100
            elif 1 < row.VALUENUM < 21:
                data.at[row.Index, 'VALUENUM'] = np.nan
            elif 21 <= row.VALUENUM <= 100:
                data.at[row.Index, 'VALUENUM'] = row.VALUENUM
            else:
                data.at[row.Index, 'VALUENUM'] = np.nan
        if row.ITEMID in (3420, 3422):
            data.at[row.Index, 'VALUENUM'] = row.VALUENUM
        if row.ITEMID == 190:
            if 0.20 < row.VALUENUM <= 1:
                data.at[row.Index, 'VALUENUM'] = row.VALUENUM * 100
    data.VALUEUOM.replace("torr", '%', regex=True, inplace=True)
    data.VALUENUM = np.round(data.VALUENUM, 3)
    return data


def drugs_processing_to_standard_units(x):
    if x.ITEMID in (30120, 221906, 30047, 30044, 30119, 221289):
        if x.RATEUOM == "mcg/kg/min":
            return np.round(x.RATE, 3)
        elif x.RATEUOM == "mcgmin":
            return np.round((x.RATE / x.VALUENUM), 3)
        else:
            return np.round(x.RATE, 3)
    elif x.ITEMID in (30051, 222315):
        if x.RATE > 0.2:
            return np.round((x.RATE * 5 / 60), 3)
        elif x.RATEUOM == "units/min":
            return np.round((x.RATE * 5), 3)
        elif x.RATEUOM == "units/hour":
            return np.round((x.RATE * 5 / 60), 3)
        else:
            return np.round(x.RATE, 3)

    elif x.ITEMID in (30128, 221749, 30127):
        if x.RATEUOM == "mcg/kg/min":
            return np.round((x.RATE * 0.45), 3)
        elif x.RATEUOM == "mcgmin":
            return np.round((x.RATE * 0.45 / x.VALUENUM), 3)
        else:
            return np.round(x.RATE, 3)

    elif x.ITEMID in (221662, 30043, 30307):
        if x.RATEUOM == "mcg/kg/min":
            return round((x.RATE * 0.01), 3)
        if x.RATEUOM == "mcgmin":
            return round((x.RATE * 0.01 / x.VALUENUM), 3)
        else:
            return round(x.RATE, 3)
    else:
        return round(x.RATE, 3)


def rate_correction(x):
    return x.RATE / x.VALUENUM


def is_subject_folder(x):
    return str.isdigit(x)


def processing_drugs(data_directory, input_directory, filename, chartevents_data):
    dn = os.path.join(data_directory, input_directory)
    if not os.path.exists(dn):
        os.makedirs(dn)
    vaso_pressors = pd.read_csv(os.path.join(dn, f"{filename}.csv"),
                                low_memory=False)
    vaso_pressors = recover_icustay(data_directory, vaso_pressors)
    vaso_pressors["CHARTTIME"].fillna(vaso_pressors['STARTTIME'], inplace=True)
    vaso_pressors["STARTTIME"].fillna(vaso_pressors['CHARTTIME'], inplace=True)
    vaso_pressors["ENDTIME"].fillna(vaso_pressors['STORETIME'], inplace=True)
    vaso_pressors = vaso_pressors[vaso_pressors.STATUSDESCRIPTION != 'Rewritten']
    chartevents_data = pd.read_csv(os.path.join(dn, f"{chartevents_data}.csv"),
                                   low_memory=False)
    weight_items = [762, 763, 3723, 3580, 226512, 3581, 3582, 226531]
    patient_weight = chartevents_data.loc[chartevents_data['ITEMID'].isin(weight_items)]
    del chartevents_data
    patient_weight = processing_weight_height_patients(patient_weight)
    patient_weight = patient_weight[(patient_weight.VALUENUM.notnull()) & (patient_weight.VALUENUM > 0)]
    print("[INFO]: Processing subject's weight and height helpers")
    patient_weight_processed = recover_icustay(data_directory, patient_weight)
    weight = patient_weight_processed.groupby(['ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID'],
                                              as_index=False, dropna=False).agg({'VALUENUM': 'max'})
    # print("length of drugs" , len(vaso_pressors))
    drugs_features = vaso_pressors[vaso_pressors.RATE.notna()]
    drugs_features_processed = pd.merge(drugs_features, weight, how='outer',
                                        left_on=["ICUSTAY_ID", "HADM_ID", "SUBJECT_ID"],
                                        right_on=["ICUSTAY_ID", "HADM_ID", "SUBJECT_ID"])

    drugs_features_processed['VALUENUM'].fillna(80, inplace=True)
    drugs_features_processed["RATE_NORMAL"] = drugs_features_processed["RATE"]
    drugs_features_processed["RATE"] = drugs_features_processed.apply(
        lambda row: drugs_processing_to_standard_units(row), axis=1)
    drugs_features_processed = drugs_features_processed[drugs_features_processed.CHARTTIME.notna()]
    drugs_features_processed['WEIGTH'] = drugs_features_processed["VALUENUM"]
    keep_colums = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "AMOUNT", "RATE", "RATE_NORMAL",
                   "RATEUOM", "WEIGTH"]
    drugs_features_processed = drugs_features_processed[keep_colums]
    drugs_features_processed.loc[drugs_features_processed.RATE == 0, "RATE"] = np.nan
    drugs_features_processed.rename(columns={'RATE': 'VALUENUM'}, inplace=True)

    drugs_features_processed.to_csv(os.path.join(dn, f"{filename}_processed.csv"), index=False)
    print("length of drugs processed", len(drugs_features_processed))


g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}
comorbidity_mapping = {"OTHERS": 0, "METSCANCERS": 1, "HEM": 2, "HIV and AIDS": 3}
admission_type_mapping = {"ScheduledSurgical": 0, "Medical": 1, "UnscheduledSurgical": 2}
e_map = {'ASIAN': 1, 'BLACK': 2, 'CARIBBEAN ISLAND': 2, 'HISPANIC': 3, 'SOUTH AMERICAN': 3,
         'WHITE': 4, 'MIDDLE EASTERN': 4, 'PORTUGUESE': 4, 'AMERICAN INDIAN': 0, 'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0, 'PATIENT DECLINED TO ANSWER': 0, 'UNKNOWN': 0, 'OTHER': 0, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


