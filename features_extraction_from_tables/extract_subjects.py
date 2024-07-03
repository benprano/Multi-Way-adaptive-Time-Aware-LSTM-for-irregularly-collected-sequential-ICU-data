import gzip
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


def dataframe_from_csv(data):
    with gzip.open(data) as f:
        return pd.read_csv(f)


def read_patients_table(data_path):
    pats = dataframe_from_csv(os.path.join(data_path, 'PATIENTS.csv.gz'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']]
    pats.DOB = pd.to_datetime(pats.DOB, format='%Y-%m-%d %H:%M:%S.%f')
    pats.DOD = pd.to_datetime(pats.DOD, format='%Y-%m-%d %H:%M:%S.%f')
    return pats


def read_admissions_table(data_path):
    admits = dataframe_from_csv(os.path.join(data_path, 'ADMISSIONS.csv.gz'))
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME,
                                      format='%Y-%m-%d %H:%M:%S.%f')
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME,
                                      format='%Y-%m-%d %H:%M:%S.%f')
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME,
                                      format='%Y-%m-%d %H:%M:%S.%f')
    return admits


def read_diagnosis_table(data_path):
    D_ICD = dataframe_from_csv(os.path.join(data_path, 'D_ICD_DIAGNOSES.csv.gz'))
    DIAGNOSIS = dataframe_from_csv(os.path.join(data_path, 'DIAGNOSES_ICD.csv.gz'))
    diagnosis = pd.merge(DIAGNOSIS, D_ICD, on="ICD9_CODE", how="left")
    diagnosis = diagnosis[["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE", "SEQ_NUM"]]
    diagnosis = diagnosis[diagnosis.SEQ_NUM.notnull()]
    diagnosis = diagnosis.loc[diagnosis.groupby(["HADM_ID"])["SEQ_NUM"].idxmax()]
    diagnosis = diagnosis[["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]]
    return diagnosis


def read_icustays_table(data_path):
    stays = dataframe_from_csv(os.path.join(data_path, 'ICUSTAYS.csv.gz'))
    stays = stays[(stays.FIRST_CAREUNIT != "NICU") & (stays.LAST_CAREUNIT != "NICU")]
    stays = stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT',
                   'INTIME', 'OUTTIME', 'DBSOURCE', 'LOS']]
    stays.INTIME = pd.to_datetime(stays.INTIME, format='%Y-%m-%d %H:%M:%S.%f')
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME, format='%Y-%m-%d %H:%M:%S.%f')
    return stays


def read_services_table(data_path):
    services = dataframe_from_csv(os.path.join(data_path, 'SERVICES.csv.gz'))
    services = services[['SUBJECT_ID', 'HADM_ID', 'TRANSFERTIME', 'PREV_SERVICE', 'CURR_SERVICE']]
    services.TRANSFERTIME = pd.to_datetime(services.TRANSFERTIME, format='%Y-%m-%d %H:%M:%S.%f')
    return services


def merge_on_subject_icustay(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def merge_on_admission_diagnosis(table1, table2):
    return table1.merge(table2, how='left', on=['SUBJECT_ID', 'HADM_ID'])


def merge_on_admission_services(table1, table2):
    return table1.merge(table2, how='left', on=['SUBJECT_ID', 'HADM_ID'])


def change(x):
    return x.date()


def add_inhospital_mortality_to_icustays(stays):
    mor = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mor = mor | (stays.DEATHTIME.notnull() & (
            (stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mor.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mor = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mor = mor | (
            stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mor.astype(int)
    return stays


def mortality_labels(stays):
    tlen = (stays.DEATHTIME - stays.INTIME).dt.total_seconds()
    livelen = (stays.DOD - stays.OUTTIME).dt.total_seconds()
    mor24 = (tlen <= 24 * 60 * 60).astype(int)
    mor48 = (tlen <= 48 * 60 * 60).astype(int)
    mor72 = (tlen <= 72 * 60 * 60).astype(int)
    mor30d = (livelen <= 30 * 24 * 60 * 60).astype(int)
    mor1y = (livelen <= 365.245 * 24 * 60 * 60).astype(int)
    stays['MORTALITY_24HRS'] = mor24
    stays['MORTALITY_48HRS'] = mor48
    stays['MORTALITY_72HRS'] = mor72
    stays['MORTALITY_30DAYS'] = mor30d
    stays['MORTALITY_1_YEAR'] = mor1y
    return stays


def age(row):
    possible_age = int((row['ADMITTIME'] - row['DOB']) / pd.Timedelta('365 days'))
    # For those holder than 89, their ages have been removed and set to 300 years prior to admission.
    if possible_age < 0:
        # 91.4 is the median math.ceil(hours) age for those with removed ages. This is what we will set all ages to that are older than 89
        possible_age = 91
    return possible_age


def dischtime_hours(outtime, intime):
    dischtime_hrs = (outtime - intime)
    days, seconds = dischtime_hrs.days, dischtime_hrs.seconds
    hours = days * 24 + seconds // 3600
    return hours


def deathtime_hours(deathtime, intime):
    death_hours = (deathtime - intime)
    days, seconds = death_hours.days, death_hours.seconds
    hours = days * 24 + seconds // 3600
    return hours


def flag_icu(row_1, row_2):
    if row_1 == 1:
        if row_2:
            return 1
    else:
        return 0


def mortality(deathtime, admittime, deathdate, dischtime):
    if deathtime:
        date1 = (deathtime - admittime).days + 1
    if dischtime:
        date2 = (deathdate - dischtime).days + 1
    x = np.array([date1, date2], dtype=np.float64)
    return np.nanmax(x)


def add_icu_mortality_label(stays):
    stays['ICU_MORTALITY'] = np.where(((stays.DOD.notna() & stays.OUTTIME.notna()) & (stays.DOD <= stays.OUTTIME)), 1,
                                      0)
    return stays


def age_correction(row):
    # For those holder than 89, their ages have been removed and set to 300 years prior to admission.
    if row.AGE >= 300:
        # 91.4 is the median age for those with removed ages. This is what we will set all ages to that are older than 89
        row.AGE = 91
        return row.AGE
    else:
        return row.AGE


def icustay_deaths(dod, outtime, flag):
    if pd.notnull(dod) and pd.notnull(outtime):
        if outtime >= dod:
            return 1
        else:
            return 0
    else:
        return flag


def inunit_mortality_to_icustays(dod, outtime, intime):
    if pd.notnull(dod) and pd.notnull(outtime):
        if intime <= dod <= outtime:
            return 1
        else:
            return 0
    else:
        return 0


def icustay_death(dod, outtime):
    if pd.notnull(dod) and (dod <= outtime):
        return 1
    else:
        return 0


def comorbidities(x):
    if "2000" <= x <= "2386":
        return "HEM"
    if "1960" <= x <= "1991":
        return "METSCANCERS"
    if '042' <= x <= '044':
        return "HIV and AIDS"
    else:
        return "OTHERS"


def recover_icustay(mimic_dir, data):
    corrected_data = data[(data.ICUSTAY_ID.notnull()) & (data.HADM_ID.notnull())]
    uncorrected = data[(data.ICUSTAY_ID.isnull()) | (data.HADM_ID.isnull())]
    icustay = pd.read_csv(os.path.join(mimic_dir, "ICUSTAYS.csv.gz"))
    for rw in tqdm(uncorrected.itertuples()):
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


def read_services_table_admission(data_path, subjects):
    services = dataframe_from_csv(os.path.join(data_path, 'SERVICES.csv.gz'))
    services = services[['SUBJECT_ID', 'HADM_ID', 'TRANSFERTIME', 'PREV_SERVICE', 'CURR_SERVICE']]
    services.TRANSFERTIME = pd.to_datetime(services.TRANSFERTIME, format='%Y-%m-%d %H:%M:%S.%f')
    services.loc[services.CURR_SERVICE.str.contains('SURG', case=False), "SURG_FLAG"] = 1
    services['SURG_FLAG'].fillna(value=0.0, inplace=True)
    services['serviceOrder'] = services.groupby(["HADM_ID"]).TRANSFERTIME.transform('count')
    services = services[['SUBJECT_ID', "HADM_ID", "CURR_SERVICE", "SURG_FLAG", "serviceOrder"]]
    services = services[services['CURR_SERVICE'].str.contains('SURG', case=False)]
    services = services.loc[services.groupby(["HADM_ID", "CURR_SERVICE"])["SURG_FLAG"].idxmax()]
    services = services[['SUBJECT_ID', "HADM_ID", "CURR_SERVICE", "SURG_FLAG"]]
    hadmid = [ham for ham in set(subjects["HADM_ID"])]
    services = services.loc[services['HADM_ID'].isin(hadmid)]
    patients = subjects.merge(services, how='outer', left_on=['SUBJECT_ID', 'HADM_ID'],
                              right_on=['SUBJECT_ID', 'HADM_ID'])
    patients['recode'] = patients['ICD9_CODE']
    patients['recode'] = patients['recode'][~patients['recode'].str.contains("[a-zA-Z]").fillna(False)]
    patients['recode'].fillna(value='999', inplace=True)
    patients['recode'] = patients['recode'].str.slice(start=0, stop=4, step=1)
    patients['recode'] = patients['recode'].astype(str)
    patients['COMORBIDITY'] = np.vectorize(comorbidities)(patients['recode'])
    patients.sort_values(['ICUSTAY_ID', 'HADM_ID', "INTIME"], inplace=True)
    patients = patients.drop_duplicates(subset=['ICUSTAY_ID'], keep="first")
    patients.loc[(patients['ADMISSION_TYPE'] == "ELECTIVE") & (
            patients['SURG_FLAG'] == 1), 'TYPE_ADMISSION'] = "ScheduledSurgical"
    patients.loc[(patients['ADMISSION_TYPE'] != "ELECTIVE") & (
            patients['SURG_FLAG'] == 1), 'TYPE_ADMISSION'] = "UnscheduledSurgical"
    patients[["TYPE_ADMISSION"]] = patients[["TYPE_ADMISSION"]].fillna('Medical')
    patients.drop(columns=['DBSOURCE', 'LOS', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE',
                           'EDREGTIME', 'EDOUTTIME', 'Survival', 'Survival_ICU', 'CURR_SERVICE', 'SURG_FLAG', 'recode',
                           "ADMIT", "intime", "dob", "HAS_CHARTEVENTS_DATA", "MORTALITY", "ICU_DEATH",
                           "HOSPITAL_EXPIRE_FLAG", "EXPIRE_FLAG", "ADMITTIME", "DISCHTIME", "DEATHTIME",
                           "DOB", "MARITAL_STATUS", "RELIGION", "LANGUAGE", "DIAGNOSIS", "MORTALITY_INUNIT"], axis=1,
                  inplace=True)
    return patients


def read_stays(data_path):
    patients = read_patients_table(data_path)
    icustays = read_icustays_table(data_path)
    diagnosis = read_diagnosis_table(data_path)
    admissions = read_admissions_table(data_path)
    stays = merge_on_subject_icustay(patients, icustays)
    admits_patients = merge_on_admission_diagnosis(admissions, diagnosis)
    stays = merge_on_subject_admission(stays, admits_patients)
    stays['Survival'] = (stays['DOD'] - stays['ADMITTIME']).dt.days + 1
    stays['Survival_ICU'] = stays.apply(lambda x: mortality(x['DEATHTIME'], x['INTIME'], x['DOD'], x['OUTTIME']),
                                        axis=1)
    stays.loc[stays.Survival_ICU < 0, "Survival_ICU"] = 0

    stays['ICU_DEATH'] = stays.apply(lambda x: icustay_deaths(x['DOD'], x['OUTTIME'], x['HOSPITAL_EXPIRE_FLAG']),
                                     axis=1)
    stays = stays[stays.Survival != -1]
    stays['ADMIT'] = stays['ADMITTIME']
    stays['ADMITTIME'] = stays['ADMITTIME'].apply(change)
    stays['intime'] = stays['INTIME']
    stays['dob'] = stays['DOB']
    stays['DOB'] = stays['DOB'].apply(change)
    stays['AGE'] = stays.apply(lambda e: round((e['ADMITTIME'] - e['DOB']).days / 365.242, 2), axis=1)
    stays["AGE"] = stays.apply(lambda row: age_correction(row), axis=1)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_icu_mortality_label(stays)
    # Calculating the Length of Stay in days per icustay
    stays['INTIME'] = pd.to_datetime(stays['INTIME'])
    stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'])
    stays['LOS_DAYS'] = round((stays['OUTTIME'] - stays['INTIME']).dt.total_seconds() / (24 * 60 * 60), 1)
    del stays['ROW_ID']
    print(f'Length of stays in MICU {len(stays)}')
    return stays


def selected_cohort(data_path):
    stays = read_stays(data_path)
    stays = mortality_labels(stays)
    patients = read_services_table_admission(data_path, stays)
    return patients
