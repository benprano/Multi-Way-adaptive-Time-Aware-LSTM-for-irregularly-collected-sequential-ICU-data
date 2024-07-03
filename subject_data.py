import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from cleanfeatures.funcs import remove_outliers_for_variable, processing_weight_height_patients, \
    preprocess_data_fio2_data, convert_temperature, processed_output_urine, read_in_chunks, transform_gender, \
    transform_ethnicity, comorbidity_mapping, admission_type_mapping, recover_icustay, is_subject_folder, \
    processing_drugs
from constants_variables.outter_range_values_for_variables import vitals_features, clean_fns_chart_var, drugs_cols, \
    drugs_features, labs_features, clean_fns
from features_extraction_from_tables import extract_subjects

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description="Running this script to extract subject's features from the different tables")
# Provide the path for the dataset where the different tables are  unzipped
parser.add_argument('-data_path', '--data_path',
                    default="/media/sangaria/8TB-FOLDERS/ALL FOLDERS/home/sangaria/Documents/DATA_MIMIC/mimiciii/1.4",
                    type=str, required=False, dest='data_path')
parser.add_argument('-ICUSTAYS', type=str, default="ICUSTAYS", required=False)
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-subject_path', type=str, required=True)
parser.add_argument('-inputs', help='inputevents file', type=str, required=True)
parser.add_argument('-charts', help='chartevents file', type=str, required=True)
parser.add_argument('-labs', help='labevents file', type=str, required=True)
parser.add_argument('-outputs', help='outputevents file', type=str, required=True)
args = parser.parse_args()

keep_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'AGE', 'MORTALITY_INHOSPITAL', 'ICU_MORTALITY',
             'LOS_DAYS', 'MORTALITY_24HRS', 'MORTALITY_48HRS', 'MORTALITY_72HRS', 'MORTALITY_30DAYS',
             'MORTALITY_1_YEAR', 'Gender', 'Ethnicity', 'Comorbidity', 'Admission_type']


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        if not os.path.exists(dn):
            os.makedirs(dn)
        stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


def process_subject_data(subject_id, stays, output_path, filename, map_func=None, map_var=None):
    subject_stays = stays[(stays.SUBJECT_ID == subject_id)]
    if subject_stays.shape[0] == 0:
        # There is no events helpers for this subject
        return
    subject_stays['HOURS'] = np.round(
        (subject_stays["CHARTTIME"].sub(subject_stays["INTIME"])).apply(lambda x: x / np.timedelta64(1, 'm')).fillna(
            0).astype('int64') / 60)
    dn = os.path.join(output_path, str(subject_id))
    if not os.path.exists(dn):
        os.makedirs(dn)
    subject_stays = remove_outliers_for_variable(subject_stays, colname="ITEMID", variable_func=map_func)
    subject_stays["VALUENUM"] = subject_stays.apply(lambda row: convert_temperature(row), axis=1)
    subject_stays["VALUENUM"] = subject_stays.apply(lambda row: processed_output_urine(row), axis=1)
    # mapping multiple itemid to unique variable name
    subject_stays['LABEL'] = subject_stays['ITEMID'].map(map_var)
    subject_stays = preprocess_data_fio2_data(subject_stays)
    subject_stays = processing_weight_height_patients(subject_stays)
    if os.path.isfile(os.path.join(dn, f"{filename}.csv")):
        events = pd.read_csv(os.path.join(dn, f"{filename}.csv"))
        events = pd.concat([events, subject_stays.sort_values(by='INTIME')])
        events.to_csv(os.path.join(dn, f"{filename}.csv"), index=False)
    else:
        subject_stays.sort_values(by='INTIME').to_csv(os.path.join(dn, f"{filename}.csv"), index=False)


# Subject vitals features helpers
def read_events_table_and_break_up_by_subject_chartevents_threading(stays, output_path, filename, subjects=None,
                                                                    map_func=None, map_var=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]

    # Prepare arguments for multiprocessing
    arg = [(subject_id, stays, output_path, filename,
            map_func, map_var) for subject_id in subjects]

    # Determine the number of processes to use
    num_processes = min(cpu_count(), nb_subjects)

    # Use a Pool to manage multiple processes
    with Pool(processes=num_processes) as pool:
        # Use starmap to pass arguments to the function
        list(tqdm(pool.starmap(process_subject_data, arg),
                  total=nb_subjects, desc='Processing subjects in parallel'))


# Subject lab features helpers
def read_events_table_and_break_up_by_subject_labevents(stays, output_path, filename, subjects=None, map_func=None,
                                                        map_var=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up events helpers by subjects'):
        print(subject_id)
        subject_stays = stays[(stays.SUBJECT_ID == subject_id)]
        if subject_stays.shape[0] == 0:
            # there is no events helpers
            continue
        subject_stays['HOURS'] = np.round((subject_stays["CHARTTIME"].sub(subject_stays["INTIME"])).apply(
            lambda x: x / np.timedelta64(1, 'm')).fillna(0).astype('int64') / 60)
        dn = os.path.join(output_path, str(subject_id))
        if not os.path.exists(dn):
            os.makedirs(dn)
        subject_stays = remove_outliers_for_variable(subject_stays, colname="ITEMID", variable_func=map_func)
        subject_stays['LABEL'] = subject_stays['ITEMID'].map(map_var)
        if os.path.isfile(os.path.join(dn, f"{filename}.csv")):
            events = pd.read_csv(os.path.join(dn, f"{filename}.csv"))
            events = pd.concat([events, subject_stays.sort_values(by='INTIME')])
            events.to_csv(os.path.join(dn, f"{filename}.csv"), index=False)
        else:
            subject_stays.sort_values(by='INTIME').to_csv(os.path.join(dn, f"{filename}.csv"), index=False)


def process_subject_data_lab(subject_id, stays, output_path, filename, map_func=None, map_var=None):
    subject_stays = stays[(stays.SUBJECT_ID == subject_id)]
    if subject_stays.shape[0] == 0:
        # There is no events helpers for this subject
        return
    subject_stays['HOURS'] = np.round(
        (subject_stays["CHARTTIME"].sub(subject_stays["INTIME"])).apply(lambda x: x / np.timedelta64(1, 'm')).fillna(
            0).astype('int64') / 60)
    dn = os.path.join(output_path, str(subject_id))
    if not os.path.exists(dn):
        os.makedirs(dn)
    subject_stays = remove_outliers_for_variable(subject_stays, colname="ITEMID", variable_func=map_func)
    subject_stays['LABEL'] = subject_stays['ITEMID'].map(map_var)
    if os.path.isfile(os.path.join(dn, f"{filename}.csv")):
        events = pd.read_csv(os.path.join(dn, f"{filename}.csv"))
        events = pd.concat([events, subject_stays.sort_values(by='INTIME')])
        events.to_csv(os.path.join(dn, f"{filename}.csv"), index=False)
    else:
        subject_stays.sort_values(by='INTIME').to_csv(os.path.join(dn, f"{filename}.csv"), index=False)


def read_events_and_break_up_by_subject_labevents(stays, output_path, filename, subjects=None,
                                                  map_func=None, map_var=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]

    # Prepare arguments for multiprocessing
    arg = [(subject_id, stays, output_path, filename,
            map_func, map_var) for subject_id in subjects]

    # Determine the number of processes to use
    num_processes = min(cpu_count(), nb_subjects)

    # Use a Pool to manage multiple processes
    with Pool(processes=num_processes) as pool:
        # Use starmap to pass arguments to the function
        list(tqdm(pool.starmap(process_subject_data_lab, arg),
                  total=nb_subjects, desc='Processing subjects in parallel'))


def extract_subject_stay_data(mimic3_path, output_path):
    stays = extract_subjects.selected_cohort(mimic3_path)
    data = {'ICUSTAY_ID': stays.ICUSTAY_ID}
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.ETHNICITY))
    data_updated = pd.DataFrame(data)
    subjects = stays.merge(data_updated, how='inner', left_on=['ICUSTAY_ID'], right_on=['ICUSTAY_ID'])
    subjects['Comorbidity'] = subjects['COMORBIDITY'].map(comorbidity_mapping)
    subjects['Admission_type'] = subjects['TYPE_ADMISSION'].map(admission_type_mapping)
    subjects = subjects[keep_cols]
    subjects.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'], inplace=True)
    all_subjects = subjects.SUBJECT_ID.unique()
    # Breaking up stays helpers by subject
    break_up_stays_by_subject(subjects, output_path, subjects=all_subjects)
    return subjects


def break_up_chartevents_by_subject(patients, data_path, path_to_store, filename_path, fileName):
    for event_chunk_data in read_in_chunks(os.path.join(filename_path, f"{fileName.lower()}.csv")):
        chuck_data = recover_icustay(data_path, event_chunk_data)
        chuck_data["VALUENUM"] = chuck_data["VALUENUM"].astype(float)
        chuck_data = chuck_data.merge(D_ITEMS[["ITEMID", "LABEL"]])
        chuck_data_patients = chuck_data.merge(patients[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
                                               on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
        chuck_data_patients.CHARTTIME = pd.to_datetime(chuck_data_patients.CHARTTIME, format='%Y-%m-%d %H:%M:%S.%f')
        chuck_data_patients.INTIME = pd.to_datetime(chuck_data_patients.INTIME, format='%Y-%m-%d %H:%M:%S.%f')
        chuck_data_patients.CHARTTIME = chuck_data_patients.CHARTTIME.astype('datetime64')
        chuck_data_patients.INTIME = chuck_data_patients.INTIME.astype('datetime64')
        chuck_data_patients = chuck_data_patients[(chuck_data_patients.VALUENUM.notna()) &
                                                  (chuck_data_patients.VALUENUM != 0)]
        chuck_data_patients.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'], inplace=True)
        subjects_charts_id = chuck_data_patients.SUBJECT_ID.unique()
        read_events_table_and_break_up_by_subject_chartevents_threading(chuck_data_patients,
                                                                        path_to_store, "events",
                                                                        subjects=subjects_charts_id,
                                                                        map_func=clean_fns_chart_var,
                                                                        map_var=vitals_features)


def break_up_labevents_by_subject(patients, path_to_store, filename_path, fileName):
    for lab_chunk_data in read_in_chunks(os.path.join(filename_path, f"{fileName.lower()}.csv")):
        chuck_data_lab = lab_chunk_data.merge(D_LABITEMS[["ITEMID", "LABEL"]])
        chuck_data_lab.drop(columns=['ICUSTAY_ID'], axis=1, inplace=True)
        chuck_data_lab["VALUENUM"] = chuck_data_lab["VALUENUM"].astype(float)
        chuck_data_lab_patients = chuck_data_lab.merge(patients[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
                                                       on=["SUBJECT_ID", "HADM_ID"])
        chuck_data_lab_patients.CHARTTIME = pd.to_datetime(chuck_data_lab_patients.CHARTTIME,
                                                           format='%Y-%m-%d %H:%M:%S.%f')
        chuck_data_lab_patients.INTIME = pd.to_datetime(chuck_data_lab_patients.INTIME, format='%Y-%m-%d %H:%M:%S.%f')
        chuck_data_lab_patients.CHARTTIME = chuck_data_lab_patients.CHARTTIME.astype('datetime64')
        chuck_data_lab_patients.INTIME = chuck_data_lab_patients.INTIME.astype('datetime64')
        chuck_data_lab_patients = chuck_data_lab_patients[(chuck_data_lab_patients.VALUENUM.notna()) &
                                                          (chuck_data_lab_patients.VALUENUM != 0)]
        chuck_data_lab_patients.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'], inplace=True)
        subjects_lab_id = chuck_data_lab_patients.SUBJECT_ID.unique()
        read_events_and_break_up_by_subject_labevents(chuck_data_lab_patients,
                                                      path_to_store, "events",
                                                      subjects=subjects_lab_id,
                                                      map_func=clean_fns,
                                                      map_var=labs_features)


def break_up_outputevents_by_subject(patients, data_path, path_to_store, filename_path, fileName):
    for output_chunk_data in read_in_chunks(os.path.join(filename_path, f"{fileName.lower()}.csv")):
        chuck_data_output = recover_icustay(data_path, output_chunk_data)
        chuck_data_output["VALUENUM"] = chuck_data_output["VALUE"]
        chuck_data_output["VALUENUM"] = chuck_data_output["VALUENUM"].astype(float)
        chuck_data_output = chuck_data_output.merge(D_ITEMS[["ITEMID", "LABEL"]])
        output_data_patients = chuck_data_output.merge(patients[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
                                                       on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
        output_data_patients.CHARTTIME = pd.to_datetime(output_data_patients.CHARTTIME, format='%Y-%m-%d %H:%M:%S.%f')
        output_data_patients.INTIME = pd.to_datetime(output_data_patients.INTIME, format='%Y-%m-%d %H:%M:%S.%f')
        output_data_patients.CHARTTIME = output_data_patients.CHARTTIME.astype('datetime64')
        output_data_patients.INTIME = output_data_patients.INTIME.astype('datetime64')
        output_data_patients = output_data_patients[(output_data_patients.VALUENUM.notna())
                                                    & (output_data_patients.VALUENUM != 0)]
        output_data_patients.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'], inplace=True)
        subjects_charts_id = output_data_patients.SUBJECT_ID.unique()
        read_events_table_and_break_up_by_subject_chartevents_threading(output_data_patients,
                                                                        path_to_store, "events",
                                                                        subjects=subjects_charts_id,
                                                                        map_func=clean_fns_chart_var,
                                                                        map_var=vitals_features)


def break_up_inputevents_by_subject(patients, path_to_store, filename_path, fileName):
    for input_chunk_data in read_in_chunks(os.path.join(filename_path, f"{fileName.lower()}_processed.csv")):
        chuck_drugs_data = input_chunk_data.merge(D_ITEMS[["ITEMID", "LABEL"]])
        chuck_drugs_data.rename(columns={'RATEUOM': 'VALUEUOM'}, inplace=True)
        chuck_drugs_data["VALUE"] = chuck_drugs_data["VALUENUM"]
        chuck_drugs_data["VALUENUM"] = chuck_drugs_data["VALUENUM"].astype(float)
        chuck_drugs_data = chuck_drugs_data[drugs_cols]

        chuck_drugs_data_pats = chuck_drugs_data.merge(patients[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
                                                       on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
        chuck_drugs_data_pats.CHARTTIME = pd.to_datetime(chuck_drugs_data_pats.CHARTTIME, format='%Y-%m-%d %H:%M:%S.%f')
        chuck_drugs_data_pats.INTIME = pd.to_datetime(chuck_drugs_data_pats.INTIME, format='%Y-%m-%d %H:%M:%S.%f')
        chuck_drugs_data_pats.CHARTTIME = chuck_drugs_data_pats.CHARTTIME.astype('datetime64')
        chuck_drugs_data_pats.INTIME = chuck_drugs_data_pats.INTIME.astype('datetime64')
        chuck_drugs_data_pats = chuck_drugs_data_pats[(chuck_drugs_data_pats.VALUENUM.notna()) &
                                                      (chuck_drugs_data_pats.VALUENUM != 0)]
        chuck_drugs_data_pats.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'], inplace=True)
        subjects_drugs_id = chuck_drugs_data_pats.SUBJECT_ID.unique()
        read_events_table_and_break_up_by_subject_chartevents_threading(chuck_drugs_data_pats,
                                                                        path_to_store, "events",
                                                                        subjects=subjects_drugs_id,
                                                                        map_var=drugs_features)


def generate_patients_stays(subjects_root_path, output_dir):
    subdirectories = os.listdir(subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))
    for subject in tqdm(subjects, desc='Iterating over subjects'):
        try:
            # reading tables of this subject
            stays = pd.read_csv(os.path.join(subjects_root_path, subject, 'stays.csv'))
            events = pd.read_csv(os.path.join(subjects_root_path, subject, "events.csv"))
        except IOError as e:
            sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject))
            continue
        if events.shape[0] == 0:
            # no valid events for this subject
            continue
        diff_icustays = stays.ICUSTAY_ID.unique()
        pat = stays.ICUSTAY_ID.unique()
        nb_subjects = pat.shape[0]
        for subject_id in tqdm(diff_icustays, total=nb_subjects, desc='Breaking up stays by subjects'):
            dn = os.path.join(os.path.join(os.path.dirname(os.path.abspath(subjects_root_path)), output_dir),
                              str(subject_id))
            if not os.path.exists(dn):
                os.makedirs(dn)
            stays.loc[stays.ICUSTAY_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                                      index=False)
            events.loc[events.ICUSTAY_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'events.csv'),
                                                                                        index=False)


if __name__ == '__main__':
    D_ITEMS = pd.read_csv(os.path.join(args.data_path, "D_ITEMS.csv.gz"))
    D_LABITEMS = pd.read_csv(os.path.join(args.data_path, "D_LABITEMS.csv.gz"))
    print("[INFO]: Processing drugs, weight, height features to standard units conversion")
    processing_drugs(args.data_path, args.output_path, args.inputs.lower(), args.charts.lower())
    print("[INFO]: Done")
    print()
    print("[INFO]: Generating subject's stay helpers")
    store_path = os.path.join(os.path.join(args.data_path, args.output_path), args.subject_path)
    all_data_path = os.path.join(args.data_path, args.output_path)
    subjects_all = extract_subject_stay_data(args.data_path, store_path)
    print("[INFO]: breaking up chartevents by subject")
    break_up_chartevents_by_subject(subjects_all, args.data_path, store_path, all_data_path, args.charts)
    print()
    print("[INFO]: breaking up labevents by subject")
    break_up_labevents_by_subject(subjects_all, store_path, all_data_path, args.labs)
    print()
    print("[INFO]: breaking up outputevents by subject")
    break_up_outputevents_by_subject(subjects_all, args.data_path, store_path, all_data_path, args.outputs)
    print()
    print("[INFO]: breaking up inputevents by subject")
    break_up_inputevents_by_subject(subjects_all, store_path, all_data_path, args.inputs)
    print()
    print("[INFO]: breaking up different stays by subject")
    generate_patients_stays(store_path, args.ICUSTAYS.lower())

    """
    Readme how to run me!!!:
    -output_path : Filename where subject's data have been saved!
    -subject_path : File to save subject's events data!
    -inputs: Save inputevents data to file INPUTEVENTS!
    
    python subject_data.py -output_path "NEW_FILE" -subject_path "subjects" -inputs "INPUTEVENTS" -charts "CHARTEVENTS" -labs "LABEVENTS" -outputs "outputevents"
    
    """
