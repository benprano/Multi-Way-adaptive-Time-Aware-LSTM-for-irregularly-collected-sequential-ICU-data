import itertools
import multiprocessing
import os
from collections import defaultdict
from itertools import groupby
from multiprocessing import Manager
import numpy as np
import pandas as pd
from tqdm import tqdm

from cleanfeatures.funcs import is_subject_folder
from constants_variables.outter_range_values_for_variables import cols_statics, targets_cols, cols_min, cols_sum, \
    all_features, features_notes, cohort_features


def read_dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col, low_memory=False)


def read_stays(subject_path):
    stays = read_dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    static_variables = stays[cols_statics]
    targets = stays[targets_cols]
    static_variables.columns = static_variables.columns.str.lower()
    targets.columns = targets.columns.str.lower()
    stays.columns = stays.columns.str.lower()
    return stays, static_variables, targets


def read_events(subject_path, hours_min=0, hours_max=47, remove_null=True):
    events = read_dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[(events.VALUENUM.notnull()) & (events.VALUENUM != 0)]
    events.loc[events.HOURS <= 0, "HOURS"] = 0
    events.HOURS = np.abs(events.HOURS)
    events = events[events.HOURS.between(hours_min, hours_max)]
    events = events[events.ITEMID != 224691]
    return events


# Helper function to handle lists
def aggregate_list(column, func):
    return column.apply(lambda x: func(x) if isinstance(x, list) else x)


def count_unique(column):
    return column.apply(lambda x: len(set(x)) if isinstance(x, list) else pd.Series([x]).nunique())


def convert_events_to_timeseries(events, variable_column='LABEL', value_column='VALUENUM',
                                 aggr_min_var=cols_min, aggr_sum_var=cols_sum):
    metadata = events[['HOURS', 'CHARTTIME']].sort_values(by=['CHARTTIME']) \
        .drop_duplicates(keep='first').set_index('HOURS')

    timeseries = events[['HOURS', variable_column, 'CHARTTIME',
                         value_column]].groupby(['HOURS', variable_column])[value_column].apply(list).reset_index()

    timeseries = timeseries.pivot(index='HOURS', columns=variable_column, values=value_column) \
        .merge(metadata, left_index=True, right_index=True) \
        .sort_index(axis=0).reset_index()
    episodes_subjects = timeseries.copy()
    del timeseries["CHARTTIME"]
    aggr_max_colums = list(set(timeseries.columns.tolist()[1:]) - set(aggr_min_var) - set(aggr_sum_var))
    # print(timeseries.columns.tolist()[1:])
    episodes_all = timeseries.copy()
    data_features_stats = timeseries.copy()
    cols = timeseries.columns.tolist()[1:]
    cols_min = set(aggr_min_var) - (set(aggr_min_var) - set(cols))
    cols_sum = set(aggr_sum_var) - (set(aggr_sum_var) - set(cols))

    for col in aggr_max_colums:
        episodes_all[col] = aggregate_list(episodes_all[col], np.nanmax)
    for col in cols_sum:
        episodes_all[col] = aggregate_list(episodes_all[col], np.nansum)
    for col in cols_min:
        episodes_all[col] = aggregate_list(episodes_all[col], np.nanmin)

    all_cols = [column for column in episodes_all.columns.tolist() if column != 'HOURS']
    for col in all_cols:
        data_features_stats[col] = count_unique(data_features_stats[col])

    features_stats = {feat: ["max"] for feat in data_features_stats.columns.to_list() if feat != "HOURS"}
    grouped = data_features_stats.groupby(['HOURS'], as_index=False, dropna=False)
    stats_data = pd.concat([grouped[col].agg(agg_func).rename(columns={col: f'{col}_{agg_func}'})
                            for col, agg_funcs in features_stats.items() for agg_func in agg_funcs],
                           axis=1)
    stats_data = stats_data.loc[:, ~stats_data.columns.duplicated()]
    del stats_data["HOURS"]
    return episodes_subjects, timeseries, episodes_all, data_features_stats, stats_data


def forward_imputation(data):
    store_index = []
    for index in range(len(data)):
        try:
            if len(data) > 1:
                store_index.append([data[index], data[index + 1]])
            else:
                store_index.append(data[index])
        except IndexError as ix:
            continue
    return store_index


def _forward_with_last_measured(data, hours_data=None):
    cols_data = data.columns.to_list()
    times_hours_data = np.arange(0, hours_data, 1)
    notna_cols_indexes, nan_cols_indexes = defaultdict(list), defaultdict(list)
    indexes_last_indicator = defaultdict(list)
    for col in cols_data:
        notna_cols_indexes[col].append(list(data.loc[pd.notna(data[col]), :].index))
        nan_cols_indexes[col].append(list(data.loc[pd.isna(data[col]), :].index))
        indexes_last_indicator[col].append([data[col].notna()[::-1].idxmax(),
                                            data[col].isna()[::-1].idxmax()])
    notna_cols_indexes_ = {key: list(itertools.chain.from_iterable(value))
                           for key, value in notna_cols_indexes.items()
                           if list(itertools.chain.from_iterable(value))}
    notna_cols_indexes_ = {key: value for key, value in notna_cols_indexes_.items() if
                           len(value) != len(times_hours_data)}

    nan_cols_indexes = {key: list(itertools.chain.from_iterable(value))
                        for key, value in nan_cols_indexes.items()}
    nan_cols_indexes_ = {key: value for key, value in nan_cols_indexes.items()
                         if len(value) != len(times_hours_data)}
    nan_cols_indexes_ = {key: value for key, value in nan_cols_indexes_.items() if value}

    indexes_last_indicator = {key: list(itertools.chain.from_iterable(value))
                              for key, value in indexes_last_indicator.items()}
    matrix_indexes_notna = {key: forward_imputation(value) for key, value in notna_cols_indexes_.items()}
    matrix_notna_with_last = {
        key: list(itertools.chain.from_iterable([matrix_indexes_notna[key], [indexes_last_indicator[key]]]))
        for key in indexes_last_indicator if key in matrix_indexes_notna}
    final_matrix_indexes = defaultdict(list)
    for key, vals in matrix_notna_with_last.items():
        for val in vals:
            if isinstance(val, list):
                final_matrix_indexes[key].append(val)
    matrix_range_indexes_cols = {key: [final_matrix_indexes[key], nan_cols_indexes_[key]]
                                 for key in nan_cols_indexes_ if key in final_matrix_indexes}

    range_indexes_cols_imputed = {}
    for key, value in matrix_range_indexes_cols.items():
        range_values = [[notna_ind, nan] for nan in value[1]
                        for notna_ind in value[0] if notna_ind[0] <= nan <= notna_ind[1]]
        range_indexes_cols_imputed[key] = range_values
    return range_indexes_cols_imputed


def forward_with_last_measured_value_with_time_elasped_interval(data_copy, mask_data, hrs_used=None):
    results = _forward_with_last_measured(data_copy, hours_data=hrs_used)
    data = data_copy.copy()
    mask_forward = mask_data.copy()
    for key, values in results.items():
        for k, group in groupby(values, lambda x: x[0]):
            vals_ = list(itertools.chain.from_iterable(list(group)))
            vals_ = np.array([val for val in vals_ if not isinstance(val, list)])
            if k[1] < hrs_used - 1:
                for index in np.arange(k[0] + 1, k[1]):
                    data.at[index, key] = data._get_value(k[0], key)
                    mask_forward.at[index, key] = index - k[0]
            else:
                for index in np.arange(k[0] + 1, k[1] + 1):
                    if pd.isnull(data._get_value(index, key)):
                        data.at[index, key] = data._get_value(k[0], key)
                        mask_forward.at[index, key] = index - k[0]
                    else:
                        pass
    return data, mask_forward


def discretized_events_to_timeseries(timeseries_data, interval_max_data=None, variables=all_features,
                                     var_cohort=cohort_features, aggr_min_var=cols_min, aggr_sum_var=cols_sum):
    timeseries = timeseries_data.copy()
    all_columns = timeseries.columns.to_list()
    features_measured_columns = [column for column in all_columns if column != 'HOURS']
    # Add supplementary variable if not measured with value nan
    additional_columns = list(set(variables) - set(features_measured_columns))
    additional_values = list(itertools.repeat(np.nan, timeseries.shape[0]))
    # Create a DataFrame with additional columns filled with NaN values
    additional_data = pd.DataFrame({col: additional_values for col in additional_columns})
    # Concatenate the new columns with the original DataFrame
    timeseries = pd.concat([timeseries, additional_data], axis=1)
    # Optional: To de-fragment the DataFrame, create a copy
    timeseries.set_index('HOURS', inplace=True)
    timeseries = timeseries.copy()
    # interval sampling for regular timeseries
    interval_sampling = np.arange(0, interval_max_data, 1)
    # present hours helpers measured
    present_hours_data = timeseries.index.values.tolist()
    # adding missing hours with value nan if the helpers has not been measured
    new_indexes = [hour for hour in interval_sampling if hour not in present_hours_data]
    # value nan for variables which has not been measured
    range_vals = list(itertools.repeat(np.nan, len(timeseries.columns.tolist())))
    new_data = pd.DataFrame(columns=timeseries.columns.tolist())
    for index in new_indexes:
        new_series = pd.Series(dict(zip(timeseries.columns.tolist(), range_vals)), name=index)
        new_data = pd.concat([new_data, new_series.to_frame().T])
    episode_timeseries = pd.concat([timeseries, new_data]).sort_index()
    episode_timeseries.index.rename('HOURS', inplace=True)
    episode_timeseries.reset_index(inplace=True)

    # differents type of aggregation based on variables
    aggr_max_colums = list(
        set(list(set(episode_timeseries.columns.tolist()[1:]) - set(aggr_min_var))) - set(aggr_sum_var))
    agg_max_colums = dict.fromkeys(aggr_max_colums, np.nanmax)
    agg_sum_colums = dict.fromkeys(aggr_sum_var, np.nansum)
    agg_min_colums = dict.fromkeys(aggr_min_var, np.nanmin)
    # aggregates functions applied on each variable
    aggregates_func_var = {**agg_max_colums, **agg_sum_colums, **agg_min_colums}

    episode_timeseries = episode_timeseries.groupby(["HOURS"],
                                                    as_index=False, dropna=False).agg(aggregates_func_var)
    episode_timeseries.set_index('HOURS', inplace=True)
    #
    features_all = episode_timeseries.columns.to_list()
    for feature_used in features_all:
        episode_timeseries.loc[episode_timeseries[feature_used] == 0, feature_used] = np.nan
    # Gcs_Eyes,Gcs_Motor, Gcs_Verbal , 'Po2', 'Fio2'
    episode_timeseries = episode_timeseries.astype(float)
    episode_timeseries = episode_timeseries[var_cohort]
    episode_timeseries["Pao2_Fio2"] = episode_timeseries.apply(lambda e: round(100 * (e['Po2'] / e['Fio2']), 2), axis=1)
    episode_timeseries['Gcs_Score'] = episode_timeseries[["Gcs_Eyes", "Gcs_Motor", "Gcs_Verbal"]].values.sum(1)
    episode_timeseries = episode_timeseries.loc[:, episode_timeseries.columns.notna()]
    episode_timeseries = episode_timeseries.reindex(sorted(episode_timeseries.columns), axis=1)
    # print(len(episode_timeseries.columns), episode_timeseries.columns.to_list())
    return episode_timeseries


def process_subject(notes, subject, subjects_root, output_dir, no_events, lock, hrs_data_min=None, hrs_data_max=None):
    subject_path = os.path.join(subjects_root, subject)
    # reading stays & events helpers
    stays, static_variables, targets = read_stays(subject_path)
    notes_subject = notes.loc[notes.ICUSTAY_ID == int(subject)]
    events = read_events(subject_path, hours_min=hrs_data_min, hours_max=hrs_data_max)
    n_events = 0
    if events.shape[0] == 0:
        # no valid events for this subject
        with lock:
            no_events.value += 1
            pass
    else:
        # print(subject)
        subject_all_events, all_events, episodes, data_features_stats, stats_data = convert_events_to_timeseries(events)
        # print("..........................")
        # transform irregular timeseries to regular timeseries
        # print(subject)
        episodes_timeseries = discretized_events_to_timeseries(episodes, interval_max_data=hrs_data_max + 1)
        # print(episodes_timeseries.shape)
        episodes_timeseries_masked = np.where((pd.isnull(episodes_timeseries.values)), 0, 1)
        # masking vector for episode timeseries for generating time delta
        masked_timeseries = np.where((pd.isnull(episodes_timeseries.values)), np.nan, 0)
        # elasped_time_masked_timeseries = np.where((pd.isnull(episodes_timeseries.values)), np.nan, 0)
        masked_episode_timeseries = pd.DataFrame(masked_timeseries,
                                                 columns=episodes_timeseries.columns.to_list())
        ep_timeseries_for_imp, time_elasped_episode_timeseries = forward_with_last_measured_value_with_time_elasped_interval(
            episodes_timeseries,
            masked_episode_timeseries,
            hrs_used=hrs_data_max + 1)

        dn = os.path.join(os.path.join(os.path.dirname(os.path.abspath(subjects_root)), output_dir),
                          str(subject))
        if not os.path.exists(dn):
            os.makedirs(dn)
        # saving helpers for each stay
        stays.to_csv(os.path.join(dn, f"{str(subject)}_stays.csv"), index=False)
        notes_subject.to_csv(os.path.join(dn, f"{str(subject)}_clinical_notes.csv"), index=False)
        targets.to_csv(os.path.join(dn, f"{str(subject)}_targets.csv"), index=False)
        static_variables.to_csv(os.path.join(dn, f"{str(subject)}_static_variables.csv"), index=False)
        np.save(os.path.join(dn, f"{str(subject)}_static_variables.npy"), static_variables.values)
        np.save(os.path.join(dn, f"{str(subject)}_targets.npy"), targets.values)
        np.save(os.path.join(dn, f"{str(subject)}_notes.npy"), notes_subject[features_notes].values)

        # episodes_timeseries
        episodes_timeseries.to_csv(os.path.join(dn, f"{str(subject)}_episodes_timeseries.csv"), index=False)
        np.save(os.path.join(dn, f"{str(subject)}_episodes_timeseries.npy"), episodes_timeseries.values)
        # masking vector for episode timeseries
        np.save(os.path.join(dn, f"{str(subject)}_episodes_timeseries_masked.npy"), episodes_timeseries_masked)
        # masking vector for time delta episode timeseries
        # episodes_timeseries with forward imputation
        ep_timeseries_for_imp.to_csv(os.path.join(dn, f"{str(subject)}_episodes_timeseries_imputated.csv"), index=False)
        np.save(os.path.join(dn, f"{str(subject)}_episodes_timeseries_imputated.npy"), ep_timeseries_for_imp.values)
        # time delta for episode timeseries
        time_elasped_episode_timeseries.to_csv(os.path.join(dn, f"{str(subject)}_time_elasped_episodes_timeseries.csv"),
                                               index=False)
        np.save(os.path.join(dn, f"{str(subject)}_time_elasped_episodes_timeseries.npy"),
                time_elasped_episode_timeseries.values)
        # stats on feature measurements
        stats_data.to_csv(os.path.join(dn, f"{str(subject)}_stats_data_features.csv"), index=False)
        data_features_stats.to_csv(os.path.join(dn, f"{str(subject)}_stats_data_features_all.csv"), index=False)
        all_events.to_csv(os.path.join(dn, f"{str(subject)}_all_events.csv"), index=False)
        subject_all_events.to_csv(os.path.join(dn, f"{str(subject)}_all_events_charttime.csv"), index=False)
        np.save(os.path.join(dn, f"{str(subject)}_stats_data_features.npy"), stats_data.values)
        n_events += events.shape[0]
    return n_events


def patients_events_processing(notes, subjects_root, output_dir, hrs_data_min=None, hrs_data_max=None):
    subdirectories = os.listdir(subjects_root)
    subjects = list(filter(is_subject_folder, subdirectories))
    with Manager() as manager:
        no_events = manager.Value('i', 0)
        lock = manager.Lock()
        n_events = 0
        with multiprocessing.Pool() as pool:
            results = []
            for subject in tqdm(subjects, desc='Iterating over subjects'):
                result = pool.apply_async(process_subject, args=(notes, subject, subjects_root, output_dir, no_events,
                                                                 lock, hrs_data_min, hrs_data_max))
                results.append(result)
            for result in tqdm(results, desc='Processing subjects'):
                n_events += result.get()
        print('n_events: {}'.format(n_events))
        print('number icustays with empty_events: {}'.format(no_events.value))


def all_subjects_data_cohorts(root_subjects_data, output_dir):
    subdirectories = os.listdir(root_subjects_data)
    subjects = list(filter(is_subject_folder, subdirectories))
    ID_subjects, all_statics_data, all_targets = [], [], []
    all_timeseries_data, all_timeseries_imp_data = [], []
    all_mask_data, all_delta_time_data, all_clinical_notes = [], [], []
    for subject in tqdm(subjects, desc='Iterating over subjects'):
        ID_subjects.append(subject)
        subject_path = os.path.join(root_subjects_data, subject)
        # load statics variables
        statics = np.load(os.path.join(subject_path, f"{subject}_static_variables.npy"),
                          allow_pickle=True)
        # load targets helpers
        targets = np.load(os.path.join(subject_path, f"{subject}_targets.npy"),
                          allow_pickle=True)
        # load timeseries helpers
        episodes_timeseries = np.load(os.path.join(subject_path, f"{subject}_episodes_timeseries.npy"),
                                      allow_pickle=True)
        episodes_timeseries_imputated = np.load(
            os.path.join(subject_path, f"{subject}_episodes_timeseries_imputated.npy"), allow_pickle=True)
        subject_notes = np.load(os.path.join(subject_path,
                                             f"{subject}_notes.npy"), allow_pickle=True)
        if subject_notes.shape[0] == 0:
            subject_notes = np.ones((1, subject_notes.shape[-1]))
            subject_notes = np.where(subject_notes == 1., np.nan, subject_notes)
        # load timeseries mask helpers
        mask_data = np.load(os.path.join(subject_path, f"{subject}_episodes_timeseries_masked.npy"),
                            allow_pickle=True)
        # load delta time helpers
        delta_time_data = np.load(os.path.join(subject_path, f"{subject}_time_elasped_episodes_timeseries.npy"),
                                  allow_pickle=True)

        all_statics_data.append(statics)
        all_targets.append(targets)
        all_timeseries_data.append(episodes_timeseries)
        all_timeseries_imp_data.append(episodes_timeseries_imputated)
        all_mask_data.append(mask_data)
        all_delta_time_data.append(delta_time_data)
        all_clinical_notes.append(subject_notes)
        ID_subjects.append(subject)
    dn = os.path.join(os.path.dirname(os.path.abspath(root_subjects_data)), output_dir)
    if not os.path.exists(dn):
        os.makedirs(dn)
    np.savez(os.path.join(dn, f"subjects_data.npz"), statics_data=np.stack(all_statics_data),
             targets_data=np.stack(all_targets), timeseries_data=np.stack(all_timeseries_data),
             timeseries_imp_data=np.stack(all_timeseries_imp_data), mask_data=np.stack(all_mask_data),
             delta_time_data=np.stack(all_delta_time_data), all_clinical_notes_data=np.stack(all_clinical_notes),
             subject_ids=ID_subjects)
