import os
import matplotlib.pyplot as plt
import numpy as np

features = ['Albumin', 'Alt', 'Amylase', 'Anion_Gap', 'Argatroban', 'Ast', 'Atypical_Lymphocytes', 'Bands','Base_Excess', 'Basophils', 'Bicarbonate', 'Bilirubin', 
	    'Calcium_Total','Calculated_Total_Co2', 'Carboxyhemoglobin', 'Chloride', 'Creatinine', 'D-Dimer', 'Diastolic_Blood', 'Diltiazem', 'Dobutamine', 'Dopamine', 
	    'Eosinophils', 'Epinephrine', 'Epithelial_Cells', 'Esmolol', 'Fentanyl', 'Ferritin', 'Fibrinogen', 'Fio2', 'Furosemide', 'Gcs_Eyes', 'Gcs_Motor', 'Gcs_Score', 
	    'Gcs_Verbal', 'Glucose', 'Granulocyte_Count', 'Heart_Rate', 'Height', 'Hematocrit', 'Hemoglobin', 'Hydromorphone', 'Inr(Pt)', 'Integrelin', 'Ketone',
            'Labetalol', 'Lactate', 'Lipase', 'Lorazepam', 'Lymphocytes', 'Magnesium', 'Mch', 'Mchc', 'Mcv', 'Mean_Arterial', 'Metamyelocytes', 'Methemoglobin', 'Midazolam', 
            'Milrinone', 'Monocytes', 'Morphine[Sulfate]', 'Myelocytes', 'Natrecor', 'Neutrophils', 'Nicardipine', 'Nitroglycerine', 'Nitroprusside', 'Norepinehrine', 'Nucleated_Rbc',
            'O2_Flow', 'Pao2_Fio2', 'Pco2', 'Ph', 'Phenylephrine', 'Phosphate', 'Platelet_Count', 'Po2', 'Potassium', 'Precedex', 'Promyelocytes', 'Propofol', 'Protein_Total', 'Ptt', 
            'Rbc', 'Required_O2', 'Respiration_Rate', 'Reticulocyte_Count', 'Sao2', 'Sodium', 'Specific_Gravity', 'Spo2', 'Systolic_Blood', 'Temperature', 'Thrombin', 'Transferrin', 'Troponin_T',
            'Urea', 'Urine_Output', 'Urobilinogen', 'Vancomycin_Level', 'Vasopressin', 'Wbc', 'Weight']
drugs_features= ['Diltiazem', 'Dobutamine', 'Dopamine', 'Epinephrine', 'Esmolol', 'Fentanyl', 'Furosemide', 'Hydromorphone', 'Integrelin', 'Labetalol', 'Lorazepam', 'Midazolam', 'Milrinone','Morphine[Sulfate]',
		 'Natrecor', 'Nicardipine', 'Nitroglycerine', 'Nitroprusside', 'Norepinehrine', 'Phenylephrine', 'Vasopressin']
features_with_indices = [(idx, feature) for idx, feature in enumerate(features)
                         if feature in drugs_features_imp]
indices = [idx for idx, feature in enumerate(features) if feature in drugs_features_imp]


class LoadingData:
    def __init__(self, data_dir):
        super(LoadingData, self).__init__()
        self.data_dir = data_dir

    def load_and_preprocess_features(self):
        subjects = np.load(os.path.join(self.data_dir, "subjects_data.npz"), allow_pickle=True)
        S = np.squeeze(subjects["statics_data"], axis=1)
        targets = subjects["targets_data"]
        timeseries = np.array(subjects["timeseries_imp_data"], dtype=float)
        timeseries_data = np.array(subjects["timeseries_data"], dtype=float)
        clinical_notes = np.array(subjects["all_clinical_notes_data"])
        timedelta = np.array(subjects["delta_time_data"], dtype=float)
        # Impute drugs features for Missing with value zero
        mask = np.zeros(timeseries.shape[-1], dtype=bool)
        columns_to_replace = np.array(indices)
        mask[columns_to_replace] = True
        new_timeseries = np.where(np.isnan(timeseries) & mask[None, None, :], 0, timeseries)
        new_timeseries_data = np.where(np.isnan(timeseries_data) & mask[None, None, :], 0, timeseries_data)
        # Imputation of others features with median value
        col_median = np.nanmedian(new_timeseries, axis=0)
        subjects_timeseries = np.where(np.isnan(new_timeseries), col_median, new_timeseries)
        repeats = new_timeseries_data.shape[1]
        # Concatenation of statics features helpers with temporal features
        temporal_statics = np.tile(S, repeats).reshape(S.shape[0], repeats, S.shape[-1])
        temporal_data = np.concatenate((new_timeseries_data, temporal_statics), axis=-1)
        temporal_timedelta = np.concatenate((timedelta, np.zeros_like(temporal_statics)), axis=-1)
        temporal_timedelta[np.isnan(temporal_timedelta)] = 999
        temporal_timedelta[np.isinf(temporal_timedelta)] = 999
        temporal_data_features = np.concatenate((subjects_timeseries, temporal_statics), axis=-1)
        freq_list, timeseries_last_obs_data = [], []
        nb_pats, seq, n_features = temporal_data.shape
        for i in range(nb_pats):
            # Extract frequencies of measurement of each feature
            data_patient = np.expand_dims(temporal_data[i, :, :], axis=0)
            nan_counts = np.sum(np.isfinite(data_patient), axis=(0, 1))
            freq_list.append(np.repeat(np.expand_dims(nan_counts, axis=0), repeats, axis=0))
            # Extract Last observation record of each feature
            Index_Last = (~np.isnan(temporal_data[i, :, :])).cumsum(0).argmax(0)
            Last_Indices = np.reshape(Index_Last, (1, n_features))
            Last_Values = np.take_along_axis(temporal_data[i, :, :], Last_Indices, axis=0)
            timeseries_last_obs_data.append(np.repeat(Last_Values, repeats, axis=0))
        freqs = np.stack(freq_list)
        last_obs_data = np.stack(timeseries_last_obs_data)
        col_median_last = np.nanmedian(last_obs_data, axis=0)
        last_obs_data = np.where(np.isnan(last_obs_data), col_median_last, last_obs_data)
        data_normalized, last_data_normalized, data_max_min = self.z_normalization(temporal_data_features,
                                                                                   last_obs_data)
        all_targets, los_max_min = self.load_targets(targets)
        return (data_normalized, clinical_notes, temporal_timedelta, last_data_normalized,
                freqs), all_targets, data_max_min, los_max_min

    @staticmethod
    def z_normalization(all_timeseries, all_last_data):
        tim_data_max = np.nanmax(all_timeseries, axis=0)
        tim_data_min = np.nanmin(all_timeseries, axis=0)
        last_data_max = np.nanmax(all_last_data, axis=0)
        last_data_min = np.nanmin(all_last_data, axis=0)
        features_data = (all_timeseries - tim_data_min) / (tim_data_max - tim_data_min)
        last_data = (all_last_data - last_data_min) / (last_data_max - last_data_min)
        return features_data, last_data, {"data_max": tim_data_max, "data_min": tim_data_min}

    @staticmethod
    def load_targets(targets_data):
        los = targets_data[:, :, 7]
        # Preprocess LOS
        TARGET_LOS = np.where(los <= 0, 0.1, los)
        # LOS in hours
        TARGET_LOS_HOURS = TARGET_LOS * 24
        # Apply log10 Transformation to LOS overcome the positive skew present in LOS distribution
        TARGET_LOS_LOG10 = np.log10(TARGET_LOS_HOURS)
        targets_max = TARGET_LOS_LOG10.max()
        targets_min = TARGET_LOS_LOG10.min()
        targets_los = (TARGET_LOS_LOG10 - targets_min) / (targets_max - targets_min)
        targets_dict = {"hos_mor": targets_data[:, :, 0], "icu_mor": targets_data[:, :, 1],
                        "mor_48": targets_data[:, :, 3],
                        "mor_72": targets_data[:, :, 4], "mor_30days": targets_data[:, :, 5],
                        "mor_1_year": targets_data[:, :, 6],
                        "los": targets_los}
        n_bins = 30
        colors = ['blue', 'orange', 'green']
        plt.hist(TARGET_LOS_LOG10, n_bins, histtype='step', stacked=True, fill=False, label=colors,
                 density=1, edgecolor='purple', linewidth=2.0)
        # plt.legend(loc="upper right")
        plt.gca().set(title='Frequency Histogram LOS', ylabel='Frequency', xlabel='LOS(hours) Normalized with LOG10')
        plt.savefig("LOS_With_LOG10.pdf", bbox_inches='tight', dpi=100)
        plt.close()
        # Without transformation
        plt.hist(TARGET_LOS_HOURS, n_bins, histtype='step', stacked=True, fill=False, label=colors,
                 density=1, edgecolor='purple', linewidth=2.0)
        # plt.legend(loc="upper right")
        plt.gca().set(title='Frequency Histogram LOS', ylabel='Frequency', xlabel='LOS(hours) UnNormalized with LOG10')
        plt.savefig("LOS_Without_LOG10.pdf", bbox_inches='tight', dpi=100)
        plt.close()
        return targets_dict, {"los_max": targets_max, "los_min": targets_min}
