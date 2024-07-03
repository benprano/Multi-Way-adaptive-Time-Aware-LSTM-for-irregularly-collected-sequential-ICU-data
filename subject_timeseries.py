import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd

from timeseries_data.generate_timeseries import patients_events_processing, all_subjects_data_cohorts

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description="Running this script to extract subject's features from the different tables")
# Provide the path for the dataset where the different tables are  unzipped
parser.add_argument('-data_path', '--data_path',
                    default="/media/sangaria/8TB-FOLDERS/ALL FOLDERS/home/sangaria/Documents/DATA_MIMIC/mimiciii/1.4",
                    type=str, required=False, dest='data_path')
parser.add_argument('-path_dir', help='subject helpers path', type=str, default="SUBJECTS_ICU_DATA", required=False)
parser.add_argument('-subject_path', help='subject icustay helpers', type=str, required=True)
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-cohort_path', type=str, required=True)
parser.add_argument('-hrs_data_min', help='hours helpers min', type=int, required=True)
parser.add_argument('-hrs_data_max', help='hours helpers max', type=int, required=True)
args = parser.parse_args()
if __name__ == '__main__':
    print("[INFO]: Generating subject's timeseries helpers with their medical history")
    root_path = os.path.join(args.data_path, args.path_dir)
    subjects_data_path = os.path.join(root_path, args.subject_path.lower())
    output_dir_data_path = os.path.join(root_path, f"{args.hrs_data_max + 1}_hours_{args.output_path}_data".lower())
    subjects_notes = pd.read_csv(os.path.join(root_path, "all_notes_medical_history_subjects.csv"))
    patients_events_processing(subjects_notes, subjects_data_path, output_dir_data_path,
                               args.hrs_data_min, args.hrs_data_max)
    print()
    print(f"[INFO]: Creating cohort helpers using {args.hrs_data_max + 1} first hours ICU helpers")
    cohort_dir = os.path.join(root_path, f"{args.cohort_path}_{args.hrs_data_max + 1}_hours_data".upper())
    all_subjects_data_cohorts(output_dir_data_path, cohort_dir)
    """ 
    Script's time execution : 7:41:26
    Readme how to run:
    -path_dir:  Main directory where all the data have been saved!
    
    python subject_timeseries.py -path_dir "NEW_FILE" -subject_path "icustays" -output_path "subjects_timeseries" -cohort_path "EPISODES_SUBJECTS" -hrs_data_min 0 -hrs_data_max 47
    """
