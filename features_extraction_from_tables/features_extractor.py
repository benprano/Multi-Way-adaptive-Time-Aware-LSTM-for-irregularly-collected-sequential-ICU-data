import os
import csv
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from constants_variables import features_lists
from features_extraction_from_tables import extract_subjects

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description="Running this script to extract subject's features from the different tables")
# Provide the path for the dataset where the different tables are  unzipped
parser.add_argument('-data_path', '--data_path',
                    default="/media/sangaria/8TB-FOLDERS/ALL FOLDERS/home/sangaria/Documents/DATA_MIMIC/mimiciii/1.4",
                    type=str, required=False, dest='data_path')
parser.add_argument('-table_name', default="CHARTEVENTS", type=str, required=True)
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-filename', help='filename to save the helpers', type=str, required=True)
args = parser.parse_args()


def data_extraction_from_tables_char_lab_out(data_path, table_name, filename, output_directory, items_dict):
    output_path = output_directory
    mimic3_path = data_path
    stays = extract_subjects.selected_cohort(mimic3_path)
    subjects = stays['SUBJECT_ID'].unique()
    subjects = set([str(s) for s in subjects])
    if table_name == 'CHARTEVENTS':
        nb_rows = {'CHARTEVENTS': 330712484}
    elif table_name == 'LABEVENTS':
        nb_rows = {'LABEVENTS': 27854056}
    else:
        nb_rows = {'OUTPUTEVENTS': 4349219}
    for table in [table_name]:
        tn = os.path.join(mimic3_path, table + '.csv')
        obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUENUM', 'VALUEUOM']
        file = filename + '.csv'
        for row in tqdm(csv.DictReader(open(tn, 'r')), total=nb_rows[table]):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if int(row['ITEMID']) not in items_dict:
                continue
            row_out = {
                'SUBJECT_ID': row['SUBJECT_ID'],
                'HADM_ID': row['HADM_ID'],
                'ICUSTAY_ID': row['ICUSTAY_ID'] if 'ICUSTAY_ID' in row else '',
                'CHARTTIME': row['CHARTTIME'],
                'ITEMID': row['ITEMID'],
                'VALUE': row['VALUE'],
                'VALUENUM': row['VALUENUM'] if 'VALUENUM' in row else '',
                'VALUEUOM': row['VALUEUOM']}
            dn = os.path.join(mimic3_path, output_path)
            if not os.path.exists(dn):
                os.makedirs(dn)
            fn = os.path.join(dn, file)
            if not os.path.exists(fn) or not os.path.isfile(fn):
                f = open(fn, 'w')
                f.write(','.join(obs_header) + '\n')
                f.close()
            with open(fn, 'a') as f_object:
                dictwriter_object = csv.DictWriter(f_object, fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
                dictwriter_object.writerow(row_out)


def data_extraction_inputevents_tables(data_path, table_name, filename, output_directory, items_dict):
    output_path = output_directory
    mimic3_path = data_path
    stays = extract_subjects.selected_cohort(mimic3_path)
    subjects = stays['SUBJECT_ID'].unique()
    subjects = set([str(s) for s in subjects])
    nb_rows = {
        'INPUTEVENTS_CV': 17527937,
        'INPUTEVENTS_MV': 3618993}
    for table in [table_name]:
        tn = os.path.join(mimic3_path, table + '.csv')
        obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'STARTTIME', 'ENDTIME', 'ITEMID', 'STORETIME',
                      'AMOUNT', 'AMOUNTUOM', 'RATE', "RATEUOM", "STATUSDESCRIPTION"]
        file = filename + '.csv'
        for row in tqdm(csv.DictReader(open(tn, 'r')), desc='Processing {} table'.format(table), total=nb_rows[table]):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if int(row['ITEMID']) not in items_dict:
                continue
            row_out = {
                'SUBJECT_ID': row['SUBJECT_ID'],
                'HADM_ID': row['HADM_ID'],
                'ICUSTAY_ID': row['ICUSTAY_ID'],
                'CHARTTIME': row['CHARTTIME'] if 'CHARTTIME' in row else '',
                'ITEMID': row['ITEMID'],
                'STARTTIME': row['STARTTIME'] if 'STARTTIME' in row else '',
                'ENDTIME': row['ENDTIME'] if 'ENDTIME' in row else '',
                'STORETIME': row['STORETIME'] if 'STORETIME' in row else '',
                'AMOUNT': row['AMOUNT'],
                'AMOUNTUOM': row['AMOUNTUOM'],
                'RATE': row['RATE'],
                'RATEUOM': row['RATEUOM'],
                'STATUSDESCRIPTION': row['STATUSDESCRIPTION'] if 'STATUSDESCRIPTION' in row else ''}
            dn = os.path.join(mimic3_path, output_path)
            if not os.path.exists(dn):
                os.makedirs(dn)
            fn = os.path.join(dn, file)
            if not os.path.exists(fn) or not os.path.isfile(fn):
                f = open(fn, 'w')
                f.write(','.join(obs_header) + '\n')
                f.close()
            with open(fn, 'a') as f_object:
                dictwriter_object = csv.DictWriter(f_object, fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
                dictwriter_object.writerow(row_out)


if __name__ == '__main__':
    if args.table_name in ['CHARTEVENTS', 'OUTPUTEVENTS', 'LABEVENTS']:
        if args.table_name == "CHARTEVENTS":
            print(f'[INFO]: Extracting subjects helpers from {args.table_name} table')
            data_extraction_from_tables_char_lab_out(args.data_path, args.table_name,
                                                     args.filename, args.output_path,
                                                     features_lists.CHART_VARIABLES)
        elif args.table_name == "OUTPUTEVENTS":
            print(f'[INFO]: Extracting subjects helpers from {args.table_name} table')
            data_extraction_from_tables_char_lab_out(args.data_path, args.table_name,
                                                     args.filename, args.output_path,
                                                     features_lists.CHART_VARIABLES)
        else:
            print(f'[INFO]: Extracting subjects helpers from {args.table_name} table')
            data_extraction_from_tables_char_lab_out(args.data_path, args.table_name,
                                                     args.filename, args.output_path,
                                                     features_lists.LAB_VARIABLES)
    else:
        if args.table_name in ['INPUTEVENTS_CV', 'INPUTEVENTS_MV']:
            print(f'[INFO]: Extracting subjects helpers from {args.table_name} table')
            data_extraction_inputevents_tables(args.data_path, args.table_name,
                                               args.filename, args.output_path,
                                               features_lists.drugs_list)

