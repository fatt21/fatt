import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

import numpy as np
import pandas as pd
import Dataset


########################################################################
# Reads dataset
data_file = base_dir + 'dataset-raw.csv'
Dataset.getFromUrl('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv', data_file)
dataset = pd.read_csv(data_file)


########################################################################
# Preprocessing
dataset = Dataset.normalizeColumnNames(dataset)
dataset = dataset[dataset['days_b_screening_arrest'] >= -30]
dataset = dataset[dataset['days_b_screening_arrest'] <= 30]
dataset = dataset[dataset['is_recid'] != -1]
dataset = dataset[dataset['c_charge_degree'] != '0']
dataset = dataset[dataset['score_text'] != 'N/A']
dataset['in_custody'] = pd.to_datetime(dataset['in_custody'])
dataset['out_custody'] = pd.to_datetime(dataset['out_custody'])
dataset['diff_custody'] = (dataset['out_custody'] - dataset['in_custody']).dt.total_seconds()
dataset['c_jail_in'] = pd.to_datetime(dataset['c_jail_in'])
dataset['c_jail_out'] = pd.to_datetime(dataset['c_jail_out'])
dataset['diff_jail'] = (dataset['c_jail_out'] - dataset['c_jail_in']).dt.total_seconds()


########################################################################
# Drops unused columns/rows
dataset.drop(
    [
        'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
        'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
    ], axis=1, inplace=True
)
dataset = dataset[dataset['race'].isin(['African-American', 'Caucasian'])]
dataset = dataset.drop(['is_recid', 'is_violent_recid', 'violent_recid'], axis=1)
dataset['two_year_recid'] = 1 - dataset['two_year_recid']
dataset = dataset[[
    'two_year_recid', 'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
    'c_charge_desc', 'v_score_text'
]]
dataset['v_score_text'] = [{'Low': 0, 'Medium': 1, 'High': 2}[score] for score in dataset['v_score_text']]

binary = ['sex', 'c_charge_degree', 'race']
categorical = ['c_charge_desc']
numerical = Dataset.numericalColumns(dataset, binary, categorical, ['two_year_recid'])
dataset = Dataset.normalizeColumnValues(dataset, binary + categorical)
dataset = Dataset.standardize(dataset, numerical)
dataset = Dataset.oneHotEncoding(dataset, binary, categorical)
dataset = Dataset.selectLabelColumn(dataset, 'two_year_recid')
training_set, test_set = Dataset.split(dataset, 0.8)


########################################################################
# Saves files
Dataset.save(dataset, base_dir + 'dataset.csv')
Dataset.save(training_set, base_dir + 'training-set.csv')
Dataset.save(test_set, base_dir + 'test-set.csv')
Dataset.exportColumns(dataset, base_dir + 'columns.csv')
