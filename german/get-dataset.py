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
Dataset.getFromUrl('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', data_file)
column_names = [
    'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
    'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
    'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
    'foreign_worker', 'credit'
]
dataset = pd.read_csv(data_file, sep=' ', header=None, names=column_names)


########################################################################
# Drops unused columns/rows
dataset['sex'] = dataset['personal_status'].replace({'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'})
dataset.drop('personal_status', axis=1, inplace=True)


########################################################################
# Preprocessing
binary = ['telephone', 'foreign_worker', 'sex']
categorical = [
    'status', 'credit_history', 'purpose', 'savings', 'employment',
    'other_debtors', 'property', 'installment_plans', 'housing',
    'skill_level'
]
numerical = Dataset.numericalColumns(dataset, binary, categorical, ['credit'])

dataset = Dataset.standardize(dataset, numerical)
dataset = Dataset.oneHotEncoding(dataset, binary, categorical)
dataset = Dataset.selectLabelColumn(dataset, 'credit')
training_set, test_set = Dataset.split(dataset, 0.8)


########################################################################
# Saves files
Dataset.save(dataset, base_dir + 'dataset.csv')
Dataset.save(training_set, base_dir + 'training-set.csv')
Dataset.save(test_set, base_dir + 'test-set.csv')
Dataset.exportColumns(dataset, base_dir + 'columns.csv')
