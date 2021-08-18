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
train_file = base_dir + 'training-raw.csv'
test_file = base_dir + 'test-raw.csv'
Dataset.getFromUrl('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', train_file)
Dataset.getFromUrl('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', test_file)
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
training_set = pd.read_csv(train_file, sep=',', header=None, names=column_names)
test_set = pd.read_csv(test_file, sep=',', header=None, names=column_names)


########################################################################
# Drops unused columns/rows
training_set = training_set.applymap(lambda x: x.strip() if isinstance(x, str) else x)
test_set = test_set.applymap(lambda x: x.strip() if isinstance(x, str) else x)
training_set.replace(to_replace='?', value=np.nan, inplace=True)
test_set.replace(to_replace='?', value=np.nan, inplace=True)
training_set.dropna(axis=0, inplace=True)
test_set.dropna(axis=0, inplace=True)


########################################################################
# Preprocessing
training_set.replace({'<=50K': 0, '>50K': 1}, inplace=True)
test_set.replace({'<=50K.': 0, '>50K.': 1}, inplace=True)
binary = ['sex']
categorical = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
numerical = Dataset.numericalColumns(training_set, binary, categorical, ['income'])

training_set = Dataset.normalizeColumnNames(training_set)
training_set = Dataset.normalizeColumnValues(training_set, binary + categorical)
training_set = Dataset.standardize(training_set, numerical)
training_set = Dataset.oneHotEncoding(training_set, binary, categorical)
training_set = Dataset.selectLabelColumn(training_set, 'income')
test_set = Dataset.normalizeColumnNames(test_set)
test_set = Dataset.normalizeColumnValues(test_set, binary + categorical)
test_set = Dataset.standardize(test_set, numerical)
test_set = Dataset.oneHotEncoding(test_set, binary, categorical)
test_set = Dataset.selectLabelColumn(test_set, 'income')

# Inserts missing column in test set
test_set.insert(
    loc=training_set.columns.get_loc('native_country=holand_netherlands'),
    column='native_country=holand_netherlands', value=0
)
dataset = pd.concat([training_set, test_set], sort=False)


########################################################################
# Saves files
Dataset.save(dataset, base_dir + 'dataset.csv')
Dataset.save(training_set, base_dir + 'training-set.csv')
Dataset.save(test_set, base_dir + 'test-set.csv')
Dataset.exportColumns(dataset, base_dir + 'columns.csv')
