from os import path
from urllib import request
from sklearn.model_selection import train_test_split
import csv
import numpy
import pandas
import ssl
import re

def getFromUrl(url, file_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not path.exists(file_path):
        request.urlretrieve(url, file_path)

def normalizeString(string):
    string = re.sub(r'[\W]', ' ', re.sub(r'(?<!^)(?=[A-Z])', ' ', str(string)).lower())
    return re.sub(r'[\s\_]+', '_', string)

def normalizeColumnNames(dataset):
    columns = dataset.columns
    new_dataset = pandas.DataFrame()
    for c in dataset.columns:
        new_dataset[normalizeString(c)] = dataset[c]
    return new_dataset

def normalizeColumnValues(dataset, columns):
    for c in columns:
        dataset[c] = [normalizeString(v) for v in dataset[c]]
    return dataset

def save(dataset, path):
    with open(path, 'w') as f:
        f.write('# {} {}\n'.format(len(dataset), len(dataset.columns) - 1))
    dataset.to_csv(path, header=False, index=False, mode='a')

def split(dataset, size):
    test_size = 1.0 - size
    return train_test_split(dataset, test_size=test_size, random_state=42)

def exportColumns(dataset, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(dataset.columns)

def binarizeByLessThanOrEqualTo(dataset, column, threshold):
    dataset[column] = dataset[column].values.astype(numpy.float32)
    dataset[column] = (dataset[column] <= threshold).astype(int)
    return dataset

def binarizeByMean(dataset, columns):
    dataset[columns] = dataset[columns].values.astype(numpy.float32)
    dataset[columns] = (dataset[columns] < dataset[columns].mean()).astype(int)
    return dataset

def binarizeByMedian(dataset, columns):
    dataset[columns] = dataset[columns].values.astype(numpy.float32)
    dataset[columns] = (dataset[columns] < dataset[columns].median()).astype(int)
    return dataset

def normalize(dataset, columns):
    dataset[columns] = dataset[columns].values.astype(numpy.float32)
    dataset[columns] = (dataset[columns] - dataset[columns].min()) / (dataset[columns].max() - dataset[columns].min())
    return dataset

def standardize(dataset, columns):
    dataset[columns] = dataset[columns].values.astype(numpy.float32)
    dataset[columns] = (dataset[columns] - dataset[columns].mean()) / dataset[columns].std()
    return dataset

def oneHotEncoding(dataset, binary_columns, categorical_columns):
    dataset = pandas.get_dummies(dataset, columns=binary_columns, drop_first=True)
    return pandas.get_dummies(dataset, columns=categorical_columns, prefix_sep='=')

def selectLabelColumn(dataset, label_column):
    return dataset[[label_column] + [c for c in dataset.columns if c != label_column]]

def numericalColumns(dataset, binary, categorical, exclude=[]):
    return [c for c in dataset.columns if c not in categorical + binary + exclude]
