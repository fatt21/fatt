import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

from os import system
from os import remove
from os import popen
import Perturbation
import numpy as np
import pandas as pd
import Dataset
import Perturbation
import Experiment
import json

def groupLabel(sample, group_indexes):
    labels = []
    for l in group_indexes.keys():
        options = group_indexes[l]
        names = options.keys()
        if len(names) == 1:
            key = list(names)[0]
            idx = options[key]
            labels.append(l + '=' + ('T' if sample[idx] == 1 else 'F'))
        else:
            found = False
            for key in names:
                    idx = options[key]
                    if sample[idx] == 1:
                        labels.append(l + '=' + key)
                        found = True
                        break
            if not found:
                labels.append(l + '=' + 'other')
    labels.sort()
    label = ','.join(labels)
    return label

def addRecord(data, group_label, label):
    if group_label not in data:
        data[group_label] = {}
    if label not in data[group_label]:
        data[group_label][label] = 0
    data[group_label][label] = data[group_label][label] + 1
    return data

def compute(columns, groups, dataset, result_path):
    group_indexes = {}
    for i in range(0, len(columns)):
        c = columns[i];
        data = c.split('=', 2)
        if len(data) == 1 and data[0] in groups:
            group_indexes[data[0]] = {
                'true': i
            }
        elif len(data) == 2 and data[0] in groups:
            if data[0] not in group_indexes:
                group_indexes[data[0]] = {}
            group_indexes[data[0]][data[1]] = i

    data = {}
    with open(result_path, 'r') as f:
        f.readline()
        for i in range(0, len(dataset)):
            sample = np.array(dataset.iloc[i][1:])
            line = f.readline()
            marker = groupLabel(sample, group_indexes)
            label = list(filter(None, line.split(' ')))[4]
            data = addRecord(data, marker, label)
    return data

def computeRandomForest(columns, groups, dataset, config):
    model_path = base_dir + '/output/result-rf-' + Experiment.testConfigurationSerialize(config) + '.dat'
    return compute(columns, groups, dataset, model_path)

def computeMetaSilvae(columns, groups, dataset, config):
    model_path = base_dir + '/output/result-ms-' + Experiment.testConfigurationSerialize(config) + '.dat'
    return compute(columns, groups, dataset, model_path)

def computeRaw(columns, groups, dataset, config):
    config = {'model': config, 'perturbation': '/output/pert-{}-cat.dat'.format(config['domain'])}
    return {
        'random-forest': computeRandomForest(columns, groups, dataset, config),
        'meta-silvae': computeMetaSilvae(columns, groups, dataset, config)
    }

def computeLabels(data):
    labels = set()
    for group_name in data:
        labels = labels.union(set(data[group_name].keys()))
    return labels

def computeGroups(data):
    return set(data.keys())

def countSamples(data):
    return sum(map(lambda x: countSamplesInGroup(data, x), data.keys()))

def countSamplesInGroup(data, group_name):
    return sum(map(lambda x: data[group_name][x], data[group_name].keys()))

def countSamplesWithLabel(data, label):
    return sum(map(lambda x: countSamplesWithLabelInGroup(data, label, x), data.keys()))

def countSamplesWithLabelInGroup(data, label, group_name):
    if group_name not in data:
        return 0
    if label not in data[group_name]:
        return 0
    return data[group_name][label]

def discrimination(data):
    labels = list(computeLabels(data))
    groups = list(computeGroups(data))
    k = len(groups)
    labels.sort()
    positive_label = list(filter(lambda x: ',' not in x, labels))[0]
    n_samples = countSamples(data)
    n_positive_samples = countSamplesWithLabel(data, positive_label)

    discrimination = 2.0 / k * sum(map(lambda x: abs(n_positive_samples / n_samples - countSamplesWithLabelInGroup(data, positive_label, x) / countSamplesInGroup(data, x)), groups))
    return discrimination
