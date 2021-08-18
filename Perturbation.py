import csv
import numpy as np
import pandas as pd
from os import path
from sys import argv
from sys import exit


def readColumns(path):
    with open(path, 'r') as f:
        columns = [line for line in csv.reader(f)][0][1:]
    return columns

def readTiers(columns):
    tier = []
    lookup = {}
    next_index = 1
    for c in columns:
        data = c.split('=', 2)
        if len(data) == 1:
            tier.append(next_index)
            next_index = next_index + 1
        else:
            prefix, value = data
            if prefix in lookup:
                tier.append(lookup[prefix])
            else:
                lookup[prefix] = next_index
                tier.append(next_index)
                next_index = next_index + 1
    return tier

def category(dataset, columns, attributes):
    rows = []
    for index, record in dataset.iterrows():
        row = []
        for i in range(0, len(columns)):
            c = columns[i]
            column_data = c.split('=', 2)
            if c in attributes or (len(column_data) == 2 and column_data[0] in attributes):
                l = -9999.9
                u = +9999.9
            else:
                l = record[i + 1]
                u = record[i + 1]
            row.append('[{};{}]'.format(l, u))
        rows.append(' '.join(row))
    return rows

def noise(dataset, columns, attributes, epsilon):
    rows = []
    for index, record in dataset.iterrows():
        row = []
        for i in range(0, len(columns)): 
            c = columns[i]
            column_data = c.split('=', 2)
            if c in attributes or (len(column_data) == 2 and column_data[0] in attributes):
                l = record[i + 1] - epsilon
                u = record[i + 1] + epsilon
            else:
                l = record[i + 1]
                u = record[i + 1]
            row.append('[{};{}]'.format(l, u))
        rows.append(' '.join(row))
    return rows

def noiseCat(dataset, columns, noise_attributes, epsilon, cat_attributes):
    rows = []
    for index, record in dataset.iterrows():
        row = []
        for i in range(0, len(columns)):
            c = columns[i]
            column_data = c.split('=', 2)
            if c in noise_attributes or (len(column_data) == 2 and column_data[0] in noise_attributes):
                l = record[i + 1] - epsilon
                u = record[i + 1] + epsilon
            elif c in cat_attributes or (len(column_data) == 2 and column_data[0] in cat_attributes):
                l = -999999.9
                u = +999999.9
            else:
                l = record[i + 1]
                u = record[i + 1]
            row.append('[{};{}]'.format(l, u))
        rows.append(' '.join(row))
    return rows

def conditionalAttribute(dataset, columns, condition_attribute, threshold, attributes, epsilon_1, epsilon_2):
    rows = []
    for index, record in dataset.iterrows():
        epsilon = epsilon_1 if record[1 + columns.index(condition_attribute)] < threshold else epsilon_2
        group_l = -999999.9 if record[1 + columns.index(condition_attribute)] < threshold else threshold
        group_u = threshold if record[1 + columns.index(condition_attribute)] < threshold else 999999.9
        row = []
        for i in range(0, len(columns)):
            c = columns[i]
            column_data = c.split('=', 2)
            if c in attributes or (len(column_data) == 2 and column_data[0] in attributes):
                l = record[i + 1] - epsilon
                u = record[i + 1] + epsilon
            elif c == condition_attribute:
                l = group_l
                u = group_u
            else:
                l = record[i + 1]
                u = record[i + 1]
            row.append('[{};{}]'.format(l, u))
        rows.append(' '.join(row))
    return rows

def savePerturbation(perturbation, output):
    with open(output, 'w') as f:
        for row in perturbation:
            f.write(row + '\n')

if __name__ == '__main__':
    if len(argv) < 5:
        print('Usage: python3 {} <dataset> <columns> <output> <command> [parameters]'.format(argv[0]))
        print('Commands:')
        print('\tshow-columns')
        print('\tshow-tiers')
        print('\tcat column_1,column_2,...,column_n')
        print('\tnoise column_1,column_2,...,column_n epsilon')
        print('\tnoise-cat noise_column_1,noise_column_2,...,noise_column_n epsilon cat_column_1,cat_column_2,...,cat_column_n')
        print('\tconditional-attribute condition_attribute threshold, column_1,column_2,...,column_n epsilon_1 epsilon_2')
        exit()

    dataset = pd.read_csv(argv[1], header=None, skiprows=1)
    columns = readColumns(argv[2])
    tiers = readTiers(columns)
    output = argv[3]
    command = argv[4]

    if command == 'show-columns':
        print('{}: {}'.format(len(columns), ' '.join(map(str, columns))))
    elif command == 'show-tiers':
        print('{}: {}'.format(len(tiers), ' '.join(map(str, tiers))))
    elif command == 'cat':
        perturbation = category(dataset, columns, argv[5].split(','))
        savePerturbation(perturbation, output)
    elif command == 'noise':
        perturbation = noise(dataset, columns, argv[5].split(','), float(argv[6]))
        savePerturbation(perturbation, output)
    elif command == 'noise-cat':
        perturbation = noiseCat(dataset, columns, argv[5].split(','), float(argv[6]), argv[7].split(','))
        savePerturbation(perturbation, output)
    elif command == 'conditional-attribute':
        perturbation = conditionalAttribute(dataset, columns, argv[5], float(argv[6]), argv[7].split(','), float(argv[8]), float(argv[9]))
        savePerturbation(perturbation, output)
        print(perturbation)
