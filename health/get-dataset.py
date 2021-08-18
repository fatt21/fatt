import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

from urllib import request
import zipfile
import numpy as np
import pandas as pd
import Dataset

def preprocess_claims(df_claims):
        df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
        df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

        df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
        df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
        df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
        df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
        df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
        df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
        df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
        df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
        df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
        df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
        df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
        df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

        df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
        df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
        df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
        df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

        df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
        df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
        df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
        df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
        df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
        df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
        df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
        df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
        df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
        df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
        df_claims['LengthOfStay'].fillna(0, inplace=True)
        df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

        for cat_name in ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc']:
            df_claims[cat_name].fillna('{}_?'.format(cat_name), inplace=True)
        binary = []
        categorical = ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc']
        numerical = Dataset.numericalColumns(df_claims, binary, categorical, [])
        df_claims = dataset = Dataset.oneHotEncoding(df_claims, binary, categorical)

        oh = [col for col in df_claims if '=' in col]

        agg = {
            'ProviderID': ['count', 'nunique'],
            'Vendor': 'nunique',
            'PCP': 'nunique',
            'CharlsonIndex': 'max',
            'PayDelay': ['sum', 'max', 'min']
        }
        for col in oh:
            agg[col] = 'sum'

        df_group = df_claims.groupby(['Year', 'MemberID'])
        df_claims = df_group.agg(agg).reset_index()
        df_claims.columns = map(lambda c: c[0] if c[1] in ['', 'sum', 'nunique'] else c[1] + '_' + c[0], df_claims.columns)

        return df_claims


def preprocess_drugs(df_drugs):
    df_drugs.drop(columns=['DSFS'], inplace=True)
    df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
    df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
    df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
    return df_drugs


def preprocess_labs(df_labs):
    df_labs.drop(columns=['DSFS'], inplace=True)
    df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
    df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
    df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
    return df_labs


def preprocess_members(df_members):
    df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
    df_members['Sex'].fillna('?', inplace=True)
    df_members = Dataset.oneHotEncoding(df_members, [], ['Sex', 'AgeAtFirstClaim'])
    return df_members


########################################################################
# Reads dataset
data_file = base_dir + 'dataset-raw.csv'
if not path.exists(data_file):
    zip_file = base_dir + 'dataset-raw.zip'
    Dataset.getFromUrl('https://foreverdata.org/1015/content/HHP_release3.zip', zip_file)
    zf = zipfile.ZipFile(zip_file)

    claims = preprocess_claims(pd.read_csv(zf.open('Claims.csv'), sep=','))
    drugs = preprocess_drugs(pd.read_csv(zf.open('DrugCount.csv'), sep=','))
    labs = preprocess_labs(pd.read_csv(zf.open('LabCount.csv'), sep=','))
    members = preprocess_members(pd.read_csv(zf.open('Members.csv'), sep=','))

    dataset_1 = pd.merge(labs, drugs, on=['MemberID', 'Year'], how='outer')
    dataset_2 = pd.merge(dataset_1, claims, on=['MemberID', 'Year'], how='outer')
    dataset = pd.merge(dataset_2, members, on=['MemberID'], how='outer')

    dataset.drop(['Year', 'MemberID'], axis=1, inplace=True)
    dataset.fillna(0, inplace=True)
    dataset.to_csv(data_file, index=False)
dataset = pd.read_csv(data_file, sep=',')


########################################################################
# Preprocessing
numerical = [c for c in dataset.columns if len(set(dataset[c])) > 2 and c != 'max_CharlsonIndex']
dataset = Dataset.standardize(dataset, numerical)
dataset = Dataset.binarizeByLessThanOrEqualTo(dataset, 'max_CharlsonIndex', 0)
dataset = Dataset.selectLabelColumn(dataset, 'max_CharlsonIndex')
training_set, test_set = Dataset.split(dataset, 0.8)


########################################################################
# Saves files
Dataset.save(dataset, base_dir + 'dataset.csv')
Dataset.save(training_set, base_dir + 'training-set.csv')
Dataset.save(test_set, base_dir + 'test-set.csv')
Dataset.exportColumns(dataset, base_dir + 'columns.csv')
