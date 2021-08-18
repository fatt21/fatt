import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

import numpy as np
import pandas as pd
import Dataset
import Perturbation
import Experiment
import StatisticalFairness
import json


########################################################################
# Setup
output = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
n_repetitions = 1000
results = {}


########################################################################
# Adult
config = Experiment.trainingConfiguration('adult')
config['iterations'] = 100
config['aggressiveness'] = 0.01
config['rf-trees'] = 5
config['rf-depth'] = 10
config['dt-standard-depth'] = 10
config['dt-hint-depth'] = 6
Experiment.train(config)
dataset = pd.read_csv(base_dir + '/adult/test-set.csv', header=None, skiprows=1)
columns = Perturbation.readColumns(base_dir + '/adult/columns.csv')
results['adult'] = {}
results['adult']['stats'] = []

print("\t- Testing [ADULT][CAT]")
perturbation = Perturbation.category(dataset, columns, ['sex_male'])
perturbation_path = base_dir + '/output/pert-adult-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['adult']['cat'] = result

print("\t- Testing [ADULT][NOISE]")
noise_on = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
perturbation = Perturbation.noise(dataset, columns, noise_on, 0.3)
perturbation_path = base_dir + '/output/pert-adult-noise.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['adult']['noise'] = result

print("\t- Testing [ADULT][NOISE-CAT]")
noise_on = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
perturbation = Perturbation.noiseCat(dataset, columns, noise_on, 0.3, ['sex_male'])
perturbation_path = base_dir + '/output/pert-adult-noise-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['adult']['noise-cat'] = result

print("\t- Testing [ADULT][STATS]")
for i in range(0, n_repetitions):
    config['seed'] = i
    Experiment.trainMetaSilvae(config)
    result = Experiment.testMetaSilvae({'model': config, 'perturbation': perturbation_path})
    results['adult']['stats'].append(result)

print("\t- Testing [ADULT][CONDITIONAL]")
config['seed'] = 1
noise_on = ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
perturbation = Perturbation.conditionalAttribute(dataset, columns, 'age', -0.13215526244206796, noise_on, 0.2, 0.4)
perturbation_path = base_dir + '/output/pert-adult-conditional.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['adult']['conditional-attribute'] = result

print("\t- Testing [ADULT][STATISTICAL-FAIRNESS]")
config['seed'] = 1
results['adult']['statistical-fairness'] = StatisticalFairness.computeRaw(columns, ['sex_male'], dataset, config)

print("\t- Testing [ADULT][TREE]")
Experiment.trainDecisionTrees(config)
perturbation_path = base_dir + '/output/pert-adult-noise-cat.dat'
results['adult']['decision-trees'] = Experiment.testDecisionTrees({'model': config, 'perturbation': perturbation_path})


########################################################################
# Crime
config = Experiment.trainingConfiguration('crime')
config['iterations'] = 10
config['aggressiveness'] = 0.1
config['dt-standard-depth'] = 6
config['dt-hint-depth'] = 3
Experiment.train(config)
dataset = pd.read_csv(base_dir + '/crime/test-set.csv', header=None, skiprows=1)
columns = Perturbation.readColumns(base_dir + '/crime/columns.csv')
results['crime'] = {}
results['crime']['stats'] = []

print("\t- Testing [CRIME][CAT]")
perturbation = Perturbation.category(dataset, columns, ['state'])
perturbation_path = base_dir + '/output/pert-crime-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['crime']['cat'] = result

print("\t- Testing [CRIME][NOISE]")
noise_on = [c for c in columns if c != 'state']
perturbation = Perturbation.noise(dataset, columns, noise_on, 0.3)
perturbation_path = base_dir + '/output/pert-crime-noise.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['crime']['noise'] = result

print("\t- Testing [CRIME][NOISE-CAT]")
noise_on = [c for c in columns if c != 'state']
perturbation = Perturbation.noiseCat(dataset, columns, noise_on, 0.3, ['state'])
perturbation_path = base_dir + '/output/pert-crime-noise-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['crime']['noise-cat'] = result

print("\t- Testing [CRIME][STATS]")
for i in range(0, n_repetitions):
    config['seed'] = i
    Experiment.trainMetaSilvae(config)
    result = Experiment.testMetaSilvae({'model': config, 'perturbation': perturbation_path})
    results['crime']['stats'].append(result)

print("\t- Testing [CRIME][STATISTICAL-FAIRNESS]")
config['seed'] = 1
results['crime']['statistical-fairness'] = StatisticalFairness.computeRaw(columns, ['state'], dataset, config)

print("\t- Testing [CRIME][TREE]")
Experiment.trainDecisionTrees(config)
perturbation_path = base_dir + '/output/pert-crime-noise-cat.dat'
results['crime']['decision-trees'] = Experiment.testDecisionTrees({'model': config, 'perturbation': perturbation_path})


########################################################################
# Compas 
config = Experiment.trainingConfiguration('compas')
config['iterations'] = 100
config['aggressiveness'] = 0.1
config['dt-standard-depth'] = 6
config['dt-hint-depth'] = 6
Experiment.train(config)
dataset = pd.read_csv(base_dir + '/compas/test-set.csv', header=None, skiprows=1)
columns = Perturbation.readColumns(base_dir + '/compas/columns.csv')
results['compas'] = {}
results['compas']['stats'] = []

print("\t- Testing [COMPAS][CAT]")
perturbation = Perturbation.category(dataset, columns, ['race_caucasian'])
perturbation_path = base_dir + '/output/pert-compas-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['compas']['cat'] = result

print("\t- Testing [COMPAS][NOISE")
noise_on = ['age', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'v_score_text', 'sex_male', 'c_charge_degree_m']
perturbation = Perturbation.noise(dataset, columns, noise_on, 0.3)
perturbation_path = base_dir + '/output/pert-compas-noise.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['compas']['noise'] = result

print("\t- Testing [COMPAS][NOISE-CAT]")
noise_on = ['age', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'v_score_text', 'sex_male', 'c_charge_degree_m']
perturbation = Perturbation.noiseCat(dataset, columns, noise_on, 0.3, ['race_caucasian'])
perturbation_path = base_dir + '/output/pert-compas-noise-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['compas']['noise-cat'] = result

print("\t- Testing [COMPAS][STATS]")
for i in range(0, n_repetitions):
    config['seed'] = i
    Experiment.trainMetaSilvae(config)
    result = Experiment.testMetaSilvae({'model': config, 'perturbation': perturbation_path})
    results['compas']['stats'].append(result)

print("\t- Testing [COMPAS][STATISTICAL-FAIRNESS]")
config['seed'] = 1
results['compas']['statistical-fairness'] = StatisticalFairness.computeRaw(columns, ['race_caucasian'], dataset, config)

print("\t- Testing [COMPAS][TREE]")
Experiment.trainDecisionTrees(config)
perturbation_path = base_dir + '/output/pert-compas-noise-cat.dat'
results['compas']['decision-trees'] = Experiment.testDecisionTrees({'model': config, 'perturbation': perturbation_path})


########################################################################
# German
config = Experiment.trainingConfiguration('german')
config['iterations'] = 1
config['aggressiveness'] = 0.01
config['rf-trees'] = 50
config['rf-depth'] = 10
config['dt-standard-depth'] = 10
config['dt-hint-depth'] = 2
Experiment.train(config)
dataset = pd.read_csv(base_dir + '/german/test-set.csv', header=None, skiprows=1)
columns = Perturbation.readColumns(base_dir + '/german/columns.csv')
results['german'] = {}
results['german']['stats'] = []

print("\t- Testing [GERMAN][CAT]")
perturbation = Perturbation.category(dataset, columns, ['sex_male'])
perturbation_path = base_dir + '/output/pert-german-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['german']['cat'] = result

print("\t- Testing [GERMAN][NOISE]")
noise_on = ['months', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'age', 'number_of_credits', 'people_liable_for', 'telephone_A192', 'foreign_worker_A202']
perturbation = Perturbation.noise(dataset, columns, noise_on, 0.3)
perturbation_path = base_dir + '/output/pert-german-noise.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['german']['noise'] = result

print("\t- Testing [GERMAN][NOISE-CAT]")
noise_on = ['months', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'age', 'number_of_credits', 'people_liable_for', 'telephone_A192', 'foreign_worker_A202']
perturbation = Perturbation.noiseCat(dataset, columns, noise_on, 0.3, ['sex_male'])
perturbation_path = base_dir + '/output/pert-german-noise-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['german']['noise-cat'] = result

print("\t- Testing [GERMAN][STATS]")
for i in range(0, n_repetitions):
    config['seed'] = i
    Experiment.trainMetaSilvae(config)
    result = Experiment.testMetaSilvae({'model': config, 'perturbation': perturbation_path})
    results['german']['stats'].append(result)

print("\t- Testing [GERMAN][CONDITIONAL]")
config['seed'] = 1
noise_on = ['months', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'number_of_credits', 'people_liable_for', 'telephone_A192', 'foreign_worker_A202']
perturbation = Perturbation.conditionalAttribute(dataset, columns, 'age', -0.2238149561371148, noise_on, 0.2, 0.4)
perturbation_path = base_dir + '/output/pert-german-conditional.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['german']['conditional-attribute'] = result

print("\t- Testing [GERMAN][STATISTICAL-FAIRNESS]")
config['seed'] = 1
results['german']['statistical-fairness'] = StatisticalFairness.computeRaw(columns, ['sex_male'], dataset, config)

print("\t- Testing [GERMAN][TREE]")
Experiment.trainDecisionTrees(config)
perturbation_path = base_dir + '/output/pert-german-noise-cat.dat'
results['german']['decision-trees'] = Experiment.testDecisionTrees({'model': config, 'perturbation': perturbation_path})


########################################################################
# Health
config = Experiment.trainingConfiguration('health')
config['iterations'] = 100
config['aggressiveness'] = 0.01
config['dt-standard-depth'] = 15
config['dt-hint-depth'] = 7
Experiment.train(config)
dataset = pd.read_csv(base_dir + '/health/test-set.csv', header=None, skiprows=1)
columns = Perturbation.readColumns(base_dir + '/health/columns.csv')
results['health'] = {}
results['health']['stats'] = []

print("\t- Testing [HEALTH][CAT]")
perturbation = Perturbation.category(dataset, columns, ['AgeAtFirstClaim', 'Sex'])
perturbation_path = base_dir + '/output/pert-health-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['health']['cat'] = result

print("\t- Testing [HEALTH][NOISE]")
noise_on = ['LabCount_total', 'LabCount_months', 'DrugCount_total', 'DrugCount_months', 'Vendor', 'PCP', 'PayDelay', 'max_PayDelay', 'min_PayDelay']
perturbation = Perturbation.noise(dataset, columns, noise_on, 0.3)
perturbation_path = base_dir + '/output/pert-health-noise.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['health']['noise'] = result

print("\t- Testing [HEALTH][NOISE-CAT]")
noise_on = ['LabCount_total', 'LabCount_months', 'DrugCount_total', 'DrugCount_months', 'Vendor', 'PCP', 'PayDelay', 'max_PayDelay', 'min_PayDelay']
perturbation = Perturbation.noiseCat(dataset, columns, noise_on, 0.3, ['AgeAtFirstClaim', 'Sex'])
perturbation_path = base_dir + '/output/pert-health-noise-cat.dat'
Perturbation.savePerturbation(perturbation, perturbation_path)
result = Experiment.test({'model': config, 'perturbation': perturbation_path})
results['health']['noise-cat'] = result

print("\t- Testing [HEALTH][STATS]")
for i in range(0, n_repetitions):
    config['seed'] = i
    Experiment.trainMetaSilvae(config)
    result = Experiment.testMetaSilvae({'model': config, 'perturbation': perturbation_path})
    results['health']['stats'].append(result)

print("\t- Testing [HEALTH][STATISTICAL-FAIRNESS]")
config['seed'] = 1
results['health']['statistical-fairness'] = StatisticalFairness.computeRaw(columns, ['AgeAtFirstClaim', 'Sex'], dataset, config)

print("\t- Testing [HEALTH][TREE]")
Experiment.trainDecisionTrees(config)
perturbation_path = base_dir + '/output/pert-health-noise-cat.dat'
results['health']['decision-trees'] = Experiment.testDecisionTrees({'model': config, 'perturbation': perturbation_path})


########################################################################
# Saves and exits
with open(output, 'w') as f:
    json.dump(results, f, sort_keys=True, indent=4)
