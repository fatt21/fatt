import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

from os import system
from os import remove
from os import popen
import Perturbation

def trainingConfiguration(domain):
    return {
        'domain': domain,
        'fitness-epsilon': 0.5,
        'fitness-accuracy': 0.9,
        'fitness-stability': 0.1,
        'mutation': 'Z',
        'iterations': 100,
        'aggressiveness': 0.01,
        'seed': 1,
        'rf-trees': 100,
        'rf-depth': 100,
        'rf-criterion': 'gini',
        'dt-standard-depth': 5,
        'dt-hint-depth': 5
    }

def trainingConfigurationSerialize(config):
    return '---'.join(map(str, [
        config['domain'], config['fitness-epsilon'], config['fitness-accuracy'], config['fitness-stability'],
        config['mutation'], config['iterations'], config['aggressiveness'], config['seed'],
        config['rf-trees'], config['rf-depth'], config['rf-criterion']
    ]))

def trainingConfigurationDeserialize(string):
    data = string.split('---')
    return {
        'domain': data[0],
        'fitness-epsilon': float(data[1]),
        'fitness-accuracy': float(data[2]),
        'fitness-stability': float(data[3]),
        'mutation': data[4],
        'iterations': int(data[5]),
        'aggressiveness': float(data[6]),
        'seed': int(data[7]),
        'rf-trees': int(data[8]),
        'rf-depth': int(data[9]),
        'rf-criterion': data[10]
    }

def trainDecisionTree(config, depth, model):
    output = base_dir + '/output/model-dt-' + config['domain'] + '-' + model + '.silva'
    if not path.exists(output):
        command = 'python3 {} {} {} {} {}'.format(
            base_dir + '/bin/train_decision_tree.py',
            base_dir + '/' + config['domain'] + '/training-set.csv',
            output,
            depth,
            'gini'
        )
        system(command)

def trainDecisionTrees(config):
    trainDecisionTree(config, config['dt-standard-depth'], 'standard')
    trainDecisionTree(config, config['dt-hint-depth'], 'hint')

def trainRandomForest(config):
    output = base_dir + '/output/model-rf-' + trainingConfigurationSerialize(config) + '.silva'
    if not path.exists(output):
        command = 'python3 {} {} {} {} {} {}'.format(
            base_dir + '/bin/train_forest.py',
            base_dir + '/' + config['domain'] + '/training-set.csv',
            output,
            config['rf-trees'],
            config['rf-depth'],
            config['rf-criterion']
        )
        system(command)

def trainMetaSilvae(config):
    output = base_dir + '/output/model-ms-' + trainingConfigurationSerialize(config) + '.silva'
    if not path.exists(output):
        command = '{} {} {} --fitness linear {} {} 0 {} 0 0 0 0 0 0 --mutation {} --max-iteration {} --split-search-aggressiveness {} --seed {}'.format(
            base_dir + '/bin/meta-silvae',
            base_dir + '/' + config['domain'] + '/training-set.csv',
            output,
            config['fitness-epsilon'],
            config['fitness-accuracy'],
            config['fitness-stability'],
            config['mutation'],
            config['iterations'],
            config['aggressiveness'],
            config['seed']
        )
        system(command)

def train(config):
    trainRandomForest(config)
    trainMetaSilvae(config)

def testConfigurationSerialize(config):
    return '----'.join(map(str, [
        trainingConfigurationSerialize(config['model']),
        path.basename(config['perturbation'])
    ]))

def readResult(path, model_path):
    command = 'cat {} | tr -s " " | cut -d" " -f4,5,6'.format(path)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    stable = 0
    unstable = 0
    p_label = None
    with popen(command) as f:
        f.readline()
        for row in f:
            data = row.strip().split(' ')
            if len(data) != 3 or data[0] == '(s)':
                break
            label, prediction, stability = data
            if p_label == None:
                p_label = label
            if label == p_label and prediction == p_label:
                tp = tp + 1
            elif label == p_label and prediction != p_label:
                fn = fn + 1
            elif label != p_label and prediction == p_label:
                fp = fp + 1
            elif label != p_label and prediction != p_label:
                tn = tn + 1

            if stability in ['STABLE', 'ROBUST', 'VULNERABLE']:
                stable = stable + 1
            elif stability in ['UNSTABLE', 'FRAGILE', 'BROKEN']:
                unstable = unstable + 1
    command = 'tail -n 1 {} | tr -s " " |  cut -d" " -f3'.format(path)
    with popen(command) as f:
        time = float(f.read().strip())
    command = 'grep LEAF {} | wc -l'.format(model_path)
    with popen(command) as f:
        size = int(f.read().strip())
    
    return {
        'time': time,
        'size': size,
        'samples': tp + tn + fp + fn,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'stable': stable,
        'unstable': unstable,
        'no-info': tp + tn + fp + fn - stable - unstable,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'stability': stable / (tp + tn + fp + fn),
        'balanced-accuracy': 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    }

def testDecisionTree(config, model):
    model_path = base_dir + '/output/model-dt-' + config['model']['domain'] + '-' + model + '.silva'
    output = base_dir + '/output/result-dt-' + config['model']['domain'] + '-' + model + '.dat'
    if not path.exists(output):
        tiers = Perturbation.readTiers(Perturbation.readColumns(base_dir + '/' + config['model']['domain'] + '/columns.csv'))
        command = '{} {} {} --perturbation from-file {} --tiers {} {} --sample-timeout 10 > {}'.format(
            base_dir + '/bin/silva',
            model_path,
            base_dir + '/' + config['model']['domain'] + '/test-set.csv',
            config['perturbation'],
            len(tiers), ' '.join(map(str, tiers)),
            output
        )
        system(command)
    return readResult(output, model_path)

def testDecisionTrees(config):
    return {
        'standard': testDecisionTree(config, 'standard'),
        'hint': testDecisionTree(config, 'hint')
    }

def testRandomForest(config):
    model_path = base_dir + '/output/model-rf-' + trainingConfigurationSerialize(config['model']) + '.silva'
    output = base_dir + '/output/result-rf-' + testConfigurationSerialize(config) + '.dat'
    if not path.exists(output):
        tiers = Perturbation.readTiers(Perturbation.readColumns(base_dir + '/' + config['model']['domain'] + '/columns.csv'))
        command = '{} {} {} --perturbation from-file {} --tiers {} {} --voting average --sample-timeout 10 > {}'.format(
            base_dir + '/bin/silva',
            model_path,
            base_dir + '/' + config['model']['domain'] + '/test-set.csv',
            config['perturbation'],
            len(tiers), ' '.join(map(str, tiers)),
            output
        )
        system(command)
    return readResult(output, model_path)

def testMetaSilvae(config):
    model_path = base_dir + '/output/model-ms-' + trainingConfigurationSerialize(config['model']) + '.silva'
    output = base_dir + '/output/result-ms-' + testConfigurationSerialize(config) + '.dat'
    if not path.exists(output):
        tiers = Perturbation.readTiers(Perturbation.readColumns(base_dir + '/' + config['model']['domain'] + '/columns.csv'))
        command = '{} {} {} --perturbation from-file {} --tiers {} {} --voting average --sample-timeout 10 > {}'.format(
            base_dir + '/bin/silva',
            model_path,
            base_dir + '/' + config['model']['domain'] + '/test-set.csv',
            config['perturbation'],
            len(tiers), ' '.join(map(str, tiers)),
            output
        )
        system(command)
    return readResult(output, model_path)

def test(config):
    rf = testRandomForest(config)
    ms = testMetaSilvae(config)
    return {'random-forest': rf, 'meta-silvae': ms}
