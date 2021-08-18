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
import csv


########################################################################
# Setup
if len(sys.argv) < 3:
    print("Usage: {} <path to result file> <output dir>".format(sys.argv[0]))
    sys.exit()
data_path = sys.argv[1]
out_dir = sys.argv[2]

with open(data_path) as f:
    data = json.load(f)
domains = ['adult', 'compas', 'crime', 'german', 'health']


########################################################################
# Random Forest vs meta-silvae: statistical fairness
with open(out_dir + '/tab-comparison-statistical-fairness.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r |}\n')
    f.write('  \\hline\n')
    f.write('  domain & RF & MS \\\\\n')
    f.write('  \\hline\n')
    for d in domains:
        f.write('  {:8s} & {:5.2f} & {:5.2f} \\\\\n'.format(
            d,
            StatisticalFairness.discrimination(data[d]['statistical-fairness']['meta-silvae']) * 100.0,
            StatisticalFairness.discrimination(data[d]['statistical-fairness']['random-forest']) * 100.0
        ))
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# Random Forest vs meta-silvae: CAT, NOISE and NOISE-CAT fairness
with open(out_dir + '/tab-comparison.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r | r r | r r | r r | r r |}\n')
    f.write('  \\hline\n')
    f.write('         & \multicolumn{2}{| c |}{Acc. \%} & \multicolumn{2}{| c |}{B-Acc. \%} & \multicolumn{6}{| c |}{Fairness \%} \\\\\n')
    f.write('         & & & & & \multicolumn{2}{| c |}{CAT} & \multicolumn{2}{| c |}{NOISE} & \multicolumn{2}{| c |}{NOISE + CAT} \\\\\n')
    f.write('  domain & RF & MS & RF & MS & RF & MS & RF & MS & RF & MS \\\\\n')
    f.write('  \\hline\n')
    for domain in domains:
        f.write('  {:16s} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} \\\\\n'.format(
            domain,
            data[domain]['cat']['random-forest']['accuracy'] * 100.0, data[domain]['cat']['meta-silvae']['accuracy'] * 100.0,
            data[domain]['cat']['random-forest']['balanced-accuracy'] * 100.0, data[domain]['cat']['meta-silvae']['balanced-accuracy'] * 100.0,
            data[domain]['cat']['random-forest']['stability'] * 100.0, data[domain]['cat']['meta-silvae']['stability'] * 100.0,
            data[domain]['noise']['random-forest']['stability'] * 100.0, data[domain]['noise']['meta-silvae']['stability'] * 100.0,
            data[domain]['noise-cat']['random-forest']['stability'] * 100.0, data[domain]['noise-cat']['meta-silvae']['stability'] * 100.0,
        ))  
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# Random Forest vs meta-silvae: ATTRIBUTE fairness
with open(out_dir + '/tab-comparison-attribute.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r |}\n')
    f.write('  \\hline\n')
    f.write('         & \multicolumn{2}{| c |}{Fairness. \%} \\\\\n')
    f.write('  domain & RF & MS \\\\\n')
    f.write('  \\hline\n')
    for domain in ['adult', 'german']:
        f.write('  {:16s} & {:6.2f} & {:6.2f} \\\\\n'.format(
            domain,
            data[domain]['conditional-attribute']['random-forest']['stability'] * 100.0,
            data[domain]['conditional-attribute']['meta-silvae']['stability'] * 100.0,
        ))
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# Random Forest vs meta-silvae: size
with open(out_dir + '/tab-comparison-size.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r |}\n')
    f.write('  domain & RF & MS \\\\\n')
    for domain in domains:
        f.write('  {:16s} & {} & {} \\\\\n'.format(
            domain,
            data[domain]['cat']['random-forest']['size'],
            data[domain]['cat']['meta-silvae']['size']
        ))
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# Random Forest vs meta-silvae: verification time
with open(out_dir + '/tab-comparison-time.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r | r r | r r |}\n')
    f.write('  \\hline\n')
    f.write('         & \multicolumn{6}{| c |}{Avg. Verification Time per Sample (ms)} \\\\\n')
    f.write('         & \multicolumn{2}{| c |}{CAT} & \multicolumn{2}{| c |}{NOISE} & \multicolumn{2}{| c |}{NOISE + CAT} \\\\\n')
    f.write('  domain & RF & MS & RF & MS & RF & MS \\\\\n')
    f.write('  \\hline\n')
    for domain in domains:
        n_samples = data[domain]['cat']['random-forest']['samples']
        factor = 1000.0 / n_samples
        f.write('  {:16s} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} & {:6.2f} \\\\\n'.format(
            domain,
            data[domain]['cat']['random-forest']['time'] * factor, data[domain]['cat']['meta-silvae']['time'] * factor,
            data[domain]['noise']['random-forest']['time'] * factor, data[domain]['noise']['meta-silvae']['time'] * factor,
            data[domain]['noise-cat']['random-forest']['time'] * factor, data[domain]['noise-cat']['meta-silvae']['time'] * factor
        ))
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# Decision Trees: standard, hint and meta-silvae
with open(out_dir + '/tab-decision-trees.tex', 'w') as f:
    f.write('\\begin{tabular}{| r | r r r | r r r | r r r |}\n')
    f.write('  \\hline\n')
    f.write('          & \multicolumn{3}{| c |}{MS} & \multicolumn{3}{| c |}{standard} & \multicolumn{3}{| c |}{hint} \\\\\n')
    f.write('  domain  & Acc. \% & Fair. \% & Size & Acc. \% & Fair. \% & Size & Acc. \% & Fair. \% & Size \\\\\n')
    f.write('  \\hline\n')
    for domain in domains:
        f.write('  {:8s} & {:6.2f} & {:6.2f} & {:5} & {:6.2f} & {:6.2f} & {:5} & {:6.2f} & {:6.2f} & {:5} \\\\\n'.format(
            domain,
            data[domain]['noise-cat']['meta-silvae']['accuracy'] * 100.0,
            data[domain]['noise-cat']['meta-silvae']['stability'] * 100.0,
            data[domain]['noise-cat']['meta-silvae']['size'],
            data[domain]['decision-trees']['standard']['accuracy'] * 100.0,
            data[domain]['decision-trees']['standard']['stability'] * 100.0,
            data[domain]['decision-trees']['standard']['size'],
            data[domain]['decision-trees']['hint']['accuracy'] * 100.0,
            data[domain]['decision-trees']['hint']['stability'] * 100.0,
            data[domain]['decision-trees']['hint']['size']
        ))
    f.write('  \\hline\n')
    f.write('\\end{tabular}\n')


########################################################################
# meta-silvae boxplots
with open(out_dir + '/plot-metrics.csv', 'w') as f:
    w = csv.writer(f, delimiter=' ')
    w.writerow(['#domain', 'criterion', 'accuracy', 'balanced-accuracy', 'fairness', 'time', 'samples', 'size'])
    for domain in domains:
        for e in data[domain]['stats']:
            w.writerow([domain, 'noise-cat', e['accuracy'], e['balanced-accuracy'], e['stability'], e['time'], e['samples'], e['size']])
