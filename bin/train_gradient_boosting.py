import sys
import dataset_mapper
import classifier_mapper
import gradient_boosting_forest
import json


#Sanity check
if (len(sys.argv)) < 2:
    print("Usage: " + sys.argv[0] + " <dataset> <output> [<n_trees> [<max_depth>]]")
    sys.exit(-1)


# Reads parameters
dataset_path = sys.argv[1]
output_path = sys.argv[2]
n_trees = 5
if len(sys.argv) > 3:
    n_trees = int(sys.argv[3])

max_depth = 5
if len(sys.argv) > 4:
    max_depth = int(sys.argv[4])


# Trains model
dataset_mapper = dataset_mapper.DatasetMapper()
x, y = dataset_mapper.read(dataset_path)

trainer = gradient_boosting_forest.GradientBoostingForest(n_trees, max_depth)
model = trainer.train(x, y)

classifier_mapper = classifier_mapper.ClassifierMapper()
classifier_mapper.create(model, output_path)
