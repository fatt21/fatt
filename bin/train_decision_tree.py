import sys
import dataset_mapper
import classifier_mapper
import decision_tree


#Sanity check
if (len(sys.argv)) < 3:
    print("Usage: " + sys.argv[0] + " <dataset> <output> [<max_depth> [<criterion>]]")
    sys.exit(-1)


# Reads parameters
dataset_path = sys.argv[1]
output_path = sys.argv[2]
max_depth = 5
if len(sys.argv) > 3:
    max_depth = int(sys.argv[3])
criterion = 'gini'
if len(sys.argv) > 4:
    criterion = sys.argv[4]


# Trains model
dataset_mapper = dataset_mapper.DatasetMapper()
x, y = dataset_mapper.read(dataset_path)

trainer = decision_tree.DecisionTree(max_depth, criterion)
model = trainer.train(x, y)

classifier_mapper = classifier_mapper.ClassifierMapper()
classifier_mapper.create(model, output_path)
