from sklearn import tree

class DecisionTree:
    def __init__(self, max_depth, criterion = 'gini', random_state = 0):
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state

    def train(self, x, y):
        clf = tree.DecisionTreeClassifier(
            max_depth = self.max_depth,
            criterion = self.criterion,
            random_state = self.random_state
        )
        clf.fit(x, y)
        return clf
