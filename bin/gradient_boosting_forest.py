from catboost import CatBoostClassifier

class GradientBoostingForest:
    def __init__(self, n_trees, max_depth, random_state = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, x, y):
        clf = CatBoostClassifier(
            num_trees = self.n_trees,
            max_depth = self.max_depth,
            learning_rate = 0.5,
            random_state = self.random_state,
            loss_function = 'MultiClass'
        )
        clf.fit(x, y)
        return clf
