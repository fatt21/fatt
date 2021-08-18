import numpy as np
from sklearn import svm

class SVM:
    def __init__(self, kernel_name, gamma = 1, degree = 1, c = 0, C = 5):
        self.kernel = kernel_name
        self.decision_function_shape = 'ovo'
        self.gamma = gamma
        self.degree = degree
        self.c = c
        self.C = C

    def train(self, x, y):
        clf = svm.SVC(
            C = self.C,
            kernel = self.kernel,
            decision_function_shape = self.decision_function_shape,
            gamma = self.gamma,
            degree = self.degree,
            coef0 = self.c
        )
        clf.fit(x, y)
        return clf
