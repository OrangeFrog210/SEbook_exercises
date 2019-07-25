# 2.8 Using the arranged data list, classifying using decision tree

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus

sample = load_breast_cancer()

x = sample.data
y = sample.target


# creating a dummy data
def normalize(x):
    min = x.min()
    max = x.max()

    result = (10 * (x-min)/(max-min)).astype(np.int64)
    return result


dummy_x = x[:, [0, 6, 27]]
dummy_y = y
dummy_x[:, 0] = normalize(dummy_x[:, 0])
dummy_x[:, 1] = normalize(dummy_x[:, 1])
dummy_x[:, 2] = normalize(dummy_x[:, 2])


ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.5, train_size=0.5)
train_index, test_index = next(ss.split(dummy_x))
x_train = dummy_x[train_index]  # data for learning
y_train = dummy_y[train_index]  # answers for learning
x_test = dummy_x[test_index]  # data for testing
y_test = dummy_y[test_index]  # answers for testing

clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
score = accuracy_score(predicted, y_test)

tree.export_graphviz(clf, out_file='dummy_tree.dot')

print(score)