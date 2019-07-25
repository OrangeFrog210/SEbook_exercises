# list 2.3: Solving a classification problem using SVM

from sklearn.datasets import load_breast_cancer
# import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

# list 2.1: sklearn.datasets.load_breast_cancer
sample = load_breast_cancer()
print("# of samples: {}".format(len(sample.data)))
print("Contents of the sample: {}".format(sample.data[0]))
print("# of features of each sample: {}".format(len(sample.data[0])))

# list 2.2: The result of classification
print("# of target: {}".format(len(sample.target)))
print("Contents of the targets: {}".format(sample.target[0:30]))


# list 2.3: Solving a classification problem using SVM
clf = SVC(kernel='linear')

x = sample.data
y = sample.target


ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.5, train_size=0.5)
train_index, test_index = next(ss.split(x))
x_train = x[train_index]
y_train = y[train_index]  # answers for learning
x_test = x[test_index]  # data for testing
y_test = y[test_index]  # answers for testing

clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))