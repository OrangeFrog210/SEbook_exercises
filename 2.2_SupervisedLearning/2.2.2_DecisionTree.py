# 2.4: Classification using decision tree

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
# import pydotplus

sample = load_breast_cancer()
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


clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(x_train, y_train)

predicted = clf.predict(x_test)
score = accuracy_score(predicted, y_test)

print(score)


# 2.5: Vizualization of decision tree
tree.export_graphviz(clf, out_file='tree.dot')
print(sample.feature_names[27])  # the feature at the top