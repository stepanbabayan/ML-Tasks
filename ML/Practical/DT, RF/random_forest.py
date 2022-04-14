from os import replace
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class RandomForestClassifier:
    DEFAULT_TREES_COUNT = 100
    DEFAULT_MAX_DEPTH = 3
    CRITERIONS = ('gini', 'entropy')
    DEFAULT_CRITERION = CRITERIONS[0]
    MIN_SAMPLES_SPLIT = 2
    DEFAULT_COLUMNS_COUNT=1

    def __init__(self, n_estimators=DEFAULT_TREES_COUNT, columns_count=DEFAULT_COLUMNS_COUNT, max_depth=DEFAULT_MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=DEFAULT_CRITERION):
        """
        :param n_estimators: Count of trees.
        """
        self._n_estimators = n_estimators
        self._columns_count = columns_count
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._criterion = criterion

        self._trees: list[DecisionTreeClassifier] = []

        if self._criterion not in self.CRITERIONS:
            possible_criterions = ', '.join(self.CRITERIONS)
            raise AttributeError(f'`criterion` should be one of the following: {possible_criterions}')
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        self._x_train: pd.DataFrame = x_train
        self._y_train: pd.Series = y_train
        self.classes = np.unique(self._y_train)

        for _ in range(self._n_estimators):
            indexes = np.random.choice(self._x_train.index, self._x_train.shape[0], replace=True)
            x_i = self._x_train.loc[indexes]
            y_i = self._y_train.loc[indexes]
            clf = DecisionTreeClassifier(
                columns_count=self._columns_count,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                criterion=self._criterion
            )
            clf.fit(x_i, y_i)
            self._trees.append(clf)
    
    def predict_trees(self, x_test):
        all_predictions = []
        tree_predictions = []
        for tree in self._trees:
            tree_predictions.append(tree.predict(x_test))
        all_predictions = np.array(tree_predictions).T
        return all_predictions

    def predict(self, x_test):
        all_predictions = self.predict_trees(x_test)
        return np.array(pd.DataFrame(all_predictions).mode(axis=1)[0])

    def predict_proba(self, x):
        pass

class Node:
    def __init__(self, feature, threshold, left_node, right_node, value, probas):
        self._feature = feature
        self._threshold = threshold
        self._left_node = left_node
        self._right_node = right_node
        self._value = value
        self._probas = probas

    @property
    def feature(self):
        return self._feature

    @property
    def threshold(self):
        return self._threshold

    @property
    def left_node(self):
        return self._left_node

    @property
    def right_node(self):
        return self._right_node

    @property
    def value(self):
        return self._value

    @property
    def probas(self):
        return self._probas


class DecisionTreeClassifier:
    DEFAULT_MAX_DEPTH = 3
    CRITERIONS = ('gini', 'entropy')
    DEFAULT_CRITERION = CRITERIONS[0]
    MIN_SAMPLES_SPLIT = 2
    DEFAULT_COLUMNS_COUNT='all'

    def __init__(self, columns_count=DEFAULT_COLUMNS_COUNT, max_depth=DEFAULT_MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=DEFAULT_CRITERION):
        """
        :param max_depth: The maximum depth of the tree.
        :param min_features: If there are less than `min_features` features in the node, do not grow the tree. If none, ignore the value.
        :param criterion: The criterion for building the tree. Can be either 'gini' or 'entropy'
        """
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._criterion = criterion
        self._columns_count = columns_count

        if self._criterion not in self.CRITERIONS:
            possible_criterions = ', '.join(self.CRITERIONS)
            raise AttributeError(f'`criterion` should be one of the following: {possible_criterions}')

        self.base_node = None  # initialy None, grow the tree (using Nodes) when fitting

    def fit(self, x_train, y_train):
        """
        Fit the classifier.
        """
        self._x_train: pd.DataFrame = x_train
        self._y_train: pd.Series = y_train
        
        self.classes = list(set(self._y_train))

        self.base_node = self._grow_tree(self._x_train, self._y_train)

    def _criterion_entropy(self, y):
        """
        Calculate the entropy criterion
        TODO: implement
        """

    def _criterion_gini(self, y):
        """
        Calculate the Gini criterion
        TODO: implement
        """
        _, counts = np.unique(y, return_counts=True)
        p_mk = counts/y.shape[0]
        return (p_mk*(1- p_mk)).sum()

    def _criterion_f(self, y):
        """
        Calculate the criterion using `self._criterion`
        Call the `_criterion_{criterion}` method.
        """
        return getattr(self, f'_criterion_{self._criterion}')(y)

    
    

    def _grow_tree(self, X: pd.DataFrame, y, depth=0):
        """
        Grow the tree.
        TODO: implement
        Hint: use recursion
        """
        if self._columns_count == 'all':
            self._columns_count = self._x_train.shape[1]
        columns = np.random.choice(X.columns.values, self._columns_count, replace=False)
        classes, counts = np.unique(y, return_counts=True)
        probas = {}
        for i,cls in enumerate(classes):
            probas[cls] = counts[i]/y.shape[0]
        for cls in self.classes:
            probas.setdefault(cls, 0)
        probas_sorted = sorted(probas.items())
        probas_list = [value[1] for value in probas_sorted]
        
        if depth >= self._max_depth or X.shape[0] < self._min_samples_split:
            node = Node(None, None, None, None, value=classes[np.argmax(counts)], probas=probas_list)
        else:   
            gini_min = 1
            node_feature=None
            node_threshold=None
            for feature in columns:
                feature_values = X[feature]
                feature_values_su = np.unique(feature_values)
                for threshold in feature_values_su[:-1]:
                    y_left, y_right = y[X[feature] <= threshold], y[X[feature] > threshold]
                    w_l, w_r = np.array([y_left.shape[0], y_right.shape[0]]) / y.shape[0]
                    gini_lr =  self._criterion_f(y_left)*w_l + self._criterion_f(y_right)*w_r
                    if gini_lr < gini_min:
                        gini_min = gini_lr
                        node_threshold = threshold
                        node_feature = feature
            
            if node_feature:
                X_left, X_right = X[X[node_feature] <= node_threshold], X[X[node_feature] > node_threshold]
                y_left, y_right = y[X[node_feature] <= node_threshold], y[X[node_feature] > node_threshold]

                left_node = self._grow_tree(X_left, y_left, depth+1)
                right_node = self._grow_tree(X_right, y_right, depth+1)

                node = Node(node_feature, node_threshold, left_node, right_node, value=classes[np.argmax(counts)], probas=probas_list)
            else:
                node = Node(None, None, None, None, value=classes[np.argmax(counts)], probas=probas_list)

        return node
    
    def tree_value(self, node: Node, x, value_type='value'):
        if value_type not in ['value', 'probas']:
            raise ValueError
        feature = node.feature
        threshold = node.threshold
        if feature is None:
            if value_type=='value':
                return node.value
            elif value_type=='probas':
                return node.probas
        if x[feature] > threshold:
            next_node = node.right_node
        else:
            next_node = node.left_node

        return self.tree_value(next_node, x, value_type)
        

    def predict(self, x_test):
        """
        Predict which class is each data in x
        :param x: features matrix
        """
        y_pred = []
        for i, row in x_test.iterrows():
            y_pred.append(self.tree_value(self.base_node, row))
        return np.array(y_pred)

    def predict_proba(self, x):
        """
        Predict the probability, that x is of class 1.
        TODO: implement
        """
        y_probas = []
        for i, row in x.iterrows():
            y_probas.append(self.tree_value(self.base_node, row, value_type='probas'))
        return np.array(y_probas)



from sklearn.datasets import load_iris


iris = load_iris()
X, y = pd.DataFrame(iris['data'], columns=iris['feature_names']), pd.Series(iris['target'])


X_resampled = X.iloc[np.random.choice(X.index, X.shape[0])]
    

clf = RandomForestClassifier(n_estimators=20, max_depth=3)
# Train f1 score: 0.9750654148068341
# Test f1 score: 0.957351290684624

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)

print(clf.predict(X_test))
print(np.array(y_test))
print(clf.predict_proba(X_test))

print('Train f1 score:', f1_score(y_train, clf.predict(X_train), average='macro'))
print('Test f1 score:', f1_score(y_test, clf.predict(X_test), average='macro'))
