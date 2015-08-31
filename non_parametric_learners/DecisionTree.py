import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy'):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

        self.depth = 0

    def fit(self, X, y, feature_names=None, pre_prune_type=None, pre_prune_size=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y, pre_prune_type, pre_prune_size)

    def _build_tree(self, X, y, pre_prune_type, pre_prune_size):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode

        Recursively build the decision tree. Return the root node.
        '''

        if pre_prune_type == 'leaf_size':
            leaf_size = pre_prune_size
        else:
            leaf_size = 1

        if pre_prune_type == 'depth':
            tree_depth = pre_prune_size
        else:
            tree_depth = X.shape[0]*X.shape[1]

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1 or len(y) < leaf_size or \
        self.depth > tree_depth:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            self.depth += 1
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1, pre_prune_type, pre_prune_size)
            node.right = self._build_tree(X2, y2, pre_prune_type, pre_prune_size)
        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the entropy of the array y.
        '''

        total = 0
        for cl in np.unique(y):
            prob = np.sum(y == cl) / float(len(y))
            total += prob * math.log(prob)
        return -total

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the gini impurity of the array y.
        '''

        total = 0
        for cl in np.unique(y):
            prob = np.sum(y == cl) / float(len(y))
            total += prob ** 2
        return 1 - total

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and
        value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''

        split_col = X[:, split_index]
        if self.categorical[split_index]:
            A = split_col == split_value
            B = split_col != split_value
        else:
            A = split_col < split_value
            B = split_col >= split_value
        return X[A], y[A], X[B], y[B]

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''

        total = self.impurity_criterion(y)
        for y_split in (y1, y2):
            ent = self.impurity_criterion(y_split)
            total -= len(y_split) * ent / float(len(y))
        return total

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits
        '''

        split_index, split_value, splits = None, None, None
        max_gain = 0
        for i in xrange(X.shape[1]):
            values = np.unique(X[:, i])
            if len(values) < 2:
                continue
            for val in values:
                temp_splits = self._make_split(X, y, i, val)
                X1, y1, X2, y2 = temp_splits
                gain = self._information_gain(y, y1, y2)
                if gain > max_gain:
                    max_gain = gain
                    split_index, split_value = i, val
                    splits = temp_splits
        return split_index, split_value, splits

    # def prune(self, node, X):

    #     if node.left.leaf == False: 
    #         self.prune(node.left, X)
    #     if node.right.leaf == False:
    #         self.prune(node.right, X)
    #     if node.left.leaf == True and node.right.leaf == True:
    #         no_merge = node.name
    #         y_pred_no_merge = self.predict(X)
    #         error_no_merge = self._entropy(y_pred_no_merge)

    #         node.classes = node.left.classes + node.right.classes 
    #         node.name = node.classes.most_common(1)[0][0]
            
    #         node.leaf = True
    #         node.left.leaf = False
    #         node.right.leaf = False
    #         y_pred_merge = self.predict(X)
    #         error_merge = self._entropy(y_pred_merge)
    #         if error_merge < error_no_merge:
    #             self.prune(self.root, X)
    #         else:
    #             node.name = no_merge
    #             node.leaf = False
    #             node.left.leaf = True
    #             node.right.leaf = True

    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array

        Return an array of predictions for the feature matrix X.
        '''

        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)
