import math
from re import L
import numpy as np
import copy

THETA = 0.5


class Tree(object):

    def __init__(self, feature=None, ys=[], left=None, right=None):
        self.feature = feature
        self.ys = ys
        self.left = left
        self.right = right

    @property
    def size(self):
        size = 1
        if type(self.left) == int:
            size += 1
        else:
            size += self.left.size
        if type(self.right) == int:
            size += 1
        else:
            size += self.right.size
        return size

    @property
    def depth(self):
        left_depth = 1 if type(self.left) == int else self.left.depth
        right_depth = 1 if type(self.right) == int else self.right.depth
        return max(left_depth, right_depth)+1


def entropy(data):
    """Compute entropy of data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        entropy of data (float)
    """
    ### YOUR CODE HERE
    
    #get y's from the data
    y=np.array([y for _,y in data])

    #get counts for each class
    unique,counts=np.unique(y,return_counts=True)
    cls=dict(zip(unique,counts))
    
    #calculate the total number of labels
    total=counts.sum()
    print(cls)
    #entropy is 1 if there are no y's
    if total==0:
        return 1
    
    #Check if there are not instance for class 1
    if(1 not in cls):
        entropy = - cls[0]/total * math.log2(cls[0]/total) 
    
    #Check if there are not instance for class 0
    elif (0 not in cls):
        entropy =  - cls[1]/total * math.log2(cls[1]/total)
    else:    
        #calculate entropy
        entropy = - cls[0]/total * math.log2(cls[0]/total) - cls[1]/total * math.log2(cls[1]/total)
    
    return entropy
    
    ### END YOUR CODE


def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data

    Returns:
        gain of splitting data by feature
    """
    ### YOUR CODE HERE

    # please call entropy to compute entropy

    #splits
    features1=[]
    features2=[]

    

    for x,y in data:
       
        #first split
        if x[feature]==0:
            features1.append((x,y))
        #other split
        else:
            features2.append((x,y))
    
    #calculate gain
    gain=-(len(features1)*entropy(features1)+len(features2)*entropy(features2))/(len(data))

    return -gain

    ### END YOUR CODE


def get_best_feature(data):
    """Find the best feature to split data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        index of feature to split data
    """
    ### YOUR CODE HERE

    # please call gain to compute gain

    #initialize maximum gain as infinity
    max_gain = float("inf")

    #calculate gain for each feature and find the one with highest gain.
    for i in range(len(data[0][0])):
      g = gain(data,i)
      if max_gain > g:
        max_gain = g
        best_feature = i
    return best_feature

    ### END YOUR CODE


def build_tree(data):
    ys = {}
    for x, y in data:
        ys[y] = ys.get(y, 0) + 1
    if len(ys) == 1:
        return list(ys)[0]
    feature = get_best_feature(data)

    ### YOUR CODE HERE

    # please split your data with feature and build two sub-trees
    # by calling build_tree recursively

    # left_tree = build_tree(...)
    # right_tree = build_tree(...)

    #return the 
    if len(set(ys)) == 1:
      return ys
        

    
    left=[]
    right=[]
    for x, y in data:
        if x[feature] < THETA:
          left.append((x,y))
        else:
          right.append((x,y))

    # please split your data with feature and build two sub-trees
    # by calling build_tree recursively

    left_tree = build_tree(left)
    right_tree = build_tree(right)

    # Use THETA to split the continous feature

    ### END YOUR CODE
    return Tree(feature, ys, left_tree, right_tree)


def test_entry(tree, entry):
    x, y = entry
    if type(tree) == int:
        return tree, y
    if x[tree.feature] < THETA:
        return test_entry(tree.left, entry)
    else:
        return test_entry(tree.right, entry)


def test_data(tree, data):
    count = 0
    for d in data:
        y_hat, y = test_entry(tree, d)
        count += (y_hat == y)
    return round(count/float(len(data)), 4)


def prune_tree(tree, data):
    """Find the best feature to split data.

    Args:
        tree: a decision tree to prune
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        a pruned tree
    """
    ### YOUR CODE HERE

    # please call test_data to obtain validation error
    # please call prune_tree recursively for pruning tree
    ptree=copy.deepcopy(tree)
    ptree.left=0
    ptree.right=1
    acc1=test_data(tree,data)
    acc2=test_data(ptree,data)
    if acc1<acc2 and ptree.depth!=1:
      prune_tree(ptree,data)
    return ptree
    ### END YOUR CODE
