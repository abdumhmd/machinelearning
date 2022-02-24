import numpy as np
import sys
from helper import *
def show_images(data):

"""Show the input images and save them.
Args:
data: A stack of two images from train data with shape (2, 16, 16).
Each of the image has the shape (16, 16)
Returns:
Do not return any arguments. Save the plots to ’image_1.*’ and ’image_2.*’ and
include them in your report
"""
### YOUR CODE HERE
for i, img in enumerate(data):
plt.clf()
fig = plt.figure()
plt.imshow(img)
fig.savefig(’image_ % d’% (i+1))
### END YOUR CODE
def show_features(X, y, save=True):

"""Plot a 2-D scatter plot in the feature space and save it.
Args:
X: An array of shape [n_samples, n_features].
y: An array of shape [n_samples,]. Only contains 1 or -1.
save: Boolean. The function will save the figure only if save is True.
Returns:
Do not return any arguments. Save the plot to ’train_features.*’ and include it
in your report.
"""
### YOUR CODE HERE
n, _ = X.shape
fig = plt.figure()
for i in range(n):
if y[i] == 1:
plt.plot(X[i, 0], X[i, 1], ’r *’)
else:
plt.plot(X[i, 0], X[i, 1], ’b +’)
if save:
fig.savefig(’train_features’)
### END YOUR CODE
5
class Perceptron(object):
def __init__(self, max_iter):


self.max_iter = max_iter
def fit(self, X, y):

"""Train perceptron model on data (X,y).
Args:
X: An array of shape [n_samples, n_features].
y: An array of shape [n_samples,]. Only contains 1 or -1.
Returns:
self: Returns an instance of self.
"""
### YOUR CODE HERE
w = np.zeros([3])
for i in range(self.max_iter):
misclass = []
for idx in range(len(X)):
if not np.sign(np.dot(w, X[idx])) == y[idx]:
misclass += [idx]
if len(misclass) == 0:
print(’All correct in training set.’)
break
else:
idx = np.random.choice(misclass)
w = w + y[idx] * X[idx]
print(’Fitting finished in %d iterations.’ % (i+1))
self.W = w
### END YOUR CODE
return self
def get_params(self):

"""Get parameters for this perceptron model.
Returns:
W: An array of shape [n_features,].
"""
if self.W is None:
print("Run fit first!")
sys.exit(-1)
return self.W
6
def predict(self, X):

"""Predict class labels for samples in X.
Args:
X: An array of shape [n_samples, n_features].
Returns:
preds: An array of shape [n_samples,]. Only contains 1 or -1.
"""
### YOUR CODE HERE
preds = np.sign(np.dot(X, self.W.transpose()))
return preds
### END YOUR CODE
def score(self, X, y):

"""Returns the mean accuracy on the given test data and labels.
Args:
X: An array of shape [n_samples, n_features].
y: An array of shape [n_samples,]. Only contains 1 or -1.
Returns:
score: An float. Mean accuracy of self.predict(X) wrt. y.
"""
### YOUR CODE HERE
preds = self.predict(X)
score = sum(preds == y)/float(len(X))
return score
### END YOUR CODE
def show_result(X, y, W):

"""Plot the linear model after training.
You can call show_features with ’save’ being False for convenience.
Args:
X: An array of shape [n_samples, 2].
y: An array of shape [n_samples,]. Only contains 1 or -1.
W: An array of shape [n_features,].
Returns:
Do not return any arguments. Save the plot to ’test_result.*’ and include it
in your report.
"""
### YOUR CODE HERE
show_features(X, y, save=False)
7
x = np.linspace(-1, 0, 2)
y = - W[1]/W[2]*x - W[0]/W[2]
plt.plot(x, y, linewidth=2.5, linestyle="-")
# plt.axis([-1, 0.1, -1, 0.5])
plt.savefig(’test_result’)
### END YOUR CODE


def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
model = Perceptron(max_iter)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
W = model.get_params()
# test perceptron model
test_acc = model.score(X_test, y_test)
return W, train_acc, test_acc
