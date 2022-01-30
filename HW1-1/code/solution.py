from cv2 import imshow
import numpy as np
import sys
from helper import *
import matplotlib.pyplot as plt


def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE

    for i in range(data.shape[0]):

        imgplot=plt.imshow(data[i])
        plt.show()
        plt.savefig('image_'+str(i)+'.png')

    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    label=y
    x = X[:,0]
    y = X[:,1]



    plt.scatter(x[label==1], y[label==1], marker='+',color='r')
    plt.scatter(x[label==-1], y[label==-1], marker='*',color='b')
    
    
    
    
    
    
    #plt.scatter(X[],y)
    #plt.show()
    #if (save):
    #    plt.savefig("train_features.png")

    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.W=None

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        
        n_samples, n_features=X.shape
        X=np.c_[ X, np.ones(X.shape[0]) ]  
        self.W=np.zeros(n_features+1)
        w=np.zeros(n_features+1)
        
        

        for i in range(self.max_iter):
            for index,row in enumerate(X):
                
                out=np.dot(row,w)
                y_pred=activation_function(out)

                if(y[index]!=y_pred):
                    w+=y[index]*row
                

        self.W=w
        # After implementation, assign your weights w to self as below:
        # self.W = w
        
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

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE

        X=np.c_[ X, np.ones(X.shape[0]) ]    
        
        linear_output=np.dot(X,self.W)

        pred=activation_function(linear_output)
        
        return pred

            

        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: A float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE

        pred_y = self.predict(X)

        return np.mean(y == pred_y)


        ### END YOUR CODE

def activation_function(x):
     return np.where(x>=0, 1, -1)


def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
   
    show_features(X,y)
    
    bias,weights=W[0],W[1:]

    
    line=(-(bias/weights[1])/(bias/weights[0]))*X+(-bias/weights[1])

    
    plt.plot(X,line)
    plt.ylim(-1, 0)
    plt.savefig('model.png')
    plt.show()
    
    
    #plt.plot()



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