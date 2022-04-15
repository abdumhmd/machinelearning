import numpy as np
'''
Homework4: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.eig(): compute the eigen decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_samples, n_features].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()

    
    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_samples, n_components].
        '''
        ### YOUR CODE HERE
        #find mean
        self.means = self.X.mean(axis=0)
        #perform mean normalization.
        self.X = self.X - self.means

        #Compute covariance matrix S.
        S=self.X.T.dot(self.X)

        #find Eigenvector and Eigen Values.
        ev , ec = np.linalg.eig(S)
        idx=ev.argsort()[::-1]
        L=ev[idx]

        #get Principal components
        principal_comps=ec[:,idx]
        eigen_pairs = [(np.abs(L[i]), principal_comps[:, i]) for i in range(len(L))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)

        Up = np.copy(principal_comps)
        Up = np.delete(Up, range(self.n_components,Up.shape[1]), axis=1)
        
        #Perfom transformation.
        Xp=self.X.dot(Up)                
        

        return Up, Xp


        ### END YOUR CODE

    def get_reduced(self):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_any, n_features] or None. 
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_any, n_components].
        '''
        return self.Xp

    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        Return:
        X_re: The reconstructed matrix of shape [n_samples, n_features].
        '''
        ### YOUR CODE HERE

        return Xp.dot(self.Up.T) + self.means

        ### END YOUR CODE


def reconstruct_error(A, B):
    '''
    To compute the reconstruction error.

    Args: 
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return: 
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''
    ### YOUR CODE HERE

    return  np.linalg.norm(A-B)

    ### END YOUR CODE

