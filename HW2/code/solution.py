from sklearn.svm import SVC


def svm_with_diff_c(train_label, train_data, test_label, test_data):
    '''
    Use different value of cost c to train a svm model. Then apply the trained model
    on testing label and data.
    
    The value of cost c you need to try is listing as follow:
    c = [0.01, 0.1, 1, 2, 3, 5]
    Please set kernel to 'linear' and keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE

    cost = [0.01, 0.1, 1, 2, 3, 5]
    print("------------Accuracy with different C values--------------")
    for c in cost:
        model=SVC(C=c,kernel='linear')
        model.fit(train_data, train_label) 
        print("Cost= {0} : Score= {1}".format(c,model.score(test_data, test_label)))
        print("Support vectors: ",model.n_support_.sum())
    
    
    ### END YOUR CODE
    

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    '''
    Use different kernel to train a svm model. Then apply the trained model
    on testing label and data.
    
    The kernel you need to try is listing as follow:
    'linear': linear kernel
    'poly': polynomial kernel
    'rbf': radial basis function kernel
    Please keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE
    kernel={'linear', 'poly', 'rbf'}
    print("------------Accuracy with different Kernels--------------")
    for k in kernel:
        model=SVC(kernel=k)
        model.fit(train_data,train_label)
        print("Kernel={0} : Score= {1}".format(k,model.score(test_data,test_label)))
        print("Support vectors: {0}".format(model.n_support_.sum()))
    ### END YOUR CODE
