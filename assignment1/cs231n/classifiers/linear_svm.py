from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW.T[j] += X[i]
                dW.T[y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    loss_matrix = X.dot(W)  # X*W
    # -np.repeat(W[y] + 1, W.shape[1]).reshape((X.shape[0], W.shape[1]))
    y_mat = np.zeros((N, C))
    y_mat[range(N), y] = 1
    # |  0   1  ... 0 |
    # |  1   0 .... 0 |
    # |  0   0 ...  1 |
    y_mat = -1*np.multiply(y_mat, loss_matrix)
    # |  0   -w_y_1*x_1  ... 0 |
    # |  -w_y_2*x_2   0 .... 0 |
    # |  0   0 ...  -w_y_2*x_3 |
    y_mat[range(N), y] += 1
    # |  0   -w_y_1*x_1 + 1 ... 0 |
    # |  w_y_2*x_2 + 1   0 .... 0 |
    # |  0   0 ...  w_y_2*x_3 + 1 |
    deltas = np.repeat(np.sum(y_mat, axis=1), C).reshape((N,C))
    # |  -w_y_1*x_1 + 1   -w_y_1*x_1 + 1 ... -w_y_1*x_1 + 1 |
    # |  w_y_2*x_2 + 1   w_y_2*x_2 + 1 .... w_y_2*x_2 + 1 |
    # |  w_y_2*x_2 + 1   w_y_2*x_2 + 1 ...  w_y_2*x_3 + 1 |

    loss_matrix += deltas
    loss_matrix[range(N), y] = 0
    result = np.maximum(np.zeros((N, C)), loss_matrix)
    loss = result.sum()
    loss /= float(N)

    loss += reg * np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    to_use = np.zeros((N,C))
    to_use[result>0] = 1
    to_use_correct_label = -1 * np.sum(to_use, axis=1)
    to_use[range(N),y]= to_use_correct_label
    dW = X.T.dot(to_use)
    dW /= N
    # dW += reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
