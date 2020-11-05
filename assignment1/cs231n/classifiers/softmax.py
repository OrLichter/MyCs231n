from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = X.shape
    C = W.shape[1]
    for i in range(X.shape[0]):
        x = X[i]
        f = x.dot(W)
        f -= np.max(f)
        sum_exp = np.sum(np.exp(f))
        loss += -f[y[i]] + np.log(sum_exp)

        # dW now
        dW.T[y[i]] -= x
        temp_dw = np.repeat(np.reshape(np.exp(f), (1, C)), D, axis=0)
        temp_dw *= np.reshape(np.repeat(x, C), (D, C))
        dW = 1/sum_exp*temp_dw + dW

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    C = W.shape[1]

    f = X.dot(W)
    f = f - np.max(f, axis=1)[:, np.newaxis]
    f = np.exp(f)
    correct_exps = f[range(N), y]
    sums_of_exp = np.sum(f, axis=1)
    loss = np.mean(-1*np.log(correct_exps/sums_of_exp))

    ind = np.zeros_like(f)
    ind[np.arange(N), y] = 1
    dW = X.T.dot(f / sums_of_exp[:, np.newaxis] - ind)



    # loss /= N  ## because we're using np.mean
    loss += 0.5 * reg * np.sum(W * W)

    dW /= N
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

W = np.array([[0, 1, 2], [1, 1, 3]])
X = np.array([[1, 2], [1, 0]])
y = np.array([1, 2])
print(softmax_loss_naive(W,X,y,0)[0])
print(softmax_loss_vectorized(W,X,y,0)[0])