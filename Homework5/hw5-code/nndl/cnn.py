import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    # 1st Layer
    C, H, W = input_dim
    
    stride = 1
    pad = (filter_size - 1) / 2
    
    out_conv_H = int(1 + (H + 2 * pad - filter_size) / stride)
    out_conv_W = int(1 + (W + 2 * pad - filter_size) / stride)
    
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size])
    self.params['b1'] = np.zeros([num_filters])
    
    # 2nd Layer
    pool_stride = 2
    pool_filter_size = 2
    
    out_pool_H = int(1 + (out_conv_H - pool_filter_size) / pool_stride)
    out_pool_W = int(1 + (out_conv_W - pool_filter_size) / pool_stride)
    
    self.params['W2'] = np.random.normal(0, weight_scale, [num_filters*out_pool_H*out_pool_W, hidden_dim])
    self.params['b2'] = np.zeros([hidden_dim])
    
    # 3rd Layer
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros([num_classes])

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    conv_outs, conv_caches = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    fc_outs, fc_caches = affine_relu_forward(conv_outs, W2, b2)
    scores, caches = affine_forward(fc_outs, W3, b3)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dx = softmax_loss(scores, y)
    reg_loss_sum = 0
    reg_loss_sum += (np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2)

    loss += 0.5 * self.reg * reg_loss_sum

    dx3, dW3, db3 = affine_backward(dx, caches)
    dx2, dW2, db2 = affine_relu_backward(dx3, fc_caches)
    dx1, dW1, db1 = conv_relu_pool_backward(dx2, conv_caches)

    grads['W1'] = dW1 + self.reg * W1
    grads['b1'] = db1
    grads['W2'] = dW2 + self.reg * W2
    grads['b2'] = db2
    grads['W3'] = dW3 + self.reg * W3
    grads['b3'] = db3 

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
