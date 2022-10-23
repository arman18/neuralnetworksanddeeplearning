
# https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4

import numpy as np

# ----------------------- sigmoid --------------------------------

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# ----------------------- Tanh --------------------------------
def tanh(x):
    
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def dtanh(x):
    return 1-tanh(x)**2

# ----------------------- ReLu --------------------------------
def relu(x):
  return np.maximum(0.,x)

def drelu(x):
  return np.greater(x, 0.).astype(np.float32)

# ----------------------- softmax --------------------------------
def softmax(x):
    """ applies softmax to an input x"""
    max_x = np.amax(x, 1).reshape(x.shape[0],1) # Get the row-wise maximum
    e_x = np.exp(x - max_x ) # For stability
    return e_x / e_x.sum(axis=1, keepdims=True) 

def dsoftmax(x):
    s = softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    temp1 = np.einsum('ij,jk->ijk',s,a)
    temp2 = np.einsum('ij,ik->ijk',s,s)
    return temp1-temp2

# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------
# ----------------------- Tanh --------------------------------