
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
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dsoftmax(x):
    s = softmax(x)
    si_sj = - s * s.reshape(3, 1)
    return np.diag(s) + si_sj

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