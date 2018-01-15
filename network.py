# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

import matplotlib.pyplot as plt

# ===================
# docs
# eclipse - pydev interactive - https://stackoverflow.com/questions/13440956/interactive-matplotlib-through-eclipse-pydev

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # start at size[1], skipping the input layer
        # [784, 10, 10]-->[..,10,10]
        self.biases = [np.random.randn(y, 1)  for y in sizes[1:]]
        
        # zip(sizes[:-1], size[1:])
        # zip([784,10,..], [..,10,10]) = (784=x,10=y) (10=x,10=y) 
        # 1 array of 10 slots, each slot holds an array of 784 slots
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

        #make a copy of the original weights array
        np.copyto(self.weights0,self.weights)
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using back-propagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # create bias and weight arrays set to 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            # calculate new biases and weights as...
            # b()=b(0) + delta_b
            # w()=w(0) + delta_w
            # accumulate the biases from each training data pair (x,y)
            # accumulate the weights  from each training data pair (x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #plot_multi_array(self.weights)            
        #plot_multi_array(nabla_w)            
        # apply the new weight to the old weight, divide by batch size, multiply by learning rate eta
        # self.weights=(10,784), (10,10)= w(x,y) where x=rows and y=columns
        # nabla_w     =(10,784), (10,10)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        #plot_multi_array(self.weights)            

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        # x is training data; starting from the left...x is the activation input
        # calculate the z vector for each layer, left to right
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # activations[ 1-2,  2-3 ]
        # 1-2 layer is input - hidden layer
        # 2-3 layer is hidden to output layer
        # 
        # backward pass
        # from right to left....
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # range(2,3) 2
        # range(4)   0,1,2,3
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

          
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# local

def array_testing():
    # slice(start, stop, increment)
    a=[1,2,3,4,5,6,7,8]
    print("a[1:]  output from index=1                {}".format(a[1:]))
    print("a[1:4] output from index(1,3)             {}".format(a[1:4]))
    print("a[::-1] output in reverse                 {}".format(a[::-1]))
    
    # [:-1] ==> start from the beginning...but go to the end index - 1.
    print("a[:-1]  output in order, ignore last item {}".format(a[:-1]))
    
    sizes=[784,10,10]
    weights = [np.random.randn(y, x) 
               for x, y in zip(sizes[:-1], sizes[1:])]
    
    plot_array(weights[0])
    plot_multi_array(weights)

# https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect
def plot_array(multi_dim_array_):
    plt.imshow(multi_dim_array_,aspect='auto')
    plt.show()    

def plot_multi_array(multi_multi_dim_array_):
    for w in multi_multi_dim_array_:
        plt.imshow(w,aspect='auto')
        plt.show()

def plot_tester():
    H = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    plt.imshow(H)
    plt.show()
            
#array_testing()
#exit(0)

    
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 10, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
