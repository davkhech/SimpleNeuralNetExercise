import random

import numpy as np


def sigmoid(s):
    return 1. / (1. + np.exp(-s))


def sigmoid_derivative(s):
    sigmoid_value = sigmoid(s)
    return sigmoid_value * (1. - sigmoid_value)


class Neuron(object):
    """
    This is a class for a single neuron in our Neural Net.
    In neural_network.py we have a two dimensional python list that represents our Neural Net. Each
    element in this list is of type Neuron.
    """

    def __init__(self,
                 n_inputs,
                 weights=None,
                 transfer_function=sigmoid,
                 transfer_function_derivative=sigmoid_derivative):
        """
        __init__ is the constructor of the Neuron
        param n_inputs: integer, number of inputs to the neuron from the previous layer.
        param weights: a python list. These are the weights corresponding to the inputs. Note
            that len(weights) should be equal to n_inputs+1 because of the "bias" term.
        param transfer_function: in our case this is the sigmoid function.
        transfer_function_derivative: in our case, this is the derivative of the sigmoid.
        """
        self.transfer_function = transfer_function
        if weights is None:
            self.weights = [1e-5 * random.random()
                            for _ in range(n_inputs + 1)]
        else:
            assert n_inputs == len(weights) - 1
            self.weights = weights
        self.derivative_function = transfer_function_derivative

        # Values for the last feed-forward
        self.weighted_sum = None
        self.output = None
        self.inputs = None
        # Values for the last feed-backward
        self.delta = None

    def forward_propagate(self, inputs):
        """
        forward propagate should take the inputs to the neuron and return the output

        param inputs: a python list of length self.n_inputs. 
        return: an integer, which is the output of the neuron
        """
        weighted_sum = self.weights[0]
        for i in range(1, len(self.weights)):
            weighted_sum += self.weights[i] * self.inputs[i - 1]
        self.weighted_sum = weighted_sum
        self.output = self.transfer_function(weighted_sum)
        self.inputs = inputs
        return self.output

    def feed_backwards(self, error):
        """
        feed_backwards should take as an input a single number - error and return a list 
        [error*w for w in the weights]
        feed_backwards also computes and saves self.delta for future use.
        """
        self.delta = self.derivative_function(self.weighted_sum) * error
        errors = []
        for i in range(1, len(self.weights)):
            errors[i - 1] = self.derivative_function(error) - self.weights[i] * self.inputs[i - 1]

    def update_weights(self, learning_rate):
        """
        update_weights takes as argument a float - learning_rate and updates the weight going into the
        current neuron.

        param learning_rate: float
        return: nothing 

        use self.delta, learning_rate and self.inputs to update the weights

        we have written this function for you, understand what this does
        you probably don't need to change this
        """
        self.weights[0] -= learning_rate * self.delta
        for i in range(1, len(self.weights)):
            self.weights[i] -= learning_rate * self.delta * self.inputs[i - 1]
