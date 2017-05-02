import numpy as np
from neuron import Neuron


class NeuralNetwork(object):

    def __init__(self, n_features, *neurons_per_layer):
        """
        param n_features: integer, number of features of the input
        param *neurons_per_layer: arbitrary number of parameters. The len(neurons_per_layer) will be
            the number of hidden layers in the network. Each number here shows the number of neurons in 
            that layer

        __init__ should set self.layers into a two dimensional list, corresponding to the structure of 
            *neurons_per_layer, and should initialize each element in this list to an object of class 
            Neuron with correct number of inputs. The n_features going into the neuron should be n_features
            if the Neuron is in the first layer, or should be the length of the previous layer otherwise.
        """
        self.layers = []
        for i in range(len(neurons_per_layer)):
            n_inputs = n_features if i == 0 else neurons_per_layer[i - 1]
            self.layers.append([Neuron(n_inputs) for _ in range(neurons_per_layer[i])])

        self.n_features = n_features
        self.neurons_per_layer = neurons_per_layer
        self.output = None

    def forward_propagate(self, features):
        """
        param features: a list of length n_features. This is the data from one training image.
        return: a python list (the outputs of the last layer in the network)


        forward_propagate should take the features list, and "forward propagate" it through the many layers
        to finally compute and return the output.

        """
        out = features
        for layer in self.layers:
            out = [neuron.forward_propagate(out) for neuron in layer]
        self.output = out
        return out

    def backward_propagate(self, correct_output_vector):
        """
        param correct_output_vector: a python list of length of the last layer of the network. This
            are the correct outputs which contains many 0s and exactly one 1.
        return: nothing

        backward_propagate takes the correct output, calculates the error and propagates that error,
            from the back of the neural net to the front. To do this, you can pass the net from back to 
            front, layer by layer. For each layer, for i-th Neuron of the layer you call
            .feed_backwards(error[i]), where error is the list of errors coming from the previous
            layer. If you are on the last layer error is simply the difference of the correct_output
            and the output given by the current state of the neural net (you calculate this in forward
            propagate).
            After you call .feed_backwards() you get a list of errors going to l-1 th layer (you
            are on l-th). NOTE that for each neuron in the l-1 th layer, you will get error terms from each
            of the neurons in the l-th layer, and you should sum them up.

            back_propagate saves some data in the individual neurons (this is done when you call
            .feed_backwards() )
            When you call update_weights, this will use those data to finally update the weights.
        """
        out_error = np.array([out - true for true, out in zip(correct_output_vector,
                                                     self.output)])
        error = out_error
        for i in range(len(self.neurons_per_layer) - 1, -1, -1):
            t = np.zeros(len(self.layers[i - 1])) if i > 0 else np.zeros(self.n_features)
            for j in range(len(self.layers[i])):
                ret = self.layers[i][j].feed_backwards(error[j])
                t = t + ret
            error = t

    def update_weights(self, learning_rate):
        """
        param learning_rate: float 

        for each of the neuron in layers[], call .update_weights(learning_rate)
        """
        for layer in self.layers:
            for neuron in layer:
                neuron.update_weights(learning_rate)

    def train(self, X, Y, learning_rate=1e-7, max_iter=1, logging=None):
        """
        param X, Y:
        param learning_rate: float
        param max_iter: integer, number of times ALL data is iterated. In other words,
        how many times we will use the same image to update weights.
        """
        for k in range(max_iter):
            logging.info('Training epoch %d' % k)
            for i in range(len(X)):
                logging.info('Epoch: %d, training example: %d' % (k, i))
                self.forward_propagate(X[i])
                self.backward_propagate(Y[i])
                self.update_weights(learning_rate)
