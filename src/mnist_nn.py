import argparse
import gzip
import struct

import matplotlib.pyplot as plt
import numpy as np

from time import time
from neural_network import NeuralNetwork

"""
this file is given to you to use as you want.
Your objective is to get as small error on the test function as possible.

Feel free to experiment with different parameters. You can change structure of
the NeuralNetwork, you can make the program train on a smaller data-set(see below)
to get faster computation times when you are just trying things...
"""


def test(network, num_data=10000):
    errors = 0
    i = 0
    for img, true_label in zip(test_data[:num_data], test_labels[:num_data]):
        i += 1
        out_v = network.forward_propagate(img.reshape(-1))
        errors += 0 if np.argmax(out_v) == true_label else 1
        print(i, '{:.2f}% error rate'.format(100. * errors / i))


def show_10_neurons(network):
    _10_neurons(network, show=True)


def save_10_neurons(network, path):
    _10_neurons(network, savefig=path)


def _10_neurons(network, show=False, savefig=None):
    plt.clf()
    plot_rows, plot_columns = 2, 5
    f, axarr = plt.subplots(plot_rows, plot_columns)
    for i in range(plot_rows):
        for j in range(plot_columns):
            n = network.layers[0][plot_columns * i + j]
            weight_img = np.reshape(n.weights[1:], (rows, columns))
            axarr[i, j].imshow(weight_img, cmap='Greys')
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-train-data',
                        default='../train-images-idx3-ubyte.gz',  # noqa
                        help='Path to train-images-idx3-ubyte.gz file '
                             'downloaded from http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--mnist-train-labels',
                        default='../train-labels-idx1-ubyte.gz',  # noqa
                        help='Path to train-labels-idx1-ubyte.gz file '
                             'downloaded from http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--mnist-test-data',
                        default='../t10k-images-idx3-ubyte.gz',
                        help='Path to t10k-images-idx3-ubyte.gz file '
                             'downloaded from http://yann.lecun.com/exdb/mnist/')  # noqa
    parser.add_argument('--mnist-test-labels',
                        default='../t10k-labels-idx1-ubyte.gz',
                        help='Path to t10k-labels-idx1-ubyte.gz file '
                             'downloaded from http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--positive-label', type=int, choices=list(range(10)),
                        default=9)
    parser.add_argument('--negative-label', type=int, choices=list(range(10)),
                        default=4)
    parser.add_argument('--limit-to', nargs='*',
                        help='Limit to the specified files.')
    args = parser.parse_args(*argument_array)
    return args


if __name__ == '__main__':
    args = parse_args()

    # Read labels file into labels
    with gzip.open(args.mnist_train_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        all_labels = struct.unpack('>60000B', in_gzip.read(60000))

    # Read data file into numpy matrices
    with gzip.open(args.mnist_train_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        all_data = [np.reshape(struct.unpack('>{}B'.format(rows * columns),
                                             in_gzip.read(rows * columns)),
                               (rows, columns))
                    for _ in range(60000)]

    # Read labels file into labels
    with gzip.open(args.mnist_test_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        test_labels = struct.unpack('>10000B', in_gzip.read(10000))

    # Read data file into numpy matrices
    with gzip.open(args.mnist_test_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        test_data = [np.reshape(struct.unpack('>{}B'.format(rows * columns),
                                              in_gzip.read(rows * columns)),
                                (rows, columns))
                     for _ in range(10000)]

    vector_data = np.array([img.reshape(-1) for img in all_data])

    vector_labels = np.array([np.zeros(10) for _ in all_labels])
    for v, y in zip(vector_labels, all_labels):
        v[y] = 1

    tStart = time()
    network = NeuralNetwork(784, 60, 30, 10)

    print 'Training started at', tStart
    training_set_size = 6000
    network.train(vector_data[:training_set_size],
                            vector_labels[:training_set_size], max_iter=1)

    print 'Training finished at', time()
    print 'It took ', time() - tStart

    test(network)
    # save_10_neurons(network, path="pic")
