import numpy as np
np.random.seed(1234)

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3):
    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
    # Input layer
    network = InputLayer(shape=(None, n_colors, imsize, imsize),
                                        input_var=input_var)
    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                          W=w_init[count], pad='same')
            count += 1
            weights.append(network.W)
        network = MaxPool2DLayer(network, pool_size=(2, 2))
    return network, weights


def test(images, labels):

    num_classes = 4

    print(images.shape)
    print(labels.shape)
    
    # Prepare Theano variables for inputs and targets
    target_var = T.ivector('labels')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # Building the appropriate model
    input_var = T.tensor4('images')
    network, _ = build_cnn(input_var)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                         num_units=256,
                         nonlinearity=lasagne.nonlinearities.rectify)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                         num_units=num_classes,
                         nonlinearity=lasagne.nonlinearities.softmax)

    with np.load('weights_lasg_cnn.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Compile a function computing the test loss and accuracy:
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    err, acc = test_fn(images, labels)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(err))
    print("  test accuracy:\t\t{:.2f} %".format(acc * 100))

if __name__ == '__main__':

    images = np.load('X_test.npy')
    labels = np.load('y_test.npy')

    print( images )
    print( labels )
    
    print('Testing the CNN Model...')
    test(images, labels)
