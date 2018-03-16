from logs import logDecorator as lD 
import json
import numpy      as np
import tensorflow as tf

from lib.NNlib import NNmodel

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.NNmodule.NNmodule'

@lD.log(logBase + '.checkNNmodel')
def checkNNmodel(logger):
    '''Check whether the NNmodel us working find
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    # y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    print('We are in the NNmodule')
    inpSize     = (2, None)
    opSize      = (1, None)
    layers      = (5, 8, 1)
    activations = [tf.tanh, tf.tanh, None]
    model1      = NNmodel.NNmodel(inpSize, opSize, layers, activations)
    model2      = NNmodel.NNmodel(inpSize, opSize, layers, activations)

    # Fitting the model.
    print('Fitting the model here ...')
    model1.fitAdam(X, y, N=10000)

    print('Setting weights in the next model ...')
    weights1 = model1.getWeights()
    model2.setWeights( weights1 )

    print('Making predictions with the next model ')
    yHat = model2.predict(X)
    
    print('plotting the data')
    plt.figure()
    plt.plot(y.ravel(), yHat.ravel(), '+')
    plt.savefig('../results/img/compare.png')
    plt.close('all')


    return

@lD.log(logBase + '.checkNNmodel1')
def checkNNmodel1(logger):
    '''Check whether the NNmodel is working find
    
    [description]

    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    # y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    print('We are in the NNmodule')
    inpSize     = (2, None)
    opSize      = (1, None)
    layers      = (5, 8, 1)
    activations = [tf.tanh, tf.tanh, None]
    model1      = NNmodel.NNmodel1(inpSize, opSize, layers, activations)
    model2      = NNmodel.NNmodel1(inpSize, opSize, layers, activations)

    # Fitting the model.
    print('Fitting the model here ...')
    weights = model1.fitAdam(X, y, N=10000)

    print('Finding the error values for the calculated weights ...')
    errVal = model1.errorValW(X, y, weights)
    print('Calculated ErrorVal = {}'.format(errVal))

    print('Transferring weights to model2 ...')
    errVal = model2.errorValW(X, y, weights)
    print('Calculated ErrorVal = {}'.format(errVal))

    weightsList = [ weights ] * 100

    print('Transferring weights to model2 ...')
    errVals = model2.errorValWs(X, y, weightsList)
    print('Calculated ErrorVal = {}'.format(errVals))

    return

@lD.log(logBase + '.checkNNmodel3')
def checkNNmodel3(logger):
    '''Check whether the NNmodel is working find
    
    [description]

    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    # X = np.random.rand(2, 10000)
    # y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    # y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    print('We are in the NNmodule')
    inpSize     = (2, None)
    opSize      = (1, None)
    layers      = (5, 8, 1)
    activations = [tf.tanh, tf.tanh, None]
    model1      = NNmodel.NNmodel3(inpSize, opSize, layers, activations)

    with tf.Session() as sess:
        sess.run(model1.init)
        weights = model1.getWeights()
        print(weights)
        weights = [w+5 for w in weights]
        
    with tf.Session() as sess:
        sess.run(model1.init)

        model1.setWeights(weights, sess)
        weights1 = model1.getWeights(sess)
        
    print(weights1)
    
    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    # checkNNmodel1()
    # checkNNmodel1()
    checkNNmodel3()

    return

