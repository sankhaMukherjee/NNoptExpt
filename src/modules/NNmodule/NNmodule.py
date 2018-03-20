from logs      import logDecorator as lD 
from lib.NNlib import NNmodel

import json
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.NNmodule.NNmodule'

@lD.log(logBase + '.checkNNmodel3')
def checkNNmodel(logger):
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
    model1      = NNmodel.NNmodel(inpSize, opSize, layers, activations)

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

    checkNNmodel()

    return

