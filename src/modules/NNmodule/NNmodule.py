from logs import logDecorator as lD 
import json
import numpy      as np
import tensorflow as tf

from lib.NNlib import NNmodel

import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.NNmodule.NNmodule'

@lD.log(logBase + '.runModel')
def runModel(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)

    print('We are in the NNmodule')
    inpSize     = (2, None)
    opSize      = (1, None)
    layers      = (5, 8, 1)
    activations = [tf.tanh, tf.tanh, None]
    model       = NNmodel.NNmodel(inpSize, opSize, layers, activations)

    weights = model.getWeights()
    for w in weights:
        print(w)

    model.fitAdam(X, y)

    weights1 = model.getWeights()
    for w, w1 in zip(weights, weights1):
        print(w-w1)



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

    runModel()

    return

