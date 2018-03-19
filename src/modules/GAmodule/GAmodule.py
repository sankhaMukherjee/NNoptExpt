from logs import logDecorator as lD 
import json
import numpy      as np
import tensorflow as tf

from lib.NNlib import NNmodel
from lib.GAlib import GAlib

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.GAmodule.GAmodule'

@lD.log(logBase + '.runModel3')
def runModel(logger):
    '''[summary]
    
    [description]
    
    Decorators:
        lD.log
    
    Arguments:
        logger {[type]} -- [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    # y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 1), 
        "activations"  : [tf.tanh, tf.tanh, None],
    }

    if True:
        print('Generating the GA model ...')    
        ga = GAlib.GA( NNmodel.NNmodel, initParams )

        ga.err(X, y)

        for i in range(10):
            ga.mutate()
            ga.crossover(X, y)
            ga.printErrors()

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

