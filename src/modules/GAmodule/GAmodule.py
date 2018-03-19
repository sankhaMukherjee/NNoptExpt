from logs import logDecorator as lD 
import json
import numpy      as np
import tensorflow as tf

import matplotlib.pyplot as plt
from datetime import datetime as dt

from lib.NNlib import NNmodel
from lib.GAlib import GAlib

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.GAmodule.GAmodule'

@lD.log(logBase + '.runModel')
def runModel(logger):
    '''[summary]
    
    [description]
    
    Decorators:
        lD.log
    
    Arguments:
        logger {[type]} -- [description]
    '''

    X = np.random.rand(2, 10000)
    # y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 1), 
        "activations"  : [tf.tanh, tf.tanh, None],
    }

    print('Generating the GA model ...')    
    initClass = GAlib.GA1( NNmodel.NNmodel3, initParams)

    # print('Generating the predictions')
    # yHats = initClass.predict(X)
    # print(yHats)

    print('Calculating the errors for the current population ...')
    errors = initClass.err(X, y)
    for e in errors:
        print('{}'.format(e))

    print('Doing mutation')
    initClass.mutate()

    print('Calculating the errors for the current population again ...')
    errors = initClass.err(X, y)
    for e in errors:
        print('{}'.format(e))

    print('Before crossover ...')
    initClass.printErrors()

    # for i in range( 2 ):
    #     print('Performing crossover ...')
    #     initClass.crossover(X, y)

    #     print('After crossover ...')
    #     initClass.printErrors()

    # for p in initClass.population:
    #     weights = p.getWeights()
    #     for w in weights:
    #         print(w)

    

    return

@lD.log(logBase + '.runModel3')
def runModel3(logger):
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
        ga = GAlib.GA2( NNmodel.NNmodel3, initParams )

        ga.err(X, y)

        errors = []
        errorsM = []
        for i in range(500):
            ga.mutate()
            ga.crossover(X, y)
            ga.printErrors()
            errors.append(min(ga.currentErr))
            errorsM.append(np.mean(ga.currentErr))


    plt.plot(errors, label='min')
    # plt.plot(errorsM, label='mean')
    plt.yscale('log')
    plt.legend()
    plt.savefig( dt.now().strftime('../results/img/%Y-%m-%d--%H-%M-%S_errors.png') )
    plt.close('all')

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

    # runModel()
    runModel3()

    return

