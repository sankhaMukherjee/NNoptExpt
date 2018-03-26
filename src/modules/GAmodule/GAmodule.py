from logs import logDecorator as lD 
import json
import numpy      as np
import tensorflow as tf

from datetime import datetime as dt
import matplotlib.pyplot as plt

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
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    # Lets generate a very nonlinear function ... 
    # Rastrigin’s function
    # ----------------------------------------------
    # X = 4*(X - 0.5)
    # y  = (X[0, :]**2 - 10 * np.cos(2 * 3.14 * X[0, :]))
    # y += (X[1, :]**2 - 10 * np.cos(2 * 3.14 * X[1, :]))
    # y += 20
    # y = y.reshape(1, -1)
    # y = y / y.max()

    print(y.max(), y.min())


    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 10, 10, 10, 1), 
        "activations"  : [tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.tanh, None],
    }

    if True:
        print('Generating the GA model ...')    
        ga = GAlib.GA( NNmodel.NNmodel, initParams )

        ga.err(X, y)

        for i in range(10):
            ga.mutate()
            ga.crossover(X, y)
            ga.printErrors()


        saveFolder = ga.saveModel()
        if saveFolder:
            print('Model saved at: {}'.format(saveFolder))
        yHat = ga.predict(X)

        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        plt.plot(y.flatten(), yHat.flatten(), 's', mfc='blue', mec='None', alpha=0.3)
        plt.savefig('../results/img/y_yHat_{}.png'.format(now))
        plt.close('all')

    return

def checkLoading():

    X = np.random.rand(2, 10000)
    # y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)


    folder = '../models/2018-03-20--12-59-44'

    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 10, 10, 10, 1), 
        "activations"  : [tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.tanh, None],
    }

    print('Generating the GA model ...')    
    ga = GAlib.GA( NNmodel.NNmodel, initParams )

    print('Now updating the GA model ...')
    ga.loadModel(folder)
    ga.err(X, y)
    ga.printErrors()

    return

@lD.log(logBase + '.withLoading')
def withLoading(logger):
    '''[summary]
    
    [description]
    
    Decorators:
        lD.log
    
    Arguments:
        logger {[type]} -- [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    # Lets generate a very nonlinear function ... 
    # Rastrigin’s function
    # ----------------------------------------------
    # X = 4*(X - 0.5)
    # y  = (X[0, :]**2 - 10 * np.cos(2 * 3.14 * X[0, :]))
    # y += (X[1, :]**2 - 10 * np.cos(2 * 3.14 * X[1, :]))
    # y += 20
    # y = y.reshape(1, -1)
    # y = y / y.max()

    print(y.max(), y.min())


    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 10, 10, 10, 1), 
        "activations"  : [tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.tanh, None],
    }

    if True:
        print('Generating the GA model ...')    
        ga = GAlib.GA( NNmodel.NNmodel, initParams )

        folder = '../models/2018-03-20--15-35-24'
        if folder is not None:
            print('Loading an earlier model ...')
            ga.loadModel(folder)

        ga.err(X, y)
        ga.printErrors()

        for i in range(10):
            ga.mutate()
            ga.crossover(X, y)
            ga.printErrors()


        saveFolder = ga.saveModel()
        if saveFolder:
            print('Model saved at: {}'.format(saveFolder))
        yHat = ga.predict(X)

        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        plt.plot(y.flatten(), yHat.flatten(), 's', mfc='blue', mec='None', alpha=0.3)
        plt.savefig('../results/img/y_yHat_{}.png'.format(now))
        plt.close('all')

    return

@lD.log(logBase + '.withFitFN')
def withFitFN(logger):
    '''[summary]
    
    [description]
    
    Decorators:
        lD.log
    
    Arguments:
        logger {[type]} -- [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    # Lets generate a very nonlinear function ... 
    # Rastrigin’s function
    # ----------------------------------------------
    # X = 4*(X - 0.5)
    # y  = (X[0, :]**2 - 10 * np.cos(2 * 3.14 * X[0, :]))
    # y += (X[1, :]**2 - 10 * np.cos(2 * 3.14 * X[1, :]))
    # y += 20
    # y = y.reshape(1, -1)
    # y = y / y.max()

    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 10, 10, 10, 1), 
        "activations"  : [tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.tanh, None],
    }

    if True:
        print('Generating the GA model ...')    
        ga = GAlib.GA( NNmodel.NNmodel, initParams )

        # ga.fit(X, y, folder = '../models/2018-03-20--16-13-40')
        ga.fit(X, y, folder = None)

        yHat = ga.predict(X)
        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        plt.plot(y.flatten(), yHat.flatten(), 's', mfc='blue', mec='None', alpha=0.3)
        plt.savefig('../results/img/y_yHat_{}.png'.format(now))
        plt.close('all')

    return

@lD.log(logBase + '.plotResults')
def plotResults(logger, y, yHat, prefix=''):

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    plt.figure(figsize=(4,3))
    plt.axes([0.2, 0.2, 0.79, 0.79])

    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', lw=2, label='expected')
    plt.plot(y.flatten(), yHat.flatten(), 's', mfc='blue', mec='None', alpha=0.1, label='calculated')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.legend()

    plt.savefig('../results/img/{}_y_yHat_{}.png'.format(prefix, now))
    plt.close('all')


    return

@lD.log(logBase + '.plotFirstFew')
def plotFirstFew(logger):
    '''[summary]
    
    [description]
    
    Decorators:
        lD.log
    
    Arguments:
        logger {[type]} -- [description]
    '''

    X = np.random.rand(2, 10000)
    y = (  2*np.sin(X[0, :]) + 3*np.cos(X[1, :]) ).reshape(1, -1)
    y = (  2*X[0, :] + 3*X[1, :] ).reshape(1, -1)

    # Lets generate a very nonlinear function ... 
    # Rastrigin’s function
    # ----------------------------------------------
    # X = 4*(X - 0.5)
    # y  = (X[0, :]**2 - 10 * np.cos(2 * 3.14 * X[0, :]))
    # y += (X[1, :]**2 - 10 * np.cos(2 * 3.14 * X[1, :]))
    # y += 20
    # y = y.reshape(1, -1)
    # y = y / y.max()

    initParams = {
        "inpSize"      : (2, None), 
        "opSize"       : (1, None), 
        "layers"       : (5, 8, 10, 10, 10, 1), 
        "activations"  : [tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.tanh, None],
    }

    
    print('Generating the GA model ...')    
    ga = GAlib.GA( NNmodel.NNmodel, initParams )
    ga.err(X, y)
    yHat = ga.predict(X)

    plotResults(y, yHat, prefix='try100_0000')

    folder = None

    for i in range(10):
        folder = ga.fit(X, y, folder = folder)
        yHat = ga.predict(X)
        plotResults(y, yHat, prefix='try100_{:04}'.format(i))
        

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
    # checkLoading()
    # withFitFN()
    plotFirstFew()

    return

