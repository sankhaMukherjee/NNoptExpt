from logs import logDecorator as lD
import json, os

import numpy as np

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

class GA():
    '''[summary]
    
    [description]
    '''

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, nnClass, initParams, initWeights=None):
        '''Initialize the GA library
        
        Initializer finction for the GA class. This function is going
        to initialize a population. The result of the current state
        is always equal to the best response of all the instances of
        the population. The initial population will not have an error
        score since the individual instances will not be fitted with 
        any values yet.  
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
            nnClass {class for optimization} -- This is the instance of
                the model that we want to train. This model only has the
                following requirements:
                1. It must have a way of initializing a class with the 
                   provided parameters. 
                2. It must have a `getWeights()` methods that will allow the
                   class to get the trainable weights for an instance. The 
                   returned weights must be a list of numpy arrays. 
                3. It must have a `setWeights()` method that will set the 
                   weights obtained by the `getWeights` method. Furthermore,
                   when the weights of one instance is set as the weights 
                   of another, both must behave such that the two instances
                   are equivalent.
                4. It must have a `predict()` method that will allow
                   the function to be evaluated for provided values of
                   input.
            initParams {dict} -- Parameters that will be passed to the 
                class for generating an instance. 
        
        Keyword Arguments:
            initWeights {[type]} -- [description] (default: {None})
        '''

        self.properConfig = False
        self.currentErr   = None

        try:
            self.GAconfig   = json.load( open('../config/GAconfig.json') )
            self.population = [ nnClass(**initParams) for _ in range(self.GAconfig['numChildren'])]

        except Exception as e:
            logger.error('Unable to initialize the GA: {}\n'.format(str(e)))
            return

        return

    @lD.log(logBase + '.predict')
    def predict(logger, self, X):
        '''predict values for all the population
        
        [description]
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
        '''

        try:
            results = [p.predict(X) for p in self.population]
            return results
        except Exception as e:
            logger.error('Unable to make predictions: {}'.format(str(e)))


        return


