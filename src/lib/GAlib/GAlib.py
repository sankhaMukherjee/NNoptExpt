from logs import logDecorator as lD
import json, os

import numpy as np
from tqdm import tqdm

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
                5. It must have an `errorVal()` method that will allow the current
                   function to find the current error values. The method of
                   calculating the error is left to the descretion of the 
                   neural network.
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
            self.tempN = nnClass(**initParams) # Use this for temp storage
            self.properConfig = True

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

    @lD.log(logBase + '.err')
    def err(logger, self, X, y):
        '''calculate the errors for the population
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
            y {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        '''

        try:
            
            if self.properConfig:
                self.currentErr = []
                for p in tqdm(self.population):
                    self.currentErr.append(p.errorVal(X, y))
            
            return self.currentErr

        except Exception as e:
            logger.error('Unable to generate errors for the population: {}'.format(str(e)))

        return

    @lD.log( logBase + '.printErrors' )
    def printErrors(logger, self):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            self.currentErr = np.array(self.currentErr)
            print('[{:}] | [{:}] | [{:}] '.format( self.currentErr.min(), self.currentErr.mean(), self.currentErr.max() ))


        except Exception as e:
            logger.error('Unable to print the error: {}'.format(str(e)))

        return

    @lD.log( logBase + '.mutate' )
    def mutate(logger, self):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            sortIndex = np.argsort( self.currentErr )
            self.population = [ self.population[i]  for i in sortIndex ]
            self.currentErr = [ self.currentErr[i]  for i in sortIndex ]

            for i in tqdm(range(len(self.population))):

                logger.info('Mutating value [{}]'.format(i))

                if self.GAconfig['elitism']['toDo'] and (i < self.GAconfig['elitism']['numElite']):
                    logger.info('Skipping this due to elitism [{}]'.format(i))
                    continue
                
                logger.info('Updating weights for the new population [{}]'.format(i))
                weights = self.population[i].getWeights()
                newWeights = []
                for w in weights:
                    t = w*( 1 + 2*self.GAconfig['mutation']['multiplier']*(np.random.random(w.shape) - 0.5) )
                    newWeights.append( t )

                self.population[i].setWeights( newWeights )

        except Exception as e:
            logger.error('Unable to do mutation: {}'.format(str(e)))

        return

    @lD.log( logBase + '.crossover' )
    def crossover(logger, self, X, y):
        '''Crossover and selection
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            sortIndex = np.argsort( self.currentErr )
            self.population = [ self.population[i]  for i in sortIndex ]
            self.currentErr = [ self.currentErr[i]  for i in sortIndex ]
            
            normalize = np.array(self.currentErr).copy()
            normalize = normalize / normalize.max()
            normalize = 1 - normalize
            normalize = normalize / normalize.sum()

            choices = np.random.choice( range(len(self.currentErr)), size=(100, 2) , p=normalize )
            alphas  = np.random.random( len(self.currentErr) )

            for i in tqdm(range(len(self.population))):

                logger.info('Crossover value [{}]'.format(i))

                if self.GAconfig['elitism']['toDo'] and (i < self.GAconfig['elitism']['numElite']):
                    logger.info('Skipping this due to elitism [{}]'.format(i))
                    continue

                c1, c2 = choices[i]
                a = alphas[i]

                # Generate a new error
                # -------------------------
                w1 = self.population[c1].getWeights()
                w2 = self.population[c2].getWeights()
                wNew = [ a*m + (1-a)*n  for m, n in zip( w1, w2 ) ]
                self.tempN.setWeights( wNew )
                errVal = self.tempN.errorVal(X, y)

                # If this is better, update the current neuron
                # There is a potential for problem here, but 
                # we shall neglect it for now. 
                # ---------------------------------------------
                if errVal < self.currentErr[i]:
                    self.population[i].setWeights( wNew )
                    self.currentErr[i] = errVal


        except Exception as e:
            logger.error('Unable to do crossover: {}'.format(str(e)))

        return

class GA1():
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
                4. It must have a `predict()` method that will allow
                   the function to be evaluated for provided values of
                   input.
                5. It must have an `errorValW()` method that will allow the
                   the NN errors to be calculated given the weights.
                6. It must have an `errorValWs()` method that will allow the
                   the NN errors to be calculated given a list of weights. The
                   function will return an error for every set of weights
                   provided. This should be significantly faster, since 
                   new tensorflow sessions will not have to be initiated for 
                   every single run ... 
                
            initParams {dict} -- Parameters that will be passed to the 
                class for generating an instance. 
        
        Keyword Arguments:
            initWeights {[type]} -- [description] (default: {None})
        '''

        self.properConfig = False
        self.currentErr   = None

        try:
            self.GAconfig     = json.load( open('../config/GAconfig.json') )
            self.population   = []
            temp = nnClass(**initParams).getWeights()
            for i in tqdm(range(self.GAconfig['numChildren'])):
                self.population.append( [ t + self.GAconfig['initMultiplier']*(np.random.rand()-0.5)  for t in temp] )
            
            self.tempN        = nnClass(**initParams) # Use this for temp storage
            self.properConfig = True

        except Exception as e:
            logger.error('Unable to initialize the GA: {}\n'.format(str(e)))
            return

        return

    @lD.log(logBase + '.err')
    def err(logger, self, X, y):
        '''calculate the errors for the population
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
            y {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        '''

        try:
            
            if self.properConfig:
                self.currentErr = self.tempN.errorValWs(X, y, self.population) 
            
            return self.currentErr

        except Exception as e:
            logger.error('Unable to generate errors for the population: {}'.format(str(e)))

        return

    @lD.log( logBase + '.printErrors' )
    def printErrors(logger, self):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            self.currentErr = np.array(self.currentErr)
            print('[{:}] | [{:}] | [{:}] '.format( self.currentErr.min(), self.currentErr.mean(), self.currentErr.max() ))


        except Exception as e:
            logger.error('Unable to print the error: {}'.format(str(e)))

        return

    @lD.log( logBase + '.mutate' )
    def mutate(logger, self):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            sortIndex = np.argsort( self.currentErr )
            self.population = [ self.population[i]  for i in sortIndex ]
            self.currentErr = [ self.currentErr[i]  for i in sortIndex ]

            for i in tqdm(range(len(self.population))):

                logger.info('Mutating value [{}]'.format(i))

                if self.GAconfig['elitism']['toDo'] and (i < self.GAconfig['elitism']['numElite']):
                    logger.info('Skipping this due to elitism [{}]'.format(i))
                    continue
                
                logger.info('Updating weights for the new population [{}]'.format(i))
                newWeights = []
                for w in self.population[i]:
                    t = w*( 1 + 2*self.GAconfig['mutation']['multiplier']*(np.random.random(w.shape) - 0.5) )
                    newWeights.append( t )

                self.population[i] = newWeights

        except Exception as e:
            logger.error('Unable to do mutation: {}'.format(str(e)))

        return

    @lD.log( logBase + '.crossover' )
    def crossover(logger, self, X, y):
        '''Crossover and selection
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        '''

        try:
            if not self.properConfig:
                logger.error('The GA has not been initialized properly. This step is skipped ...')
                return

            if self.currentErr is None:
                logger.error('Errors have not been calculated yet. This step will be skipped ...')
                return

            sortIndex = np.argsort( self.currentErr )
            self.population = [ self.population[i]  for i in sortIndex ]
            self.currentErr = [ self.currentErr[i]  for i in sortIndex ]
            
            normalize = np.array(self.currentErr).copy()
            normalize = normalize / normalize.max()
            normalize = 1 - normalize
            normalize = normalize / normalize.sum()

            choices = np.random.choice( range(len(self.currentErr)), size=(100, 2) , p=normalize )
            alphas  = np.random.random( len(self.currentErr) )

            for i in tqdm(range(len(self.population))):

                logger.info('Crossover value [{}]'.format(i))

                if self.GAconfig['elitism']['toDo'] and (i < self.GAconfig['elitism']['numElite']):
                    logger.info('Skipping this due to elitism [{}]'.format(i))
                    continue

                c1, c2 = choices[i]
                a = alphas[i]

                # Generate a new error
                # -------------------------
                w1 = self.population[c1]
                w2 = self.population[c2]
                wNew = [ a*m + (1-a)*n  for m, n in zip( w1, w2 ) ]

                errVal = self.tempN.errorValW(X, y, wNew)

                # If this is better, update the current neuron
                # There is a potential for problem here, but 
                # we shall neglect it for now. 
                # ---------------------------------------------
                if errVal < self.currentErr[i]:
                    self.population[i] = wNew
                    self.currentErr[i] = errVal


        except Exception as e:
            logger.error('Unable to do crossover: {}'.format(str(e)))

        return

class GA2():

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, nnClass, initParams):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        nnClass : {[type]}
            [description]
        initParams : {[type]}
            [description]
        '''

        self.properConfig = False
        self.currentErr   = None

        try:

            self.GAconfig     = json.load( open('../config/GAconfig.json') )

            self.population   = []
            temp = nnClass(**initParams).getWeights()
            for i in tqdm(range(self.GAconfig['numChildren'])):
                self.population.append( [ t + self.GAconfig['initMultiplier']*(np.random.rand()-0.5)  for t in temp] )
            
            self.tempN        = nnClass(**initParams) # Use this for temp storage
            self.properConfig = True
        except Exception as e:
            logger.error('Unable to generate the GA class properly: {}'.format(str(e)))

        return
