import json, os
import numpy as np

from tqdm     import tqdm
from logs     import logDecorator as lD
from datetime import datetime as dt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

class GA():

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
            self.tempN        = nnClass(**initParams) # Use this for temp storage
            temp = self.tempN.getWeights()
            for i in tqdm(range(self.GAconfig['numChildren'])):
                self.population.append( [ t + self.GAconfig['initMultiplier']*(np.random.rand()-0.5)  for t in temp] )
            
            self.properConfig = True
        except Exception as e:
            logger.error('Unable to generate the GA class properly: {}'.format(str(e)))

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

            for i in range(len(self.population)):

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
            self.populationOld = [ self.population[i]  for i in sortIndex ]
            self.population    = [ self.population[i]  for i in sortIndex ]
            
            self.currentErrOld = [ self.currentErr[i]  for i in sortIndex ]
            self.currentErr    = [ self.currentErr[i]  for i in sortIndex ]
            
            normalize = np.array(self.currentErr).copy()
            normalize = normalize / normalize.max()
            normalize = 1 - normalize
            normalize = normalize / normalize.sum()

            choices = np.random.choice( range(len(self.currentErr)), size=(100, 2) , p=normalize )
            alphas  = np.random.random( len(self.currentErr) )

            for i in range(len(self.population)):

                logger.info('Crossover value [{}]'.format(i))

                if self.GAconfig['elitism']['toDo'] and (i < self.GAconfig['elitism']['numElite']):
                    logger.info('Skipping this due to elitism [{}]'.format(i))
                    continue

                c1, c2 = choices[i]
                a = alphas[i]

                # Generate a new error
                # -------------------------
                w1 = self.populationOld[c1]
                w2 = self.populationOld[c2]
                wNew = [ a*m + (1-a)*n  for m, n in zip( w1, w2 ) ]

                errVal = self.tempN.errorValW(X, y, wNew)

                # If this is better, update the current neuron
                # There is a potential for problem here, but 
                # we shall neglect it for now. 
                # ---------------------------------------------
                if errVal < self.currentErrOld[i]:
                    self.population[i] = wNew
                    self.currentErr[i] = errVal


        except Exception as e:
            logger.error('Unable to do crossover: {}'.format(str(e)))

        return

    @lD.log( logBase + '.saveModel' )
    def saveModel(logger, self):
        '''save the current model
        
        This function is responsible for saving the
        current parameters of the model. This will 
        include the calculated weights for the entire
        population, along with the other configuration
        information available for the model.

        It will NOT save the NN model. Thus, that will
        need to be generated from scratch.
        '''

        folder = None

        try:
            
            if not self.GAconfig['saveModel']:
                return

            # Generate a folder for the current array
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
            folder = os.path.join(self.GAconfig['modelFolder'], now)
            os.makedirs(folder)

            # Save the config file
            with open(os.path.join(folder, 'config.json'), 'w') as fOut:
                fOut.write(json.dumps(self.GAconfig))

            # Save the weights
            for i, weights in enumerate(self.population):
                fTemp = os.path.join(folder, 'weights_{:010}'.format(i))
                os.makedirs(fTemp)

                for j, w in enumerate(weights):
                    np.save('{}/w_{:010}.npy'.format(fTemp, j), w)

        except Exception as e:
            logger.error('Unable to save the current model: {}'.format(str(e)))

        return folder
