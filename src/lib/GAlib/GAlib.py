import json, os
import numpy as np

from multiprocessing import cpu_count, Pool

from tqdm     import tqdm
from logs     import logDecorator as lD
from datetime import datetime as dt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

@lD.log( logBase + '.crossover_i' )
def crossover_i(logger, vals):
    '''Crossover the i_th gene
    
    This does crossover for the ith gene. This will be useful
    when we are doing crossover in a multiprocessing environment. 
    This will allow us to significantly improve on execution time
    by parallelizing the computation. 
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    self : {[type]}
        [description]
    i : {[type]}
        [description]
    choices : {[type]}
        [description]
    alphas : {[type]}
        [description]
    X : {[type]}
        [description]
    y : {[type]}
        [description]
    '''

    try:

            i, numElite, w1, w2, a = vals

            logger.info('Crossover value [{}]'.format(i))

            if (i < numElite):
                logger.info('Skipping this due to elitism [{}]'.format(i))
                return i, None

            # Generate a new error
            # -------------------------
            wNew = [ a*m + (1-a)*n  for m, n in zip( w1, w2 ) ]

            return i, wNew

    except Exception as e:
        logger.error('Unable to crossover for gene number: {}'.format( str(e) ))
        return i, None

    return

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
            logger.info('Error info|{}|{}|{}|{}|{}'.format(
                    np.mean(self.currentErr), np.std(self.currentErr),
                    np.min(self.currentErr), np.max(self.currentErr), np.median(self.currentErr), 
                ))


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

            logger.info('Minimum error after mutation: {}'.format( np.array(self.currentErr).min() ))

        except Exception as e:
            logger.error('Unable to do mutation: {}'.format(str(e)))

        return

    @staticmethod
    def generateValues(N, numElite, choices, alphas, population):
        '''Generator for the imap
        '''

        for i in range(N):
            c1, c2 = choices[i]
            a      = alphas[i]
            w1     = population[c1]
            w2     = population[c2]
            yield i, numElite, w1, w2, a

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

            logger.info('------------ AAAA -------> [     0]')
            
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
            
            choices = np.random.choice( range(len(self.currentErr)), size=(len(self.population), 2) , p=normalize )
            alphas  = np.random.random( len(self.currentErr) )

            if self.GAconfig['nPool'] > 0:
                p = Pool(self.GAconfig['nPool'])
            else:
                p = Pool(cpu_count())

            if not self.GAconfig['elitism']['toDo']:
                numElite =  0
            else: 
                numElite = self.GAconfig['elitism']['numElite']

            arguments = self.generateValues(
                len(self.currentErr), numElite, choices, alphas, 
                [p for p in self.populationOld])

            # Generate a new population
            for i, wNew in p.imap(crossover_i, arguments):
                if wNew is None:
                    continue                    
                self.population[i] = wNew

            p.close()

            # Recalculate the errors
            self.err(X, y)

            # Do a selection from the old population if necessary
            for i in range(len(self.population)):
                if self.currentErr[i] > self.currentErrOld[i]:
                    self.currentErr[i] = self.currentErrOld[i]
                    self.population[i] = self.populationOld[i]

            logger.info('Minimum error after crossover: {}'.format( min( self.currentErr ) ))

        except Exception as e:
            logger.error('Unable to do crossover: {}'.format(str(e)))

        return

    @lD.log( logBase + '.predict' )
    def predict(logger, self, X):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
        '''

        prediction = None

        try:
            prediction = self.tempN.predict(X, self.population[0])
        except Exception as e:
            logger.error('Unable to make predictions ... : {}'.format( str(e) ))

        return prediction

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

    @lD.log( logBase + '.loadModel' )
    def loadModel(logger, self, folder):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        folder : {[type]}
            [description]
        '''

        try:
            temp = []
            weightFolders = [ os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('weights_') ]
            weightFolders = sorted(weightFolders)

            for weightFolder in weightFolders:
                files = [ os.path.join(weightFolder, f) for f in os.listdir(weightFolder) if f.endswith('.npy') ]
                weights = [np.load(f) for f in sorted(files)]

                temp.append(weights)

            self.population = temp

            logger.info('New model generated. Note that the errors need to be recalculated ...')


        except Exception as e:
            logger.error('Unable to load the model: {}'.format(str(e) ))

        return

    @lD.log( logBase + '.fit' )
    def fit(logger, self, X, y, folder=None, verbose=True):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        X : {[type]}
            [description]
        y : {[type]}
            [description]
        folder : {[type]}, optional
            [description] (the default is None, which [default_description])
        '''

        if folder is not None:
            print('Loading an earlier model ...')
            self.loadModel(folder)

        self.err(X, y)
        if verbose:
            self.printErrors()

        for i in range(self.GAconfig['numIterations']):
            self.mutate()
            self.crossover(X, y)

            if verbose:
                print('{:8.2f}'.format( i*100.0/self.GAconfig['numIterations'] ),end='-->')
                self.printErrors()

        saveFolder = self.saveModel()
        if saveFolder:
            if verbose:
                print('Model saved at: {}'.format(saveFolder))
                return saveFolder

        return


