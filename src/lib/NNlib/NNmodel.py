from logs     import logDecorator as lD

import json
import numpy       as np 
import tensorflow  as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.NNlib.NNmodel'

# Saving variablez
# https://stackoverflow.com/questions/45179556/key-variable-name-not-found-in-checkpoint-tensorflow

class NNmodel():
    '''[summary]
    
    [description]
    '''

    @lD.log(logBase + '.NNmodel3.__init__')
    def __init__(logger, self, inpSize, opSize, layers, activations):
        '''generate a model that will be used for optimization.
        
        Similar to the previous function, except that it totally does away
        with saving the state at any time. This will significantly save 
        on the time required to save the state and will speed things up. 
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- Inserted by the decorator
            self {instance of class} -- inserted by the class
            inpSize {[type]} -- [description]
            layers {[type]} -- [description]
            activations {[type]} -- [description]
        '''

        self.modelOK = False
        self.checkPoint = None
        self.optimizer = None

        self.fitted = False
        self.currentErrors = None

        try:

            self.NNmodelConfig = json.load(open('../config/NNmodelConfig.json'))

            logger.info('Generating a new model')
            self.inpSize = inpSize
            
            self.Inp     = tf.placeholder(dtype=tf.float32, shape=inpSize, name='Inp')
            self.Op      = tf.placeholder(dtype=tf.float32, shape=opSize, name='Op')
            
            self.allW         = []
            self.allWPH       = []
            self.allAssignW   = []

            self.allB         = []
            self.allBPH       = []
            self.allAssignB   = []


            self.result  = None

            prevSize = inpSize[0]
            for i, l in enumerate(layers):

                tempW       = tf.Variable( 0.1*(np.random.rand(l, prevSize) - 0.5), dtype=tf.float32, name='W_{}'.format(i) )
                tempWPH     = tf.placeholder(dtype=tf.float32, shape=(l, prevSize), name='PHW_{}'.format(i))
                tempAssignW = tf.assign(tempW, tempWPH, name='AssignW_{}'.format(i))

                tempB       = tf.Variable( 0, dtype=tf.float32, name='B_{}'.format(i) )
                tempBPH     = tf.placeholder(dtype=tf.float32, shape=tuple(), name='PHB_{}'.format(i))
                tempAssignB = tf.assign(tempB, tempBPH, name='AssignB_{}'.format(i))

                self.allW.append( tempW )
                self.allWPH.append( tempWPH )
                self.allAssignW.append( tempAssignW )

                self.allB.append( tempB )
                self.allBPH.append( tempBPH )
                self.allAssignB.append( tempAssignB )


                if i == 0:
                    self.result = tf.matmul( tempW, self.Inp ) + tempB
                else:
                    self.result = tf.matmul( tempW, self.result ) + tempB

                prevSize = l

                if activations[i] is not None:
                    self.result = activations[i]( self.result )

            self.err = tf.sqrt(tf.reduce_mean((self.Op - self.result)**2))

            self.init    = tf.global_variables_initializer()
            self.modelOK = True

        except Exception as e:
            logger.error('Unable to geberate the required model: {}'.format(str(e)))


        return

    @lD.log(logBase + '.NNmodel3.getWeights')
    def getWeights(logger, self, sess=None):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        '''

        weights = None
        try:

            if sess is not None:
                weights = sess.run( self.allW + self.allB  )
                return weights

            with tf.Session() as sess:
                sess.run(self.init)
                weights = sess.run( self.allW + self.allB )
                return weights

        except Exception as e:
            logger.error('Unable to get the weights: {}'.format(str(e)))


        return weights

    @lD.log(logBase + '.NNmodel3.setWeights')
    def setWeights(logger, self, weights, sess):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        weights : {[type]}
            [description]
        sess : {[type]}
            [description]
        '''

        try:
            Nw = len(self.allW)
            Ws = weights[:Nw]
            Bs = weights[Nw:]

            for i, (w, mW) in enumerate(zip(Ws, self.allAssignW)):
                sess.run(mW, feed_dict={ 'PHW_{}:0'.format(i) : w } )

            for i, (b, mW) in enumerate(zip(Bs, self.allAssignB)):
                sess.run(mW, feed_dict={ 'PHB_{}:0'.format(i) : b } )

        except Exception as e:
            logger.error('Unable to set the weights ...: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel3.errorValW' )
    def errorValW(logger, self, X, y, weights):

        errVal = None

        try:
            
            with tf.Session() as sess:
                sess.run(self.init)
                self.setWeights(weights, sess)

                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                logger.info('Calculated errVal: {}'.format( errVal ))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVal

    @lD.log( logBase + '.NNmodel3.predict' )
    def predict(logger, self, X, weights):

        yHat = None

        try:
            
            with tf.Session() as sess:
                sess.run(self.init)
                self.setWeights(weights, sess)

                yHat = sess.run(self.result, feed_dict = {self.Inp: X})
                logger.info('Calculated prediction: {}'.format( yHat ))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return yHat

    @lD.log( logBase + '.NNmodel3.errorValWs' )
    def errorValWs(logger, self, X, y, weightsList):

        errVals = []

        try:
            
            with tf.Session() as sess:
                sess.run(self.init)

                for weights in weightsList:
                    self.setWeights(weights, sess)
                    errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                    errVals.append(errVal)

                    logger.info('Calculated errVal: {}'.format( errVal ))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVals

