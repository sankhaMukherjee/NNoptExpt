from logs import logDecorator as lD
from datetime import datetime as dt

import json, os
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

    @lD.log(logBase + '.NNmodel.__init__')
    def __init__(logger, self, inpSize, opSize, layers, activations):
        '''generate a model that will be used for optimization.
        
        This is used for generating a general purpose model. This model
        can later be used as a template for generating more complicated 
        models. Since this is going to be used for generating very simple 
        models, the entire model is composed in Tensorflow, that makes 
        handling things much easier. 

        Instantiating a new model requires that layers are provided. Given 
        layers, a new model is generated with random weights, and zero biases.
        
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

            logger.info('Generating a new model')
            self.inpSize = inpSize
            self.Inp     = tf.placeholder(dtype=tf.float32, shape=inpSize, name='Inp')
            self.Op      = tf.placeholder(dtype=tf.float32, shape=opSize, name='Op')
            
            self.allW    = []
            self.allB    = []

            self.result  = None

            prevSize = inpSize[0]
            for i, l in enumerate(layers):
                tempW = tf.Variable( 0.1*(np.random.rand(l, prevSize) - 0.5), dtype=tf.float32, name='W_{}'.format(i) )
                tempB = tf.Variable( 0, dtype=tf.float32, name='B_{}'.format(i) )

                self.allW.append( tempW )
                self.allB.append( tempB )

                if i == 0:
                    self.result = tf.matmul( tempW, self.Inp ) + tempB
                else:
                    self.result = tf.matmul( tempW, self.result ) + tempB

                prevSize = l

                if activations[i] is not None:
                    self.result = activations[i]( self.result )

            self.err = tf.sqrt(tf.reduce_mean((self.Op - self.result)**2))
            self.modelOK = True

        except Exception as e:
            logger.error('Unable to geberate the required model: {}'.format(str(e)))


        return

    @lD.log(logBase + '.NNmodel.fitAdam')
    def fitAdam(logger, self, X, y, N = 1000, **params):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
            y {[type]} -- [description]
            **params {[type]} -- [description]
        
        Keyword Arguments:
            N {number} -- [description] (default: {1000})
        '''

        if not self.modelOK:
            logger.error('The model is not generated properly. The optimizer will not be evaluated.')
            return

        try:


            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            if self.optimizer is None:
                logger.info('Generating a new optimizer ...')
                self.optimizer = tf.train.AdamOptimizer(name = 'opt', **params).minimize( self.err )

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)
                
                logger.info('Optimization in progress ...')
                if self.currentErrors is None:
                    self.currentErrors = []

                for i in range(N):
                    _, err = sess.run([self.optimizer, self.err ], feed_dict={
                            self.Inp: X, self.Op: y
                        })
                    self.currentErrors.append(err)

                    if i %100 == 0:
                        print(i, err)

                    logger.info('Optimization error at iteration {} = {}.'.format(i, err))
                    
                # Checkpoint the session before you exit ...
                # ------------------------------------------
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))

        except Exception as e:
            logger.error('Unable to optimize the model given the data: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel.getWeights' )
    def getWeights(logger, self):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        '''

        weights = None

        if not self.modelOK:
            logger.error('The model is not generated properly. The optimizer will not be evaluated.')
            return

        saver = tf.train.Saver(tf.trainable_variables())

        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                if self.checkPoint is not None:
                    saver.restore(sess, self.checkPoint)

                weights = sess.run(self.allW + self.allB)
                return weights

        except Exception as e:
            logger.error('Unable to get the weights: {}'.format(str(e)))


        return weights

    @lD.log( logBase + '.NNmodel.setWeights' )
    def setWeights(logger, self, weights):

        try:
            nW = len(self.allW)
            W = weights[:nW]
            B = weights[nW:]

            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                for i in range(len(W)):
                    sess.run(tf.assign( self.allW[i], W[i] ))

                for i in range(len(B)):
                    sess.run(tf.assign( self.allB[i], B[i] ))

                # Checkpoint the session before you exit ...
                # ------------------------------------------
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))

        except Exception as e:
            logger.error('Problem with setting weights to new values ...: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel.predict' )
    def predict(logger, self, X):

        yHat = None


        try:
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                yHat = sess.run(self.result, feed_dict = {self.Inp: X})
                logger.info('Calculated yHat: {}'.format( yHat ))

                # Checkpoint the session before you exit ...
                # ------------------------------------------
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))


                return yHat

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return yHat

    @lD.log( logBase + '.NNmodel.predict' )
    def err(logger, self, X, y):

        errVal = None
        try:
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                logger.info('Calculated errVal: {}'.format( errVal ))

                # Checkpoint the session before you exit ...
                # ------------------------------------------
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))


                return errVal

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVal

