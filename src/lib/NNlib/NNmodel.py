from logs     import logDecorator as lD
from time     import time
from tqdm     import tqdm
from datetime import datetime as dt

import json, os, shutil
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

            self.NNmodelConfig = json.load(open('../config/NNmodelConfig.json'))

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
                # Checkpoint the session before you exit ...
                # ------------------------------------------
                if (self.checkPoint is not None) and (not self.NNmodelConfig['keepChkPts']):
                    # delete the old checkpoints first
                    folder = os.path.dirname(self.checkPoint)
                    shutil.rmtree( folder )

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

            t0 = time()
            with tf.Session() as sess:

                t1 = time()
                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                t2 = time()
                for i in range(len(W)):
                    sess.run(tf.assign( self.allW[i], W[i] ))

                for i in range(len(B)):
                    sess.run(tf.assign( self.allB[i], B[i] ))

                t3 = time()
                # Checkpoint the session before you exit ...
                # ------------------------------------------
                if (self.checkPoint is not None) and (not self.NNmodelConfig['keepChkPts']):
                    # delete the old checkpoints first
                    folder = os.path.dirname(self.checkPoint)
                    shutil.rmtree( folder )

                t4 = time()
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))

                t5 = time()

            toPrint = []
            toPrint.append('--------------------[{}]-----------------'.format(now))
            toPrint.append('[{}] [setWeights] Time to start a session     : {:.3}'.format( now,  t1 - t0))
            toPrint.append('[{}] [setWeights] Time to read a checkpoint   : {:.3}'.format( now,  t2 - t1))
            toPrint.append('[{}] [setWeights] Time to calculate a value   : {:.3}'.format( now,  t3 - t2))
            toPrint.append('[{}] [setWeights] Time to delete old folder   : {:.3}'.format( now,  t4 - t3))
            toPrint.append('[{}] [setWeights] Time to save new checkpoint : {:.3}'.format( now,  t5 - t4))
            if self.NNmodelConfig['showTimes']:
                print('\n'.join(toPrint))
            logger.info('\n'.join(toPrint))

        except Exception as e:
            logger.error('Problem with setting weights to new values ...: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel.predict' )
    def predict(logger, self, X):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        '''
        yHat = None

        try:
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            t0 = time()
            with tf.Session() as sess:

                t1 = time()
                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                t2 = time()
                yHat = sess.run(self.result, feed_dict = {self.Inp: X})
                logger.info('Calculated yHat: {}'.format( yHat ))

                t3 = time()
                # Checkpoint the session before you exit ...
                # ------------------------------------------
                if (self.checkPoint is not None) and (not self.NNmodelConfig['keepChkPts']):
                    # delete the old checkpoints first
                    folder = os.path.dirname(self.checkPoint)
                    shutil.rmtree( folder )

                t4 = time()
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))

                t5 = time()
                
            toPrint = []
            toPrint.append('--------------------[{}]-----------------'.format(now))
            toPrint.append('[{}] [predict] Time to start a session     : {:.3}'.format( now,  t1 - t0))
            toPrint.append('[{}] [predict] Time to read a checkpoint   : {:.3}'.format( now,  t2 - t1))
            toPrint.append('[{}] [predict] Time to calculate a value   : {:.3}'.format( now,  t3 - t2))
            toPrint.append('[{}] [predict] Time to delete old folder   : {:.3}'.format( now,  t4 - t3))
            toPrint.append('[{}] [predict] Time to save new checkpoint : {:.3}'.format( now,  t5 - t4))
            if self.NNmodelConfig['showTimes']:
                print('\n'.join(toPrint))
            logger.info('\n'.join(toPrint))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return yHat

    @lD.log( logBase + '.NNmodel.predict' )
    def errorVal(logger, self, X, y):

        errVal = None
        try:
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
            saver = tf.train.Saver(tf.trainable_variables())

            t0 = time()
            with tf.Session() as sess:

                t1 = time()
                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    logger.info('An earlier checkpoint is available at {}. Using that.'.format(self.checkPoint))
                    saver.restore(sess, self.checkPoint)

                t2 = time()
                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                logger.info('Calculated errVal: {}'.format( errVal ))

                t3 = time()
                # Checkpoint the session before you exit ...
                # ------------------------------------------
                if (self.checkPoint is not None) and (not self.NNmodelConfig['keepChkPts']):
                    # delete the old checkpoints first
                    folder = os.path.dirname(self.checkPoint)
                    shutil.rmtree( folder )

                t4 = time()
                os.makedirs('../data/checkpoints/{}'.format(now))
                self.checkPoint = saver.save(sess, '../data/checkpoints/{0}/{0}.ckpt'.format(now))
                logger.info( 'Checkpoint saved at : {}'.format(self.checkPoint))

                t5 = time()

            toPrint = []
            toPrint.append('--------------------[{}]-----------------'.format(now))
            toPrint.append('[{}] [errorVal] Time to start a session     : {:.3}'.format( now,  t1 - t0))
            toPrint.append('[{}] [errorVal] Time to read a checkpoint   : {:.3}'.format( now,  t2 - t1))
            toPrint.append('[{}] [errorVal] Time to calculate a value   : {:.3}'.format( now,  t3 - t2))
            toPrint.append('[{}] [errorVal] Time to delete old folder   : {:.3}'.format( now,  t4 - t3))
            toPrint.append('[{}] [errorVal] Time to save new checkpoint : {:.3}'.format( now,  t5 - t4))
            if self.NNmodelConfig['showTimes']:
                print('\n'.join(toPrint))
            logger.info('\n'.join(toPrint))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVal

class NNmodel1():
    '''[summary]
    
    [description]
    '''

    @lD.log(logBase + '.NNmodel.__init__')
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

        weights = None

        if not self.modelOK:
            logger.error('The model is not generated properly. The optimizer will not be evaluated.')
            return

        try:


            if self.optimizer is None:
                logger.info('Generating a new optimizer ...')
                self.optimizer = tf.train.AdamOptimizer(name = 'opt', **params).minimize( self.err )

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

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

                print('Final error: {}'.format(err))

                weights = sess.run(self.allW + self.allB)
                    
        except Exception as e:
            logger.error('Unable to optimize the model given the data: {}'.format(str(e)))

        return weights

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

        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                weights = sess.run(self.allW + self.allB)
                return weights

        except Exception as e:
            logger.error('Unable to get the weights: {}'.format(str(e)))

        return weights

    @lD.log( logBase + '.NNmodel.setWeights' )
    def setWeights(logger, self, weights, sess):
        '''set weights for a sesstion
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            weights {[type]} -- [description]
            sess {[type]} -- [description]
        '''

        try:
            nW = len(self.allW)
            W = weights[:nW]
            B = weights[nW:]

            for i in range(len(W)):
                sess.run(tf.assign( self.allW[i], W[i] ))

            for i in range(len(B)):
                sess.run(tf.assign( self.allB[i], B[i] ))

        except Exception as e:
            logger.error('Problem with setting weights to new values ...: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel.errorValW' )
    def errorValW(logger, self, X, y, weights):

        errVal = None

        try:
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                self.setWeights(weights, sess)

                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                logger.info('Calculated errVal: {}'.format( errVal ))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVal

    @lD.log( logBase + '.NNmodel.errorValWs' )
    def errorValWs(logger, self, X, y, weightsList):

        errVals = []

        try:
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for weights in tqdm(weightsList):

                    self.setWeights(weights, sess)

                    errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                    logger.info('Calculated errVal: {}'.format( errVal ))

                    errVals.append( errVal )

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVals

class NNmodel2():
    '''[summary]
    
    [description]
    '''

    @lD.log(logBase + '.NNmodel2.__init__')
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
            self.modelOK = True

        except Exception as e:
            logger.error('Unable to geberate the required model: {}'.format(str(e)))


        return

    @lD.log(logBase + '.NNmodel2.fitAdam')
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

        weights = None

        if not self.modelOK:
            logger.error('The model is not generated properly. The optimizer will not be evaluated.')
            return

        try:


            if self.optimizer is None:
                logger.info('Generating a new optimizer ...')
                self.optimizer = tf.train.AdamOptimizer(name = 'opt', **params).minimize( self.err )

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

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

                print('Final error: {}'.format(err))

                weights = sess.run(self.allW + self.allB)
                    
        except Exception as e:
            logger.error('Unable to optimize the model given the data: {}'.format(str(e)))

        return weights

    @lD.log( logBase + '.NNmodel2.getWeights' )
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

        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                weights = sess.run(self.allW + self.allB)
                return weights

        except Exception as e:
            logger.error('Unable to get the weights: {}'.format(str(e)))

        return weights

    @lD.log( logBase + '.NNmodel2.setWeights' )
    def setWeights(logger, self, weights, sess):
        '''set weights for a sesstion
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            weights {[type]} -- [description]
            sess {[type]} -- [description]
        '''

        try:
            nW = len(self.allW)
            W = weights[:nW]
            B = weights[nW:]

            for i in range(len(W)):
                sess.run(self.allAssignW[i], feed_dict = { 'PHW_{}:0'.format(i): W[i] } )

            for i in range(len(B)):
                sess.run(self.allAssignB[i], feed_dict = { 'PHB_{}:0'.format(i): B[i] } )
                

        except Exception as e:
            logger.error('Problem with setting weights to new values ...: {}'.format(str(e)))

        return

    @lD.log( logBase + '.NNmodel2.errorValW' )
    def errorValW(logger, self, X, y, weights):

        errVal = None

        try:
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                self.setWeights(weights, sess)

                errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                logger.info('Calculated errVal: {}'.format( errVal ))

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVal

    @lD.log( logBase + '.NNmodel2.errorValWs' )
    def errorValWs(logger, self, X, y, weightsList):

        errVals = []

        try:
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for weights in tqdm(weightsList):

                    self.setWeights(weights, sess)

                    errVal = sess.run(self.err, feed_dict = {self.Inp: X, self.Op: y})
                    logger.info('Calculated errVal: {}'.format( errVal ))

                    errVals.append( errVal )

        except Exception as e:
            logger.error( 'Unable to make a prediction: {}'.format(str(e)) )

        return errVals

class NNmodel3():
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

