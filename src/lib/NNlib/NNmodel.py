from logs import logDecorator as lD
from datetime import datetime as dt

import json
import numpy       as np 
import tensorflow  as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.NNlib.NNmodel'

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
            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
            saver = tf.train.Saver(tf.trainable_variables())
            if self.optimizer is None:
                self.optimizer = tf.train.AdamOptimizer(name = 'opt', **params).minimize( self.err )

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                if self.checkPoint is not None:
                    saver.restore(sess, self.checkPoint)
                
                print('Optimization ...')
                for i in range(N):
                    _, err = sess.run([self.optimizer, self.err ], feed_dict={
                            self.Inp: X, self.Op: y
                        })
                    # print('{:6d} --> {}'.format(i, err))

                self.checkPoint = saver.save(sess, '../data/checkpoints/{}.ckpt'.format(now))
                print(self.checkPoint)


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

