from logs import logDecorator as lD
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

        try:

            logger.info('Generating a new model')

            self.inpSize = inpSize
            self.Inp     = tf.placeholder(dtype=tf.float32, shape=inpSize)
            self.Op      = tf.placeholder(dtype=tf.float32, shape=opSize)
            
            self.allW    = []
            self.allB    = []

            self.result  = None

            prevSize = inpSize[0]
            for i, l in enumerate(layers):
                tempW = tf.Variable( 0.1*(np.random.rand(l, prevSize) - 0.5), dtype=tf.float32 )
                tempB = tf.Variable( 0, dtype=tf.float32 )

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


        optimizer = tf.train.AdamOptimizer(**params).minimize( self.err )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(N):
                _, err = sess.run([optimizer, self.err ], feed_dict={
                        self.Inp: X, self.Op: y
                    })
                print('{:6d} --> {}'.format(i, err))

        return

