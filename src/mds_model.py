import tensorflow as tf
import numpy as np
import pandas as pd
from copy import deepcopy
from tensorflow.keras.layers import Dense, Multiply, Add
from tensorflow.keras import Model
from tensorflow.keras.constraints import NonNeg 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

## TODO move to matrix_utils.py 
## MAYBE rewrite in pure tensorflow
### need config_dim 
### -> then apply np.outer on each to get Aij
##
def self_outer(basis_vector:tf.Tensor)->tf.Tensor:
    return tf.einsum('i,j->ij', basis_vector, basis_vector)
    

def all_pairwise_basis_diffs(max_index:int)->tf.Tensor:
  
    assert max_index > 1 ## ignore first basis vector  
    eye_ref = tf.eye(num_rows=max_index, num_columns=max_index, dtype=tf.float64)
    
    ## tensor with shape given by all pairs 
    output = tf.TensorArray(dtype=tf.float64, size=(max_index**2 - max_index)//2)
    idx = 0
    for upper_index in range(2, max_index):
        for lower_index in range(upper_index):
            diff = eye_ref[:,lower_index] - eye_ref[:,upper_index]
            output = output.write(idx, self_outer(diff))
            idx += 1
#                      np.apply_along_axis(func1d=self_outer, axis=0, arr=ei_minus_ejs)
    return output 
                     
### then apply func1d on Aijs 
def pairwise_dist_block_chunk(outer_distance:np.array, configuration:np.array):
    # Compute tr X'A_ij X 
    outer_distance = tf.cast(outer_distance, tf.float64)
    configuration = tf.cast(configuration, tf.float64)
    return tf.linalg.trace(tf.einsum('ij,jk->ik',tf.einsum('ij,jk->ik', tf.transpose(configuration), outer_distance),configuration))
    
@tf.function
def get_pairwise_distances(configuration:tf.Variable)->tf.Variable:
    number_obs = configuration.shape[0]
    number_dists = (number_obs**2 - number_obs)//2
    ## tensorarray with shape given by all pairs; this one is to hold the individual differences
    output = tf.TensorArray(dtype=tf.float64, size=number_dists)
    
    outerL = all_pairwise_basis_diffs(number_obs)
 
    ## loop over outerL and apply pairwise_dist_block_chunck
    for j in range(number_dists):
        output = output.write(j, pairwise_dist_block_chunk(outerL.read(j), configuration))
    return output.stack()


## wrap pairwise distance in a layer
class PDistLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PDistLayer, self).__init__()
        
    def call(self, inputs):
        return tf.expand_dims(get_pairwise_distances(inputs),0)

#todo move to mdsmodel.py
class MDSModel(Model):
    def __init__(self,configuration):
        super(MDSModel, self).__init__()
        self.config = tf.Variable(configuration, dtype=tf.float64, trainable=True, name ='initial_config')
        self.new_config = tf.Variable(tf.ones_like(self.config), trainable=True, name = 'derived_config')
        self.number_obs = self.config.shape[0]
        self.number_dists = (self.number_obs**2 - self.number_obs)//2
        self.distance = PDistLayer()
        self.d1 = Dense(128, activation='relu', kernel_constraint=NonNeg(),use_bias=False)
        self.d2 = Dense(self.number_dists, activation='relu', kernel_constraint=NonNeg(),use_bias=False)
        

    def call(self, x):
        #nb we are throwing away x here 
        x = Multiply()([self.new_config, self.config])
        x = self.distance(x)
        # x = self.d1(x)
        x = self.d2(x)
        x = tf.math.divide_no_nan((x - tf.math.minimum(x)),(tf.math.maximum(x) - tf.math.minimum(x)))
        return x

@tf.function
def stress(y_true, y_pred):
    Sstar = tf.math.reduce_sum(tf.math.square(y_true - y_pred),axis=-1)
    Tstar = tf.math.reduce_sum(tf.square(y_true),axis=-1)
    S = tf.math.sqrt(tf.math.divide_no_nan(Sstar,Tstar))
    return S  # Note the `axis=-1`



class MDSModelFit:
    def __init__(self, start_config):
        # The monotone regression: wrap in function
        self.configuration = deepcopy(start_config)
        self.model = MDSModel(self.configuration)
        self.model.compile(optimizer='adam', loss=stress)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.loss_object = stress
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(self, config, dis_vec):
        with tf.GradientTape(persistent=False) as tape:
    #         tape.watch(config)
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(config, training=True)
            print("made predictions")
            loss = self.loss_object(dis_vec, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        return predictions 