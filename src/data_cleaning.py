from re import X
import tensorflow as tf
import numpy as np 

def tf_generate_starting_configuration(dis_vec, n_points, dim=2):
    ## Generate kruskal_style t-dimensional starting configuration
    #  N = 1/2 + sqrt(2Y + 1/4)

    assert n_points == int(1/2 + tf.sqrt(2*len(dis_vec) +1/4))
    
    max_multiplier = n_points//dim + 1
    #say 503 points in2 dim, then we need 252 diags, drop the last row of the last one , s0 have to go from 
    # 1 to n_points/dim +2
    starting_config = tf.stack([tf.linalg.diag(j*tf.ones(dim)) for j in range(1,max_multiplier)])
    config_out = tf.reshape(starting_config[:n_points,:],(-1,2))
    return tf.cast(config_out, tf.float64)


def generate_starting_configuration(dis_vec, n_points, dim=2, random=True):
    ## Generate kruskal_style t-dimensional starting configuration
    #  N = 1/2 + sqrt(2Y + 1/4)

    assert n_points == int(1/2 + np.sqrt(2*len(dis_vec) +1/4))
    
    max_multiplier = n_points//dim + 1
    #say 503 points in2 dim, then we need 252 diags, drop the last row of the last one , s0 have to go from 
    # 1 to n_points/dim +2
    if random:
        starting_config = np.random.uniform(size=(n_points, dim))
    else:
        starting_config = np.stack([np.diag(j*np.ones(dim)) for j in range(1,max_multiplier)])
    config_out = np.reshape(starting_config[:n_points,:],(-1,2))
    config_out /= np.linalg.norm(config_out)
    return config_out
     
