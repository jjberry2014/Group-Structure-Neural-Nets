import tensorflow as tf
import numpy as np
def Gaussian_NLL_max_mean_reduce(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))
    eps = 0
    logprob = -tf.math.log(sigma+eps) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma+eps)**2
    loss = tf.reduce_max(-logprob, axis=ax) # along columns (index 1)
    return tf.reduce_mean(loss) if reduce else loss # along rows (index 0)

def Gaussian_NLL_Loss(true,pred):
    mu, sigma = tf.split(pred, 2, axis=-1)
    loss = Gaussian_NLL_max_mean_reduce(true,mu,sigma)
    return loss

def max_mean_reduce(y,y_true,reduce=True):
    squaredError = tf.math.square(y-y_true)
    loss = tf.reduce_max(tf.math.sqrt(squaredError), axis=1) # reduce along sample's output 
    return tf.reduce_mean(loss) if reduce else loss # 