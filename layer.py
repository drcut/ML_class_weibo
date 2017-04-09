#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import time
import numpy as np
from six.moves import xrange
import random
import warnings
set_keep = globals()
set_keep['_layers_name_list'] =[]
set_keep['name_reuse'] = False

try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def print_all_variables(train_only=False):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def get_layers_with_name(network=None, name="", printable=False):
    """Get layer list in a network by a given name scope.
    >>> layers = tl.layers.get_layers_with_name(network, "CNN", True)
    """
    assert network is not None
    print("  [*] geting layers with %s" % name)

    layers = []
    i = 0
    for layer in network.all_layers:
        # print(type(layer.name))
        if name in layer.name:
            layers.append(layer)
            if printable:
                # print(layer.name)
                print("  got {:3}: {:15}   {}".format(i, layer.name, str(layer.get_shape())))
                i = i + 1
    return layers

def initialize_global_variables(sess=None):
    """Excute ``sess.run(tf.global_variables_initializer())`` for TF12+ or
    sess.run(tf.initialize_all_variables()) for TF11.

    Parameters
    ----------
    sess : a Session
    """
    assert sess is not None
    try:    # TF12
        sess.run(tf.global_variables_initializer())
    except: # TF11
        sess.run(tf.initialize_all_variables())

## Basic layer
class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    """
    def __init__(
        self,
        inputs = None,
        name ='layer'
    ):
        self.inputs = inputs
        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)
    def print_params(self, details=True):
        ''' Print all info of parameters in the network'''
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                except Exception as e:
                    print(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(i, str(p.get_shape()), p.name))
        print("  num of params: %d" % self.count_params())
    def print_layers(self):
        ''' Print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            print("  layer %d: %s" % (i, str(p)))
    def count_params(self):
        ''' Return the number of parameters in the network '''
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params
    def __str__(self):
        print("\nIt is a Layer class")
        self.print_params(False)
        self.print_layers()
        return "  Last layer is: %s" % self.__class__.__name__
class Cnn_Softmax_Wrapper(Layer):
  def __init__(self,
               buckets,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               class_num=9,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               name='wrapper'):
    Layer.__init__(self)#, name=name)

    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.class_num=class_num

    with tf.variable_scope(name) as vs:   
        # Feeds for inputs.
        self.encoder_inputs=tf.placeholder(tf.float32, shape=[self.batch_size,70],
                                                    name="encoder")
        self.target_inputs=tf.placeholder(tf.float32, shape=[self.batch_size,self.class_num],
                                                    name="target")

        # Our targets are decoder inputs shifted by one.
        self.targets = self.target_inputs
        # Training outputs and losses.
        conv_stride=3
        pool_stride=3
        '''
        print("encoder")
        print(self.encoder_inputs)
        print("after reshape")
        print(tf.reshape(self.encoder_inputs,[batch_size,-1,1]))
        '''
        state = tf.layers.conv1d(tf.reshape(self.encoder_inputs,[batch_size,-1,1]),  filters = 11, kernel_size=5, strides=conv_stride, padding='valid', data_format='channels_last', activation=tf.sigmoid, use_bias=True, trainable=True, name="conv1d") 
        
        after_pool = tf.reshape(tf.layers.max_pooling1d(state , pool_size=3, strides=pool_stride, padding='valid', 
          data_format='channels_last', name="pool"),[batch_size,-1])
        
        full_connect1 = tf.contrib.layers.legacy_fully_connected(after_pool,1024, activation_fn=tf.sigmoid, name=None, trainable=True)
        #print(full_connect1)
        full_connect2 = tf.contrib.layers.legacy_fully_connected(full_connect1,self.class_num, activation_fn=tf.nn.relu, name=None, trainable=True)
        y_conv=tf.nn.softmax(full_connect2)
        #print("y_conv")
        #print(y_conv)
        #print("target inputs")
        #print(self.target_inputs)
        self.losses = tf.losses.softmax_cross_entropy(tf.reshape(self.target_inputs,[batch_size,class_num]), y_conv)
        self.outputs = tf.argmax(y_conv,1)
        #print(self.outputs)

        params = tf.trainable_variables()
        #self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam').minimize(self.losses)
        
        self.gradient_norms = []
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.losses, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
        self.gradient_norms.append(norm)
        #self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
        self.updates=opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        
        self.all_params = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        
  def step(self, session, encoder_inputs, target_inputs,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.
    """
    encoder_size = self.buckets[bucket_id]
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    tmp_target_inputs = []
    for batch_idx in xrange(self.batch_size):
      one_hot = [0.0]* self.class_num
      one_hot[target_inputs[batch_idx]]=1.0
      tmp_target_inputs.append(one_hot)

    input_feed[self.target_inputs.name] = tmp_target_inputs
    input_feed[self.encoder_inputs.name] = encoder_inputs

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates,  # Update Op that does SGD.
                     #self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses
                     ]  # Loss for this batch.
    else:
      output_feed = [self.losses]  # Loss for this batch.
      output_feed.append(self.outputs[bucket_id])
    #session.run(self.opt)
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id,PAD_ID=0):
    """Get a random batch of data from the specified bucket, prepare for step.
    """
    encoder_size = self.buckets[bucket_id]
    encoder_inputs, target_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    for _ in xrange(self.batch_size):
      encoder_input, target_input = random.choice(data[bucket_id])
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(encoder_input+encoder_pad)
      target_inputs.append(target_input)

    batch_encoder_inputs = encoder_inputs
    batch_target_inputs = target_inputs
    '''
    print("batch_encoder_inputs")
    print(batch_encoder_inputs)
    print("batch_target_inputs")
    print(batch_target_inputs)
    '''
    #batch_encoder_inputs:shape[batch_size*len],each element is an intergre

    return batch_encoder_inputs, batch_target_inputs
