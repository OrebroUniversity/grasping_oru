#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from utils import *

class NeuralNetworkValueFunction(object):

    def __init__(self, num_inputs , num_outputs, path):

        self.vg = tf.Graph()
        self.session = tf.InteractiveSession(graph=self.vg)
        self.net = None

        self.num_inputs = num_inputs
        self.num_outputs = 1
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_inputs], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.batch_size = tf.placeholder(tf.float32, name="batch_size")
        self.construct_ValueNet()
	self.vg_writer = tf.summary.FileWriter(path+'vgraphs',self.vg)
        self.initialize = True

    def construct_ValueNet(self):
        with self.vg.as_default():

            hidden_size = 64

            weight_init = tf.random_uniform_initializer(-0.05, 0.05)
            bias_init = tf.constant_initializer(0)

            h1 = tf.nn.tanh(fully_connected(self.x, self.num_inputs, hidden_size, weight_init, bias_init, "h1"))
            h2 = tf.nn.tanh(fully_connected(h1, hidden_size, hidden_size, weight_init, bias_init, "h2"))
            h3 = fully_connected(h2, hidden_size, 1, weight_init, bias_init, "h3")
            self.net = tf.reshape(h3, (-1,))
            self.l2 = tf.nn.l2_loss(self.net - self.y)/self.batch_size
            self.optimizer = tf.train.AdamOptimizer().minimize(self.l2)
	    #freezing the value graph
            self.session.run(tf.global_variables_initializer())
	    self.vg.finalize()

    def reshape_input(self, input_):
        return input_.reshape(1,self.num_inputs)

    def reshape_output(self, output_):
        return output_.reshape(output_.shape[0],self.num_outputs)


    def predict(self, states):
        with self.vg.as_default():
            if self.initialize:
                return np.asarray([0])
            else:
                ret = self.session.run(self.net, {self.x: self.reshape_input(states)})
                return ret

    def train(self, input_data, output_data, batch_size = 1.0, n_iter=50):
        with self.vg.as_default():
            self.initialize = False
            for _ in range(n_iter):
                self.session.run(self.optimizer, {self.x: input_data, self.y: output_data, self.batch_size: batch_size})

class LinearValueFunction(object):

    def __init__(self):
        self.coeffs = None

    def train(self, input_data, output_data, batch_size):
        input_data = np.hstack((input_data, np.ones((input_data.shape[0],1))))
        self.coeffs = np.linalg.lstsq(input_data, output_data)[0]

    def predict(self, state):
        return np.zeros_like(1) if self.coeffs is None else np.dot(np.append(state, 1),self.coeffs)