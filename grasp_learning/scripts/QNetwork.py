#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from utils import *
from FileHandler import *


class QNetwork(object):
    def __init__(self, gamma, relative_path, model_name):
        self.gamma = gamma
        self.relative_path = relative_path
        self.model_name = model_name
    def create_tensorflow_graph(self,num_inputs, num_outputs, num_rewards, hidden_layers_sizes,load_example_model):

        self.g = tf.Graph()

        self.sess = tf.InteractiveSession(graph=self.g) #,config=tf.ConfigProto(log_device_placement=True))
        with tf.name_scope('input'):
            # Placeholder for the states i.e. task errors e 
            self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs],name="Task_error_placeholder") 
            # Placeholder for the chosen Q value
            self.Qselected = tf.placeholder(tf.int32, [None], name="Task_dynamics_placeholder")
            # Placeholder for the target Q values
            self.Qtarget = tf.placeholder(tf.float32, [None], name="Task_dynamics_placeholder")

        with tf.name_scope("Hyper_params"):
            # Placeholder for the learning rate
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate_placeholder")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.QNetwork = self.construct_Neural_Network(self.state_placeholder, num_outputs, hidden_layers_sizes)

        self.indexes = tf.range(0, tf.shape(self.QNetwork)[0]) * tf.shape(self.QNetwork)[1] + self.Qselected
        self.responsible_outputs = tf.gather(tf.reshape(self.QNetwork, [-1]), self.indexes)


        self.var_list = tf.trainable_variables()

        with tf.name_scope("Q_learning"):
            # The output from the neural network is the mean of a Gaussian distribution. This variable is simply the
            # log likelihood of that Gaussian distribution

            self.loss = tf.reduce_mean(tf.square(self.responsible_outputs-self.Qtarget), name='loss_reduce_mean')#/self.batch_size 
            self.updateModel = self.optimizer.minimize(self.loss)
            # Get the list of all trainable variables in our tensorflow graph
            # Compute the analytic gradients of the loss function given the trainable variables in our graph
            #with tf.device('/cpu:0'):
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        if load_example_model:
            self.saver.restore(self.sess, self.relative_path+'models/'+self.model_name)
        else:
            self.saver.save(self.sess, self.relative_path+'models/'+self.model_name)

        self.setup_tensorboard()

        print "Network initialized"

    def construct_Neural_Network(self, input_placeholder, num_outputs, hidden_units):
        
            prev_layer = input_placeholder
            layer = 0
            for units in hidden_units:
                prev_layer = self.create_layer(prev_layer, units, tf.nn.tanh, layer , "Hidden_"+str(layer))
                layer+=1

            # last layer is a linear output layer
            num_outputs = int(num_outputs)
            return self.create_layer(prev_layer, num_outputs, None, layer, "Output")

    def create_layer(self, prev_layer, layer_size, activation_fun, num_layer, name):
        with tf.name_scope(name):
                with tf.name_scope('weights'):
                    weights = tf.Variable(tf.truncated_normal([int(prev_layer.shape[1]), layer_size]),name="NN/w"+str(num_layer)+"/t")
                with tf.name_scope('biases'):
                    biases = tf.Variable(tf.truncated_normal([layer_size]), name="NN/b"+str(num_layer)+"/t")
                if activation_fun == None:
                    layer_output = tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases,name="NN/output/t")
                else:
                    layer_output = activation_fun(tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases),name="NN/L"+str(num_layer)+"/t")

                return layer_output

    def predict(self, states):
        feed_dict = {self.state_placeholder: states}
        return self.sess.run(self.QNetwork, feed_dict)

    def train(self, states, Qtarget, actions, lr):

        feed_dict={self.state_placeholder  : states,
                   self.Qtarget            : Qtarget,
                   self.Qselected          : actions,
                   self.learning_rate      : lr}
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for _ in xrange(1):
            _ , error, _ = self.sess.run([self.updateModel , self.loss, self.merged_summary], feed_dict,
                        run_options,
                        run_metadata)
        return error

    def setup_tensorboard(self):
        with tf.name_scope('summaries'):
            tf.summary.histogram('loss_hist', self.loss)
            tf.summary.histogram('task_errors_hist', self.state_placeholder)
            tf.summary.histogram('Q_out_hist', self.Qselected)
            # tf.summary.histogram('Q_target_hist', self.Qtarget)

            tf.summary.scalar('learning_rate',self.learning_rate)

            self.store_containers_in_tensorboard(self.var_list)

            self.merged_summary = tf.summary.merge_all()
            self.set_tensorflow_summary_writer()

    def store_containers_in_tensorboard(self, container):

        if type(container[0]) is tuple:
            for grad,var in container:
                name = grad.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , grad)
        elif type(container) is list:
            for var in container:
                name = var.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , var)

    def set_tensorflow_summary_writer(self):
        dir_path = self.relative_path+'/graphs/trial_1'
        create_directory(dir_path)

        #setup tensor board writers
        self.train_writer = tf.summary.FileWriter(dir_path,self.g)

    def get_network_parameters(self):
        var = np.array([])
        for trainable_variable in self.var_list:
            var = np.append(var, trainable_variable.eval(session=self.sess))
        return var

    def calculate_target_Q(self, action, reward, terminal, next_state):
        Qtarget = np.zeros_like(reward)
        if terminal:
            Qtarget = reward
        else:
            next_state = np.reshape(next_state, (1,self.state_placeholder.shape[1]))
            Qtarget = reward + self.gamma*self.sess.run(self.QNetwork,feed_dict={self.state_placeholder  : next_state}).flatten()[action]
            # Qtarget = reward+self.gamma*np.max(self.sess.run(self.QNetwork,feed_dict={self.state_placeholder  : next_state}))
        return Qtarget

    def get_trainable_variables(self):
        res = []
        for var in self.var_list:
            res.append(var.eval(session=self.sess))
        return res

    def set_new_parameters(self, new_params, tau):
        i = 0
        for var in tf.trainable_variables():
            assign_op = var.assign(tau*var.eval(session=self.sess)+(1-tau)*new_params[i])
            self.sess.run(assign_op)
            i+=1    

    def print_parameters(self):
        for trainable_variable in self.var_list:
            print trainable_variable.eval(session=self.sess)

    def get_best_action(self, state):
        actions = self.predict(state)
        return np.argmax(actions)

    def restore_graph(self):
        self.saver.restore(self.sess, self.relative_path+'models/'+self.model_name)