#!/usr/bin/env python
import roslib

import rospy
import sys
import bisect
import itertools
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from grasp_learning.srv import *
from grasp_learning.srv import *
from std_msgs.msg import Empty
from copy import deepcopy
from math import sqrt
from math import pow
import tensorflow as tf
import numpy as np
import time
from scipy.signal import butter, lfilter, freqz, filtfilt

class ValueNet(object):

    def __init__(self, num_inputs , num_outputs):

        self.vg = tf.Graph()

        self.sess = tf.InteractiveSession(graph=self.vg)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.input_data_placeholder = tf.placeholder(tf.float32, [None, num_inputs])
        self.output_data_placeholder = tf.placeholder(tf.float32, [None, num_outputs])

        self.w1 = self.w2 = self.w3 = self.b1 = self.b2 = self.b3 = 0
        self.logits = 0

    def construct_ValueNet(self, input_data, output_data, HIDDEN_UNITS_L1 = 20, HIDDEN_UNITS_L2 = 5):
        with self.vg.as_default():

            weights_1 = tf.Variable(tf.truncated_normal([self.num_inputs, HIDDEN_UNITS_L1]), name="input_layer")
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            layer_1_outputs = tf.nn.tanh(tf.matmul(self.input_data_placeholder, weights_1) + biases_1)

            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]))
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]))
            layer_2_outputs = tf.nn.tanh(tf.matmul(layer_1_outputs, weights_2) + biases_2)
            # logits = tf.matmul(layer_1_outputs, weights_2)+biases_2

            weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, self.num_outputs]))
            biases_3 = tf.Variable(tf.zeros([self.num_outputs]))
            logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, self.output_data_placeholder)))

            train_step = tf.train.AdamOptimizer(0.001).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            # task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            output_data = self.reshape_output(output_data)
            for i in range(500):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={self.input_data_placeholder: input_data,
                                      self.output_data_placeholder: output_data})


            self.w1, self.w2, self.w3, self.b1, self.b2, self.b3 = weights_1, weights_2, weights_3, biases_1, biases_2, biases_3
            self.logits = logits

    def reshape_input(self, input_):
        return input_.reshape(1,self.num_inputs)

    def reshape_output(self, output_):
        return output_.reshape(output_.shape[0],self.num_outputs)

    def get_value(self, states):
        with self.vg.as_default():
            state = self.reshape_input(states)
            return self.sess.run(self.logits, feed_dict={self.input_data_placeholder: state})

    def get_vars(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def train(self, input_data, output_data, batch_size = 1.0, n_iter=500):

        with self.vg.as_default():
            temp = set(tf.global_variables())


            error_function = tf.reduce_mean(tf.square(tf.subtract(self.logits, self.output_data_placeholder)))
            train_step = tf.train.AdamOptimizer(0.001).minimize(error_function)
            self.sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
            output_data = self.reshape_output(output_data) 
            for i in xrange(n_iter):
                self.sess.run(train_step, feed_dict={self.input_data_placeholder: input_data, self.output_data_placeholder: output_data})





class Policy(object):
    """docstring for Policy"""

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')

        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.sigma = 2
        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.gamma = 0.99
        self.batch_size = 10
        self.g = tf.Graph()
        self.train = True
        self.learning_rate = 0.5
        self.eval_episode = True


        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_unnormalized_returns = []
        self.all_disc_returns = []
        self.all_disc_returns1 = []
        self.all_disc_returns2 = []

        self.all_actions = []
        self.actions = []
        
        self.task_errors = []
        self.all_task_errors = []

        self.task_measure = []
        self.all_task_measure = []

        self.NN_output = []

        self.prev_batch_mean_return = 0

        self.VN = ValueNet(num_inputs, num_rewards)

        self.sess = tf.InteractiveSession(graph=self.g)

        self.task_error_placeholder = tf.placeholder(tf.float32, [None, num_inputs])
        self.task_dyn_placeholder = tf.placeholder(tf.float32, [None, num_outputs])
        self.advant = tf.placeholder(tf.float32,[None,num_rewards])

        self.train_weights = [ ]
        self.eval_weights = [ ]

        input_data, output_data = self.parse_input_output_data(input_output_data_file)

        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards.txt'
        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.task_errors_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/task_errors/task_errors.txt'
        self.loss_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/losses.txt'
        self.log_likelihood_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/log_likelihood.txt'
        self.NN_output_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/action_dist_mean.txt'
        self.task_measure_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/task_measure.txt'
        self.fisher_matrix_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients/fisher_matrix.txt'
        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'

        if num_rewards>1:
            self.reward_file_name1 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards1.txt'
            self.disc_reward_file_name1 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards1.txt'
            self.baseline_file_name1 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/baseline1.txt'
            self.advantages_file_name1 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/advantages1.txt'

            self.reward_file_name2 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards2.txt'
            self.disc_reward_file_name2 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards2.txt'
            self.baseline_file_name2 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/baseline2.txt'
            self.advantages_file_name2 = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/advantages2.txt'

            self.reset_files([self.weights_file_name, self.log_likelihood_file_name, self.advantages_file_name1, self.loss_file_name, self.baseline_file_name1,
                             self.reward_file_name1, self.actions_file_name, self.task_errors_file_name, self.disc_reward_file_name1, self.NN_output_file_name,
                             self.task_measure_file_name, self.fisher_matrix_file_name, self.advantages_file_name2, self.baseline_file_name2, self.reward_file_name2,
                             self.disc_reward_file_name2])

        else:  
            self.disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards.txt'

            self.baseline_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/baseline.txt'
            self.advantages_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/advantages.txt'
            self.reset_files([self.weights_file_name, self.log_likelihood_file_name, self.advantages_file_name, self.loss_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.task_errors_file_name, self.disc_reward_file_name, self.NN_output_file_name,
                             self.task_measure_file_name, self.fisher_matrix_file_name])




        self.ff_NN_train, w1_train, w2_train, w3_train = self.construct_ff_NN(self.task_error_placeholder, self.task_dyn_placeholder , input_data, output_data, num_inputs, num_outputs)
        # self.ff_NN_train, w1_train, w2_train = self.construct_ff_NN(self.task_error_placeholder, self.task_dyn_placeholder , input_data, output_data, num_inputs, num_outputs)

        self.train_weights.append(w1_train)
        self.train_weights.append(w2_train)
        # self.train_weights.append(w3_train)
        self.store_weights()

    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()

    def parse_input_output_data(self, input_file):
        input_data = []
        output_data = []
        input_ = []
        output_ = []
        i = 0
        with open(input_file, 'rU') as f:
            for line in f:
                if (i==0 or i==1):
                # if (i==0):
                    i+=1
                    continue
                line = line.split()
                for string in xrange(len(line)):
                    if string%2==0:
                        # input_data.append(float(line[string]))
                        input_data.append(float(line[string])+np.random.normal(0, 0.15))

                    else:
                        # if string == 1:
                        # output_data.append(line[string])
                        output_data.append(float(line[string])+np.random.normal(0, 0.25))
                        # output_data.append(float(line[string]))

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)


    def construct_ff_NN(self, task_error_placeholder, task_dyn_placeholder, task_errors, task_dyn, NUM_INPUTS = 1, NUM_OUTPUTS = 1, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 5):
        with self.g.as_default():

            weights_1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, HIDDEN_UNITS_L1]), name="input_layer")
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            layer_1_outputs = tf.nn.tanh(tf.matmul(task_error_placeholder, weights_1) + biases_1)

            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]))
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]))
            layer_2_outputs = tf.nn.tanh(tf.matmul(layer_1_outputs, weights_2) + biases_2)
            # logits = tf.matmul(layer_1_outputs, weights_2)+biases_2

            weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, NUM_OUTPUTS]))
            biases_3 = tf.Variable(tf.zeros([NUM_OUTPUTS]))
            logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, task_dyn_placeholder)))

            train_step = tf.train.AdamOptimizer(0.001).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            # task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            # task_errors= task_errors.reshape(task_errors.shape[0],1)
            for i in range(2500):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={task_error_placeholder: task_errors,
                                      task_dyn_placeholder: task_dyn})

            print "Network trained"
            print loss
            # return logits, weights_1, weights_2
            return logits, weights_1, weights_2, weights_3

    def store_weights(self):

        var = np.array([])
        for trainable_variable in tf.trainable_variables():
            var = np.append(var, trainable_variable.eval(session=self.sess))

        f_handle = file(self.weights_file_name,'a')
        np.savetxt(f_handle, [var] , delimiter='\t')
        f_handle.close()

    def store_NN_output(self):
        print "Storing neural network output"
        self.save_data_to_file(self.NN_output_file_name, self.NN_output)

    def store_task_measure(self):
        print "Storing task measure"
        self.save_data_to_file(self.task_measure_file_name, self.task_measure)

    def store_task_errors(self):
        print "Storing task measures"
        self.task_errors.pop()
        self.all_task_errors.append(np.asarray(self.task_errors))
        self.save_data_to_file(self.task_errors_file_name, self.task_errors)

    def store_actions(self):
        print "Storing actions"
        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))
        self.save_data_to_file(self.actions_file_name, self.actions)

    def store_advantages(self, advantages):
        print "Storing advantages"

        f_handle = file(self.advantages_file_name,'a')
        np.savetxt(f_handle, [advantages], delimiter='\t')
        f_handle.close()

        # f_handle = file(self.advantages_file_name1,'a')
        # np.savetxt(f_handle, [advantages[:,0]], delimiter='\t')
        # f_handle.close()

        # f_handle = file(self.advantages_file_name2,'a')
        # np.savetxt(f_handle, [advantages[:,1]], delimiter='\t')
        # f_handle.close()


    def store_baseline(self, baseline):
        print "Storing baseline"

        f_handle = file(self.baseline_file_name,'a')
        np.savetxt(f_handle, [baseline], delimiter='\t')
        f_handle.close()

        # f_handle = file(self.baseline_file_name1,'a')
        # np.savetxt(f_handle, [baseline[:,0]], delimiter='\t')
        # f_handle.close()

        # f_handle = file(self.baseline_file_name2,'a')
        # np.savetxt(f_handle, [baseline[:,1]], delimiter='\t')
        # f_handle.close()


    def store_loss_function(self, losses):
        print "Storing losses"
        f_handle = file(self.loss_file_name,'a')
        np.savetxt(f_handle, [losses], delimiter='\t')
        f_handle.close()

    def store_log_likelihood(self, log_likelihood):
        print "Storing log likelohood"
        self.save_data_to_file(self.log_likelihood_file_name, log_likelihood)

    def store_fisher_matrix(self, fisher_matrix):
        print "Storing fisher matrix"
        self.save_data_to_file(self.fisher_matrix_file_name, fisher_matrix)

    def save_data_to_file(self, filename, data):
        f_handle = file(filename,'a')
        for inner_list in data:
            for elem in inner_list:
                f_handle.write(str(elem)+" ")
        f_handle.write("\n")
        f_handle.close()

    def store_rewards(self):

        print "Storing reward"

        self.task_measure.pop(0)
        curr_rollout_return = self.calculate_return(self.task_measure)
        self.all_returns.append(curr_rollout_return)

        curr_rollout_disc_return = self.discount_rewards(curr_rollout_return) 
        self.all_disc_returns.append(curr_rollout_disc_return)
        # self.all_disc_returns1.append(curr_rollout_disc_return[:,0])
        # self.all_disc_returns2.append(curr_rollout_disc_return[:,1])

        f_handle = file(self.reward_file_name,'a')
        np.savetxt(f_handle, [curr_rollout_return], delimiter='\t')
        f_handle.close()

        # f_handle = file(self.reward_file_name1,'a')
        # np.savetxt(f_handle, [curr_rollout_return[:,0]], delimiter='\t')
        # f_handle.close()

        # f_handle = file(self.reward_file_name2,'a')
        # np.savetxt(f_handle, [curr_rollout_return[:,1]], delimiter='\t')
        # f_handle.close()


        f_handle = file(self.disc_reward_file_name,'a')
        np.savetxt(f_handle, [curr_rollout_disc_return], delimiter='\t')
        f_handle.close()

        # f_handle = file(self.disc_reward_file_name1,'a')
        # np.savetxt(f_handle, [curr_rollout_disc_return[:,0]], delimiter='\t')
        # f_handle.close()

        # f_handle = file(self.disc_reward_file_name2,'a')
        # np.savetxt(f_handle, [curr_rollout_disc_return[:,1]], delimiter='\t')
        # f_handle.close()


    def discount_rewards(self, reward):

        discounted_r = np.zeros_like(reward)
        running_add = 0#np.zeros(reward.shape[1])
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        discounted_r = self.normalize_data(discounted_r)
        return discounted_r


    def calculate_return(self, curr_reward):


        dist_square = np.square(np.asarray(curr_reward))
        alpha = 1e-15

        rollout_return = np.zeros_like(dist_square[:,0])

        # for e in xrange(dist_square.shape[1]):
        #     if e==0:
        #         rollout_return[:,e] = -15*dist_square[:,e]-0.4*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
        #     else:
        #         rollout_return[:,e] = -15*dist_square[:,e]-0.4*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)

        for e in xrange(dist_square.shape[1]):
            if e==0:
                rollout_return += -15*dist_square[:,e]-0.4*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
            else:
                rollout_return += -15*dist_square[:,e]-0.4*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)


        self.all_unnormalized_returns.append(rollout_return) 

        return rollout_return

    def normalize_data(self, data):
        return (data-np.mean(data,axis=0))/(np.std(data,axis=0)+1e-8)

    def calculate_reward_baseline(self, rewards):

        inner_max_len = max(map(len, rewards))

        padded_rewards = np.zeros([len(rewards), inner_max_len], np.float64)

        for i, row in enumerate(rewards):
            for j, val in enumerate(row):
                padded_rewards[i][j] = val

        padded_rewards = padded_rewards.T
        y = np.ma.masked_where(padded_rewards == 0, padded_rewards)
        return np.ma.mean(y, axis=1).filled(0)

    def calculate_advantage(self, rewards, baseline):


        inner_max_len = max(map(len, rewards))

        padded_rewards = np.zeros([len(rewards), inner_max_len], np.float64)

        for i, row in enumerate(rewards):
            for j, val in enumerate(row):
                padded_rewards[i][j] = val

        padded_rewards = padded_rewards.T
        y = np.ma.masked_where(padded_rewards == 0, padded_rewards)
        d = y-baseline.reshape(padded_rewards.shape[0],1)
        res = []
        for col in xrange(d.shape[1]):
            res.append(d[:,col].compressed())
        res = np.concatenate(res)

        res = res.reshape(res.shape[0],1)
        res = (res-np.mean(res))/(np.std(res)+1e-8)

        return res.reshape(res.shape[0],1)

    def calculate_advantages_with_VN(self, rewards, states):
        advantages = np.zeros_like(rewards)
        baseline = np.zeros_like(rewards)
        for i in xrange(len(states)):
            baseline[i] = self.VN.get_value(states[i,:]).flatten()
            advantages[i] = rewards[i]-baseline[i]

        advantages = self.normalize_data(advantages)
        advantages = advantages.reshape(advantages.shape[0],1)
        return advantages, baseline

    def get_rewards(self):
        self.task_measure.pop(0)
        curr_rollout_return = self.calculate_return(self.task_measure)
        curr_rollout_disc_return = self.discount_rewards(curr_rollout_return)

        # return np.asarray([curr_rollout_disc_return[:,0],curr_rollout_disc_return[:,1]]).T
        return curr_rollout_disc_return

    def get_actions(self):
        self.actions.pop()
        return self.actions

    def get_task_errors(self):
        self.task_errors.pop()
        return self.task_errors

    def reset_episode(self):
        self.task_measure[:] = []
        self.task_errors[:]  = []
        self.actions[:]      = []
        self.NN_output[:]    = []

    def reset_batch(self,fisher,advant,baseline):

        self.store_fisher_matrix(fisher)
        self.store_advantages(advant)
        self.store_baseline(baseline)
        self.store_weights()

        self.eval_episode = True

        self.all_unnormalized_returns[:] = []
        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_disc_returns1[:] = []
        self.all_disc_returns2[:] = []
        self.all_actions[:]      = []
        self.all_task_errors[:]  = []
        self.all_task_measure[:] = []


    def get_NN_output_mean(self):
        print "Mean of actions: " + str(abs(np.mean(self.actions,axis=0)))
        # return np.mean(abs(np.mean(self.actions,axis=0)))
        return abs(np.mean(self.actions,axis=0))

    def calculate_Fisher_Matrix(self, input_data, output, loglik):
        with self.g.as_default():
            var_list = tf.trainable_variables()
            pg = tf.gradients(tf.log(self.ff_NN_train),var_list)
            pg = np.asarray(self.sess.run(pg, feed_dict={self.task_error_placeholder:input_data, self.task_dyn_placeholder : output}))
            fisher = []
            eps = 1e-8
            for g in pg:
                g /= self.batch_size
                fisher.append((np.asarray(1/np.square(g.flatten())+eps).reshape(g.shape)))
            return fisher, pg



    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec

    def calculate_learning_rate(self, fisher, pg):
        flatten_fisher = self.flattenVectors(fisher)
        flatten_pg = self.flattenVectors(pg)
        flatten_pg = np.square(flatten_pg).reshape(flatten_pg.shape[0],1)
        flatten_fisher = flatten_fisher.reshape(flatten_fisher.shape[0],1)
        eps = 1e-8
        # kl = 10000000.0
        kl = 0.01
        numerator = eps+np.square(flatten_pg).T.dot(flatten_fisher)

        step_size = np.sqrt(kl/numerator)
        print "STEPSIZE"
        print step_size.flatten()[0]
        return step_size.flatten()[0]

    def policy_search(self, req):
        with self.g.as_default():
            

            #Do an evaluation episode (episode with no noise) every self.eval_episode
            if self.eval_episode:
                self.num_eval_episode += 1
                print "Evaluation episode number "+str(self.num_eval_episode)+" finished!"
                self.sigma = self.get_NN_output_mean()
                # self.sigma[self.sigma>5] = 5
                self.sigma[self.sigma<0.5] = 0.5

                # Bootstrap the variance to avoid too large expolration
                # if self.sigma>3:
                #     self.sigma = 3

                if self.num_eval_episode == 1:
                    self.VN.construct_ValueNet(self.get_task_errors(), self.get_rewards())
                else:
                    self.VN.train(self.get_task_errors(), self.get_rewards())

                self.eval_episode = False
                self.store_NN_output()
                self.reset_episode()
                return PolicySearchResponse(not self.train)


            self.num_train_episode +=1
            print "Training episode number "+str(self.num_train_episode)+" finished!" 

            self.store_NN_output()
            self.store_actions()
            self.store_task_measure()
            self.store_rewards()
            self.store_task_errors()

            if self.num_train_episode % self.batch_size == 0 and self.train:

                print "Updating policy"

                rewards = np.concatenate(self.all_disc_returns)
                task_errors = np.concatenate(self.all_task_errors)
                actions = np.concatenate(self.all_actions)

                # reward1 = np.concatenate(self.all_disc_returns1)
                # reward2 = np.concatenate(self.all_disc_returns2)
                # baseline = self.calculate_reward_baseline(self.all_disc_returns)
                # advantage = self.calculate_advantage(self.all_disc_returns, baseline)
                # baseline = self.calculate_reward_baseline(reward1)
                # advantage1 = self.calculate_advantage(reward1, baseline)
                # baseline = self.calculate_reward_baseline(reward2)
                # advantage2 = self.calculate_advantage(reward2, baseline)
                # advantage = np.asarray([advantage1,advantage2])
                # advantage = advantage.reshape(advantage.shape[0],advantage.shape[1])
                # advantage = np.transpose(advantage)
                # rewards = np.asarray([np.concatenate(self.all_disc_returns1), np.concatenate(self.all_disc_returns2)]).T

                advantage, baseline = self.calculate_advantages_with_VN(rewards, task_errors)

                curr_batch_mean_return = np.mean([self.all_unnormalized_returns[i].sum() for i in xrange(len(self.all_unnormalized_returns))])
                print "Averaged return of the batch is "+ str(curr_batch_mean_return)

                if curr_batch_mean_return<=7000:
                    if curr_batch_mean_return>=self.prev_batch_mean_return:
                        print "Policy improving"
                        print "Reducing learning rate"
                        # self.learning_rate /= 2.0#1.0/abs(curr_batch_mean_return)
                    else:
                        print "Policy not improving"
                        print "Increasing learning rate"
                        # self.learning_rate = 1 # 0.1
                else:
                    print "Policy converged in " +str(self.num_train_episode+self.num_eval_episode)+" episodes!"
                    self.train = False


                if self.train:
                    var_list = tf.trainable_variables()

                    # loglik = tf.multiply(self.task_dyn_placeholder - self.ff_NN_train, self.advant/np.square(self.sigma))
                    loglik = tf.multiply(self.task_dyn_placeholder - self.ff_NN_train, 1/np.square(self.sigma))
                    # loglik = (tf.constant(1.0/pow(self.sigma,2))*(self.task_dyn_placeholder - self.ff_NN_train))

                    # loss = -tf.reduce_mean(loglik) 
                    loss = -tf.reduce_mean(loglik*self.advant) 

                    fisher, pg = self.calculate_Fisher_Matrix(task_errors, actions, loglik)
                    learning_rate = float(self.calculate_learning_rate(fisher, pg))

                    temp = set(tf.global_variables())
                    
                    optimizer = tf.train.AdamOptimizer(learning_rate/self.batch_size)

                    loss_grads = optimizer.compute_gradients(loss, var_list)

                    masked_grad_and_vars = []
                    i = 0
                    for g,v in loss_grads:
                        masked_grad_and_vars.append((tf.multiply(tf.constant(fisher[i]), g), v))
                        i+=1

                    train_op = optimizer.apply_gradients(masked_grad_and_vars)

                    self.sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

                    for i in range(3500):

                        _, advant, ll, loss_ = self.sess.run([train_op, self.advant, loglik, loss],
                                    feed_dict={self.task_error_placeholder : task_errors,
                                               self.task_dyn_placeholder   : actions,
                                               self.advant                 : advantage})
                    self.VN.train(task_errors, rewards, self.batch_size)
                    self.reset_batch(fisher,advant,baseline)
                    self.prev_batch_mean_return = curr_batch_mean_return

            self.reset_episode()

            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default(): 
            self.task_measure.append(req.task_measures)
            self.task_errors.append(req.task_measures)
            # start_time = time.time()
            mean_values = self.sess.run(self.ff_NN_train, feed_dict={self.task_error_placeholder: np.asarray([req.task_measures])})
            # print("--- %s seconds ---" % (time.time() - start_time))
            if self.train and not self.eval_episode:
                task_dynamics = np.random.normal(mean_values, self.sigma).tolist()
                # Low-pass filter the task dynamics to also depend on the previous action such that realizable expolration on
                # the physical system is achieved.  
                task_dynamics = np.multiply(0.5,self.prev_action)+np.multiply(0.5,task_dynamics[0])
                self.prev_action = task_dynamics
            else:
                task_dynamics = mean_values.tolist()

            self.NN_output.append(mean_values.flatten())
            self.actions.append(task_dynamics)
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_search')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass