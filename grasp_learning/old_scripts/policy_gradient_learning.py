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
from grasp_learning.srv import QueryNN
from grasp_learning.srv import PolicySearch
from std_msgs.msg import Empty
from copy import deepcopy
from math import sqrt
from math import pow
import tensorflow as tf
import numpy as np

class Policy(object):
    """docstring for Policy"""

    def __init__(self):
        # input_data_file = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/training_data/joint_states/joints_random_episode_1.txt'
        input_data_file = rospy.get_param('~input_training_data', ' ')
        # output_data_file = "/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/training_data/task_dynamics/task_dynamics_random_episode_1.txt"
        output_data_file = rospy.get_param('~output_training_data', ' ')

        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.mu = 0
        self.sigma = 2
        self.num_episode = 0
        self.num_inputs = 7
        self.num_outputs = 1
        self.gamma = 0.99
        self.batch_size = 7
        self.g = tf.Graph()
        self.train = True
        self.learning_rate = 0.02
        self.all_returns = []
        self.all_disc_returns = []

        self.all_actions = []
        self.actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        self.all_task_measure = []

        self.NN_output = []

        self.prev_batch_mean_return = -1e-8

        self.sess = tf.InteractiveSession(graph=self.g)

        self.joint_placeholder = tf.placeholder(tf.float32, [None, 7])
        self.task_dyn_placeholder = tf.placeholder(tf.float32, [None, 1])

        self.advant = tf.placeholder(tf.float32,[None,1])

        self.train_weights = [ ]
        self.eval_weights = [ ]
        input_data = self.parse_input_data(input_data_file)
        output_data = self.parse_output_data(output_data_file)

        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards.txt'
        self.disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards.txt'
        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'
        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.state_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/joint_states/joint_states_python.txt'
        self.baseline_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/baseline.txt'
        self.advantages_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/advantages.txt'
        self.loss_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/losses.txt'
        self.log_likelihood_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/log_likelihood.txt'
        self.NN_output_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/action_dist_mean.txt'
        self.task_measure_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/task_measure.txt'

        self.reset_files([self.weights_file_name, self.log_likelihood_file_name, self.advantages_file_name, self.loss_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.state_file_name, self.disc_reward_file_name, self.NN_output_file_name,
                             self.task_measure_file_name])


        # self.ff_NN_train, w1_train, w2_train, w3_train = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)
        self.ff_NN_train, w1_train, w2_train = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)

        self.train_weights.append(w1_train)
        self.train_weights.append(w2_train)
        # self.train_weights.append(w3_train)
        self.store_weights()
        # np.savetxt(self.weights_file_name, [np.append(w1_train.eval(session=self.sess),w2_train.eval(session=self.sess))], delimiter='\t')

    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()

    def parse_input_data(self, input_file):
        data = []
        temp = []
        i = 0
        with open(input_file, 'rU') as f:
            for line in f:
                if (i==0):
                    i+=1
                    continue
                line = line.split()
                for j in range(len(line)):
                    if (j>2 and j  %2==0):
                        temp.append(line[j])
                data.append(temp)
                temp = []
        return np.asarray(data)

    def parse_output_data(self, output_file):
        data = []
        i=0
        with open(output_file, 'rU') as f:
            for line in f:
                if (i==0):
                    i+=1
                    continue
                line = line.split()
                data.append(line[1])
        return  np.asarray(data)

    def construct_ff_NN(self ,joint_placeholder, task_dyn_placeholder, joints, task_dyn, HIDDEN_UNITS_L1 = 15, HIDDEN_UNITS_L2 = 3):
        with self.g.as_default():

            weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)

            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, 1]))
            biases_2 = tf.Variable(tf.zeros([1]))
            logits = tf.matmul(layer_1_outputs, weights_2) + biases_2

            # weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            # biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            # layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)


            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]))
            # biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]))
            # layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)

            # weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, 1]))
            # biases_3 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_2_outputs, weights_3)

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, task_dyn_placeholder)))
            # error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=task_dyn_placeholder)) 
            # 0.5 * tf.reduce_sum(tf.subtract(logits, task_dyn_placeholder) * tf.subtract(logits, task_dyn_placeholder))

            train_step = tf.train.AdamOptimizer(0.02).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            for i in range(1000):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={joint_placeholder: joints,
                                      task_dyn_placeholder: task_dyn})

            print "Network trained"

            return logits, weights_1, weights_2

            # return logits, weights_1, weights_2, weights_3

    def store_weights(self):

        var = np.array([])
        for trainable_variable in tf.trainable_variables():
            var = np.append(var, trainable_variable.eval(session=self.sess))

        f_handle = file(self.weights_file_name,'a')
        np.savetxt(f_handle, [var] , delimiter='\t')
        f_handle.close()


    def store_rewards(self):

        print "Storing reward"

        self.task_measure.pop(0)
        curr_rollout_return = self.calculate_return(self.task_measure)
        self.all_returns.append(curr_rollout_return)

        curr_rollout_disc_return = self.discount_rewards(curr_rollout_return) 
        self.all_disc_returns.append(curr_rollout_disc_return)
        
        f_handle = file(self.reward_file_name,'a')
        np.savetxt(f_handle, [curr_rollout_return], delimiter='\t')
        f_handle.close()

        f_handle = file(self.disc_reward_file_name,'a')
        np.savetxt(f_handle, [curr_rollout_disc_return], delimiter='\t')
        f_handle.close()

        self.task_measure[:] = []

    def store_actions(self):
        print "Storing actions"

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))
        f_handle = file(self.actions_file_name,'a')
        np.savetxt(f_handle, [self.actions], delimiter='\t')
        f_handle.close()

        self.actions[:] = []



    def store_states(self):
        print "Storing states"

        self.states.pop()
        self.all_states.append(np.asarray(self.states))
        
        f_handle = file(self.state_file_name,'a')
        np.savetxt(f_handle, [list(itertools.chain.from_iterable(self.states))], delimiter='\t')
        f_handle.close()

        self.states[:] = []

    def store_baseline_and_advantages(self, baseline, advantages):
        print "Storing baseline and advantages"

        f_handle = file(self.baseline_file_name,'a')
        np.savetxt(f_handle, [baseline], delimiter='\t')
        f_handle.close()

        f_handle = file(self.advantages_file_name,'a')
        np.savetxt(f_handle, [advantages], delimiter='\t')
        f_handle.close()

    def store_loss_function(self, losses):
        print "Storing losses"
        f_handle = file(self.loss_file_name,'a')
        np.savetxt(f_handle, [losses], delimiter='\t')
        f_handle.close()

    def store_log_likelihood(self, log_likelihood):
        print "Storing log likelohood"
        f_handle = file(self.log_likelihood_file_name,'a')
        np.savetxt(f_handle, [log_likelihood], delimiter='\t')
        f_handle.close()

    def store_NN_output(self):
        print "Storing neural network output"
        f_handle = file(self.NN_output_file_name,'a')
        np.savetxt(f_handle, [self.NN_output], delimiter='\t')
        f_handle.close()

        self.NN_output[:] = []

    def store_task_measure(self):
        print "Storing task measure"
        f_handle = file(self.task_measure_file_name,'a')
        np.savetxt(f_handle, [self.task_measure], delimiter='\t')
        f_handle.close()

    def discount_rewards(self, reward):
        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add
        # print discounted_r
        # discounted_r -= np.mean(discounted_r)

        # print discounted_r

        # discounted_r /= (np.std(discounted_r)+1e-8)
        # print discounted_r

        return discounted_r


    def calculate_return(self, curr_reward):

        dist_square = np.square(np.asarray(curr_reward))
        alpha = 1e-15
        rollout_return = -20*dist_square - 0.4*np.log(dist_square+alpha)

        return rollout_return

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
        # res = res.reshape(res.shape[0],1)
        return res.reshape(res.shape[0],1)

    def policy_search(self, req):
        self.num_episode +=1
        print "Episode number "+str(self.num_episode)+" finished!" 

        with self.g.as_default():
            
            self.store_actions()
            self.store_task_measure()
            self.store_rewards()
            self.store_states()
            self.store_NN_output()

            if self.num_episode % self.batch_size == 0 and self.train:
                print "Updating policy"

             

                batch_lower_idx = self.num_episode-self.batch_size
                batch_upper_idx = self.num_episode

                #Normalize the rewards
             
                rewards = np.concatenate((self.all_disc_returns[batch_lower_idx:batch_upper_idx]), axis=0)
                states = np.concatenate((self.all_states[batch_lower_idx : batch_upper_idx]), axis=0)
                actions = np.concatenate((self.all_actions[batch_lower_idx : batch_upper_idx]), axis=0)

                actions= actions.reshape(actions.shape[0],1)
                rewards= rewards.reshape(rewards.shape[0],1)
                baseline = self.calculate_reward_baseline(self.all_disc_returns[batch_lower_idx:batch_upper_idx])

                advantage = self.calculate_advantage(self.all_disc_returns[batch_lower_idx:batch_upper_idx], baseline)

                curr_batch_mean_return = np.mean([ self.all_returns[i].sum() for i in range(batch_lower_idx,batch_upper_idx)])

                print "mean of batch rewards is "+ str(curr_batch_mean_return)

                if curr_batch_mean_return<=2500:
                    if curr_batch_mean_return>=self.prev_batch_mean_return:
                        print "Policy improving"
                        print "Reducing learning rate and variance"
                        self.learning_rate = 0.005#1.0/abs(curr_batch_mean_return)
                        self.sigma = 0.1
                    else:
                        print "Policy not improving"
                        print "Increasing learning rate and variance"
                        self.learning_rate = 0.01
                        self.sigma = 2
                else:
                    self.train = False

                self.prev_batch_mean_return = curr_batch_mean_return

                # if batch_mean_return > -100:
                #     self.train = False
                # else:
                #     self.train = True
                # if not self.train:
                #     print "Policy seems to have converged"
                if self.train:
                    loglik = (tf.constant(1.0/pow(self.sigma,2))*(self.task_dyn_placeholder - self.ff_NN_train))

                    temp = loglik * self.advant

                    loss = -tf.reduce_mean(temp) 

                    temp = set(tf.global_variables())
                    train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                    self.sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
                    # sess.run(init_op)

                    # self.sess.run(tf.global_variables_initializer())
                    # self.sess.run(loss.initilizer)
                    # self.sess.run(train_step.initilizer)
                    for i in range(20):

                        _, error, ll, advant, actions, states = self.sess.run([train_step, loss, loglik, self.advant, self.task_dyn_placeholder, self.joint_placeholder],
                                    feed_dict={self.joint_placeholder : states,
                                              self.task_dyn_placeholder : actions,
                                              self.advant : advantage})

                    self.store_log_likelihood(ll)
                    self.store_loss_function(error)
                    self.store_baseline_and_advantages(baseline, advant)
                    self.store_weights()
                    # print error

            success = True
            return success

    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures[0]);
            self.states.append(req.joint_angles)
            mean = self.sess.run(self.ff_NN_train, feed_dict={self.joint_placeholder: np.asarray([req.joint_angles])})
            if self.train:
                task_dynamics = np.random.normal(mean, self.sigma)
            else:
                task_dynamics = mean
            self.NN_output.append(mean)
            self.actions.append(task_dynamics)

            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_gradient_learner')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass