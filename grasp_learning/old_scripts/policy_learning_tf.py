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
        self.sample_and_rewight_sub_ = rospy.Subscriber('sample_and_rewight', Empty, self.sample_and_rewight)        
        self.NN_params_ = 0
        self.noises = np.zeros((80,0))
        self.returns = []
        self.mu = 0
        self.sigma = 0.7
        self.max_iter_ = 10
        self.burn_in_trials_ = 15
        self.num_rollout = 0
        self.best_rollouts = []
        self.num_inputs = 7
        self.num_outputs = 1
        self.g = tf.Graph()

        self.sess = tf.InteractiveSession(graph=self.g)

        self.joint_placeholder = tf.placeholder(tf.float32, [None, 7])
        self.task_dyn_placeholder = tf.placeholder(tf.float32, [None, 1])

        self.joint_placeholder_test = tf.placeholder(tf.float32, [7])

        self.train_weights = [ ]
        self.eval_weights = [ ]
        self.added_noises = [ ]
        input_data = self.parse_input_data(input_data_file)
        output_data = self.parse_output_data(output_data_file)

        self.noise_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/noise/noises.txt'
        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards.txt'
        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'
        self.noisy_weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/noisy_weights.txt'
        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.state_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/joint_states/joint_states_python.txt'

        self.param_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/param_to_add/param_to_add.txt'

        self.reset_files([self.noise_file_name, self.reward_file_name, self.param_file_name, self.noisy_weights_file_name, self.actions_file_name, self.state_file_name])

        # self.saver = tf.train.Saver()

        self.ff_NN_train, w1_train, w2_train, w3_train = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)
        self.ff_NN_eval, w1_eval, w2_eval, w3_eval = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)

        self.train_weights.append(w1_train)
        self.train_weights.append(w2_train)
        self.train_weights.append(w3_train)

        self.eval_weights.append(w1_eval)
        self.eval_weights.append(w2_eval)
        self.eval_weights.append(w3_eval)

        np.savetxt(self.weights_file_name, [np.append(w1_train.eval(session=self.sess),w2_train.eval(session=self.sess))], delimiter='\t')

        # self.saver(self.sess,'/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/policy/my-model')
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

    def construct_ff_NN(self ,joint_placeholder, task_dyn_placeholder, joints, task_dyn, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 4):
        with self.g.as_default():

            # weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            # biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            # layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)

            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, 1]))
            # biases_2 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_1_outputs, weights_2) + biases_2

            weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)


            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]))
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]))
            layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)

            weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, 1]))
            biases_3 = tf.Variable(tf.zeros([1]))
            logits = tf.matmul(layer_2_outputs, weights_3)

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, task_dyn_placeholder)))
            # error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=task_dyn_placeholder)) 
            # 0.5 * tf.reduce_sum(tf.subtract(logits, task_dyn_placeholder) * tf.subtract(logits, task_dyn_placeholder))

            train_step = tf.train.GradientDescentOptimizer(0.02).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            for i in range(100):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={joint_placeholder: joints,
                                      task_dyn_placeholder: task_dyn})
            # print weights_2.eval(session=self.sess)
            print loss
            return logits, weights_1, weights_2, weights_3

    def sample_noise(self):
        with self.g.as_default():

            noises = []
            # self.sess.run(tf.global_variables_initializer())
            for weight in self.train_weights:
                noises.append(tf.Variable(tf.truncated_normal(weight.get_shape(),stddev=self.sigma)))


            init_new_vars_op = tf.variables_initializer(noises[:])
            self.sess.run(init_new_vars_op)

            self.added_noises.append(noises)
            f_handle = file(self.noise_file_name,'a')
            np.savetxt(f_handle, [np.append(noises[0].eval(session=self.sess),noises[1].eval(session=self.sess))],delimiter='\t')
            f_handle.close()

            return noises

    def set_weights(self,added_weights):
        print "Add noise to weights"
        with self.g.as_default():
            for i in xrange(len(added_weights)):
                new_weight =  self.eval_weights[i].assign(self.train_weights[i])
                new_weight =  tf.assign_add(new_weight,added_weights[i])
                new_weight.eval(session=self.sess)

            f_handle = file(self.noisy_weights_file_name,'a')
            np.savetxt(f_handle, [np.append(self.eval_weights[0].eval(session=self.sess),self.eval_weights[1].eval(session=self.sess))],delimiter='\t')
            f_handle.close()



    def update_params(self, delta_theta):
        print "Updating policy parametres"
        with self.g.as_default():
            for i in xrange(len(delta_theta)):

                new_weight_train =  self.train_weights[i].assign_add(delta_theta[i])
                new_weight_train.eval(session=self.sess)

                self.eval_weights[i].assign(self.train_weights[i])
                self.eval_weights[i].eval(session=self.sess)


            f_handle = file(self.param_file_name,'a')
            np.savetxt(f_handle, [np.append(delta_theta[0],delta_theta[1])], delimiter='\t')
            f_handle.close()

            f_handle = file(self.weights_file_name,'a')
            np.savetxt(f_handle, [np.append(self.train_weights[0].eval(session=self.sess),self.train_weights[1].eval(session=self.sess))], delimiter='\t')
            f_handle.close()




    def imp_sampling_order(self,reward):

        print "Storing trajectory"
        pos = bisect.bisect_left(sorted(self.returns), reward)
        self.returns.append(reward)
        self.best_rollouts.insert(pos, self.num_rollout)

        self.num_rollout+=1

        f_handle = file(self.reward_file_name, 'a')
        np.savetxt(f_handle, [[reward, int(pos), self.sigma]], delimiter='\t')
        f_handle.close()

    def merge_states(self, states):
        merged_states = []
        for state in states:
            merged_states.append(state.state[:])
        return list(itertools.chain.from_iterable(merged_states))


    def policy_search(self, req):
        print "Storing new reward"

        with self.g.as_default():

            self.imp_sampling_order(req.rewards[-1])
            print "Rollout Number = " + str(self.num_rollout)
            print "New reward = " + str(req.rewards[-1])
            # asd = np.asarray(req.states[:])
            
            f_handle = file(self.actions_file_name,'a')
            np.savetxt(f_handle, [req.actions], delimiter='\t')
            f_handle.close()

            f_handle = file(self.state_file_name,'a')
            np.savetxt(f_handle, [self.merge_states(req.states)], delimiter='\t')
            f_handle.close()


            if self.num_rollout >= self.burn_in_trials_:

                numerator_hidded_layer = np.zeros((self.eval_weights[0].get_shape()))
                numerator_output_layer = np.zeros((self.eval_weights[1].get_shape()))

                denominator = 0
                num_iter = self.max_iter_ if len(self.returns) >= 10 else len(self.returns)
                for i in xrange(num_iter):
                    curr_best_rollout = self.best_rollouts[-(i+1)]
                    denominator += self.returns[curr_best_rollout]
                    noise = self.added_noises[curr_best_rollout]
                    numerator_hidded_layer += self.returns[curr_best_rollout]*noise[0].eval(session=self.sess)
                    numerator_output_layer += self.returns[curr_best_rollout]*noise[1].eval(session=self.sess)
                
                new_param_hidden = numerator_hidded_layer/denominator
                new_param_output = numerator_output_layer/denominator

                self.update_params([new_param_hidden,new_param_output])
                self.sigma = sqrt(1/denominator)
                print "new variance is " + str(self.sigma)

            success = True
            return success

    def sample_and_rewight(self, req):
        print "Sampling noise"
        noises = self.sample_noise()
        self.set_weights(noises)

    def handle_query_NN_(self,req):
        with self.g.as_default():
            task_dynamics = self.sess.run(self.ff_NN_eval, feed_dict={self.joint_placeholder: np.asarray([req.joint_angles])})
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_learner_tf')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass