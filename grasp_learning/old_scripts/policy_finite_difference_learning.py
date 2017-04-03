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
        input_data_file = rospy.get_param('~input_training_data', ' ')
        output_data_file = rospy.get_param('~output_training_data', ' ')

        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)
        self.sample_and_rewight_sub_ = rospy.Subscriber('sample_and_rewight', Empty, self.sample_and_rewight)        

        self.sigma = 0.09

        self.num_episode = 0
        self.num_inputs = 7
        self.num_outputs = 1
        self.gamma = 0.7
        self.batch_size = 5
        self.g = tf.Graph()
        self.train = True
        self.evaluation_episode = True
        self.training_complete = False
        self.learning_rate = 0.001#1
        self.num_noisy_episodes = 0
        self.all_param_noises = [ ]


        self.all_noisy_returns = []
        self.all_noisy_disc_returns = []

        self.all_evaluated_returns = []
        self.all_evaluated_disc_returns = []

        self.all_actions = []
        self.actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        self.all_task_measure = []

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

        self.noisy_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/noisy_rewards.txt'
        self.noisy_disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/noisy_discounted_rewards.txt'

        self.noise_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/noise/noises.txt'

        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'
        self.noisy_weight_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/noisy_weights.txt'

        self.added_params_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/added_weights.txt'

        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'

        self.state_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/joint_states/joint_states_python.txt'

        self.gradient_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients/gradients.txt'


        self.reset_files([self.noise_file_name, self.noisy_reward_file_name, self.noisy_disc_reward_file_name, self.added_params_file_name, self.gradient_file_name, self.weights_file_name, self.noisy_weight_file_name,self.reward_file_name, self.actions_file_name, self.state_file_name, self.disc_reward_file_name])

        # self.saver = tf.train.Saver()

        self.ff_NN_train, w1_train, w2_train, w3_train, self.num_train_var = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)
        self.ff_NN_eval, w1_eval, w2_eval, w3_eval, _ = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data)

        self.train_weights.append(w1_train)
        self.train_weights.append(w2_train)
        self.train_weights.append(w3_train)

        self.eval_weights.append(w1_eval)
        self.eval_weights.append(w2_eval)
        self.eval_weights.append(w3_eval)

        np.savetxt(self.weights_file_name, [np.append([np.append(w1_train.eval(session=self.sess), w2_train.eval(session=self.sess))],w3_train.eval(session=self.sess)) ] , delimiter='\t')

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

    def construct_ff_NN(self ,joint_placeholder, task_dyn_placeholder, joints, task_dyn, INPUTS = 7, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 3):
        with self.g.as_default():

            # weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            # biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            # layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)

            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, 1]))
            # biases_2 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_1_outputs, weights_2) + biases_2

            weights_1 = tf.Variable(tf.truncated_normal([INPUTS, HIDDEN_UNITS_L1]))
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

            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            for i in range(1500):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={joint_placeholder: joints,
                                      task_dyn_placeholder: task_dyn})
            # print weights_2.eval(session=self.sess)
            print loss

            num_trainable_var = INPUTS*HIDDEN_UNITS_L1+HIDDEN_UNITS_L1*HIDDEN_UNITS_L2+HIDDEN_UNITS_L2*1

            return logits, weights_1, weights_2, weights_3, num_trainable_var


    def sample_param_noise(self, num_params = 0):
        with self.g.as_default():

            noises = []

            # Only vary a subset of all weights
            if num_params>0 and num_params<=self.num_train_var:
                params_2_vary = np.sort(np.random.choice(self.num_train_var, num_params, replace=False))
                print params_2_vary
                num_var = 0
                for weight in self.train_weights:
                    var = np.zeros(weight.get_shape())
                    temp = var.flat[:]
                    print "Outer Loop" + str(weight.get_shape())
                    for param in xrange(len(params_2_vary)):
                        if params_2_vary[param]-num_var<temp.size:
                            try:
                                var[np.unravel_index(params_2_vary[param]-num_var, var.shape)] = np.random.normal(0, self.sigma)
                            except ValueError:
                                print params_2_vary

                                print params_2_vary[param]
                                print num_var
                                print params_2_vary[param]-num_var
                                print temp.size
                                print  var.shape

                                print np.unravel_index(params_2_vary[param]-num_var, var.shape)

                        else:
                            if param>0:
                                params_2_vary = params_2_vary[param:]
                            break
                    num_var += temp.size
                    noise = tf.Variable(var,dtype=tf.float32)
                    noises.append(noise)



            # Vary all the weights
            else:
                for weight in self.train_weights:
                    noises.append(tf.Variable(tf.truncated_normal(weight.get_shape(),stddev=self.sigma)))



            init_new_vars_op = tf.variables_initializer(noises[:])
            self.sess.run(init_new_vars_op)

            temp = [np.append([np.append(noises[0].eval(session=self.sess), noises[1].eval(session=self.sess))],noises[2].eval(session=self.sess)) ]
            self.all_param_noises.append(temp)

            f_handle = file(self.noise_file_name,'a')
            np.savetxt(f_handle, temp, delimiter='\t')
            f_handle.close()

            return noises


    def set_weights(self,added_weights):
        # print "Add noise to weights"
        with self.g.as_default():
            for i in xrange(len(added_weights)):
                new_weight =  self.eval_weights[i].assign(self.train_weights[i])
                new_weight =  tf.assign_add(new_weight,added_weights[i])
                new_weight.eval(session=self.sess)

            # f_handle = file(self.noisy_weights_file_name,'a')
            # np.savetxt(f_handle, [np.append(self.eval_weights[0].eval(session=self.sess),self.eval_weights[1].eval(session=self.sess))],delimiter='\t')
            # f_handle.close()

    def set_default_params(self):
        with self.g.as_default():
            for i in xrange(len(self.eval_weights)):
                self.eval_weights[i].assign(self.train_weights[i])
                self.eval_weights[i].eval(session=self.sess)


    def sample_and_rewight(self, req):

        if self.evaluation_episode or self.training_complete:
            print "Evaluation episode. No noise added to the parameters"
            self.set_default_params()
        else:
            print "Exploring episode. Noise added to the parameters"
            noises = self.sample_param_noise()
            self.set_weights(noises)


    def update_params(self, gradient, alpha):
        # print "Updating policy parametres"
        with self.g.as_default():
            delta_param=[]
            for i in xrange(len(self.train_weights)):

                upper_idx = self.train_weights[i].get_shape()
                if i>0:
                    lower_idx = self.train_weights[i-1].get_shape()
                    var = tf.Variable(alpha*gradient[lower_idx[0]*lower_idx[1]:lower_idx[0]*lower_idx[1]+upper_idx[0]*upper_idx[1]].reshape(upper_idx),dtype=tf.float32)
                    new_weight_train =  self.train_weights[i].assign_add(var)
                else:
                    var = tf.Variable(alpha*gradient[0:upper_idx[0]*upper_idx[1]].reshape(upper_idx),dtype=tf.float32)
                    new_weight_train =  self.train_weights[i].assign_add(var)

                self.sess.run(tf.global_variables_initializer())

                delta_param.append(var)

                new_weight_train.eval(session=self.sess)
                self.eval_weights[i].assign(self.train_weights[i])
                self.eval_weights[i].eval(session=self.sess)

            f_handle = file(self.added_params_file_name,'a')
            np.savetxt(f_handle, [np.append([np.append(delta_param[0].eval(session=self.sess), delta_param[1].eval(session=self.sess))],delta_param[2].eval(session=self.sess)) ], delimiter='\t')
            f_handle.close()


    def discount_rewards(self, reward):
        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        # discounted_r -= np.mean(discounted_r)
        # discounted_r /= (np.std(discounted_r)+1e-7)

        return discounted_r


    def calculate_return(self, curr_reward):

        rollout_return = -2*np.square(np.asarray(curr_reward))
        return rollout_return

    def store_rewards(self, eval_episode):


        curr_rollout_return = self.calculate_return(self.task_measure)
        curr_rollout_disc_return = self.discount_rewards(curr_rollout_return) 
        
        print "Sum of discounted rewards: " + str(curr_rollout_disc_return.sum())

        if eval_episode:

            # print "Storing reward for evaluation episode"

            self.all_evaluated_returns.append(curr_rollout_return)
            self.all_evaluated_disc_returns.append(curr_rollout_disc_return)
            
            f_handle = file(self.reward_file_name,'a')
            np.savetxt(f_handle, [curr_rollout_return], delimiter='\t')
            f_handle.close()

            f_handle = file(self.disc_reward_file_name,'a')
            np.savetxt(f_handle, [curr_rollout_disc_return], delimiter='\t')
            f_handle.close()

        else:

            # print "Store reward for noisy evauluation"
            self.all_noisy_returns.append(curr_rollout_return)
            self.all_noisy_disc_returns.append(curr_rollout_disc_return)

            f_handle = file(self.noisy_reward_file_name,'a')
            np.savetxt(f_handle, [curr_rollout_return], delimiter='\t')
            f_handle.close()

            f_handle = file(self.noisy_disc_reward_file_name,'a')
            np.savetxt(f_handle, [curr_rollout_disc_return], delimiter='\t')
            f_handle.close()
        
        self.task_measure[:] = []

    def store_actions(self):

        # print "Storing actions"

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))
        f_handle = file(self.actions_file_name,'a')
        np.savetxt(f_handle, [self.actions], delimiter='\t')
        f_handle.close()

        self.actions[:] = []

    def store_states(self):

        # print "Storing states"

        self.all_states.append(np.asarray(self.states))
        
        f_handle = file(self.state_file_name,'a')
        np.savetxt(f_handle, [list(itertools.chain.from_iterable(self.states))], delimiter='\t')
        f_handle.close()

        self.states[:] = []

    def store_params(self, eval_episode):

        if eval_episode:

            # print "Storing evaluation parameters"
            f_handle = file(self.weights_file_name,'a')
            np.savetxt(f_handle, [np.append([np.append(self.train_weights[0].eval(session=self.sess), self.train_weights[1].eval(session=self.sess))],self.train_weights[2].eval(session=self.sess)) ] , delimiter='\t')
            f_handle.close()

        else:
            # print "Storing noisy parameters"
            f_handle = file(self.noisy_weight_file_name,'a')
            np.savetxt(f_handle, [np.append([np.append(self.eval_weights[0].eval(session=self.sess), self.eval_weights[1].eval(session=self.sess))],self.eval_weights[2].eval(session=self.sess)) ] , delimiter='\t')
            f_handle.close()



    def calculate_gradient(self, num_noisy_episodes):
        # print "Calculating gradients"
        batch_lower_idx = num_noisy_episodes-(self.batch_size-1)
        batch_upper_idx = num_noisy_episodes

        batch_ref_return = sum(self.all_evaluated_disc_returns[-1])
        batch_noisy_returns = np.mat([self.all_noisy_disc_returns[i].sum() for i in range(batch_lower_idx,batch_upper_idx)])

        delta_returns = batch_noisy_returns-batch_ref_return

        print delta_returns

        batch_noises = np.mat(np.vstack(self.all_param_noises[batch_lower_idx:batch_upper_idx]))

        # nonzero_idx = np.where(batch_noises.any(axis=1))[0]
        # batch_noises = batch_noises[nonzero_idx]

        gradient = np.linalg.inv(batch_noises.T*batch_noises)*batch_noises.T*delta_returns.T

        # gradient = np.zero(self.num_train_var)
        # np.put(gradient, nonzero_idx, temp)

        f_handle = file(self.gradient_file_name,'a')
        np.savetxt(f_handle, [gradient], delimiter='\t')
        f_handle.close()


        return gradient

    def policy_search(self, req):
        self.num_episode +=1
        if not self.evaluation_episode:
            self.num_noisy_episodes +=1
        print "Episode number "+str(self.num_episode)+" finished!" 

        with self.g.as_default():
            
            self.store_actions()
            self.store_rewards(self.evaluation_episode)
            self.store_states()
            self.store_params(self.evaluation_episode)


            # resp = PolicySearchResponse()

            if self.training_complete:
                # resp.converged = True
                return True


            if self.evaluation_episode:
                mean_ret_eval_episode = np.mean(self.all_evaluated_returns[-1].sum())
                print "Mean of evaluation run: " + str(mean_ret_eval_episode)
                if mean_ret_eval_episode> -25:
                    print "Evaluation policy has converged. No need for further training"
                    self.evaluation_episode = True
                    self.training_complete = True
                else:
                    self.evaluation_episode = False
                    # if abs(mean_ret_eval_episode)/100.0<0.7:
                    #     self.sigma= abs(mean_ret_eval_episode)/100.0
                    # else:
                    #     self.sigma=0.09
                print "Current standard deviation is " +str(self.sigma)
            if not self.evaluation_episode and self.num_episode % self.batch_size == 0:
                print "Updating policy"

                gradients = self.calculate_gradient(self.num_noisy_episodes)
                self.update_params(gradients, self.learning_rate)

                self.evaluation_episode = True

            # resp.converged = False
            return True

    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures[0]);
            self.states.append(req.joint_angles)
            task_dynamics = self.sess.run(self.ff_NN_eval, feed_dict={self.joint_placeholder: np.asarray([req.joint_angles])})
            self.actions.append(task_dynamics)
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_finite_difference_learner')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass