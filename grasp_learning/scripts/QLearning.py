#!/usr/bin/env python
import roslib

import rospy
import sys
import bisect
import itertools
from utils import *
from value_function import *
from grasp_learning.srv import QueryNN
from grasp_learning.srv import *
from std_msgs.msg import Empty
from copy import deepcopy
from math import sqrt
from math import pow
import numpy as np
import os
from std_srvs.srv import Empty
import csv
import random
from FileHandler import *
from itertools import product
from replay_buffer import ReplayBuffer
from QNetwork import QNetwork
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

RANDOM_SEED = 1234
BUFFER_SIZE = 10000

class QLearning(object):

    def __init__(self):

        load_example_model = rospy.get_param('~load_model', ' ')
        self.model_name = rospy.get_param('~model_name', ' ')
        self.num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        hidden_layers_sizes  =  rospy.get_param('~hidden_units', ' ')
        self.lr = rospy.get_param('~learning_rate', '1')
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.gamma = rospy.get_param('~discount_factor', '0.99') #0.99 #TSV: testing a much more local approach
        self.relative_path = rospy.get_param('~relative_path', ' ')
        self.min_dist_norm = rospy.get_param('~min_dist_norm', '0.04')
        self.epsilon = rospy.get_param('~epsilon', '0.1')
        self.num_actions = rospy.get_param('~num_actions', '1')
        self.num_samples = rospy.get_param('~num_samples', '10')

        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.num_trial=1

        self.train = True
        self.eval_episode = True
        self.NN_output = np.zeros(num_outputs)
        self.all_outputs = []

        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        
        self.all_actions = []
        self.actions = []
        self.eval_actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        # This list holds the natural policy gradients which is composed of the inverse of the fisher
        # matrix mulitpled by the policy gradient. For more info check http://www.scholarpedia.org/article/Policy_gradient_methods
        self.prev_eval_mean_return = 0


        self.set_actions()
        self.set_services()
        self.set_bookkeeping_files()

        self.num_outputs = len(self.possible_actions)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        self.QNetwork = QNetwork(self.gamma, self.relative_path, self.model_name)
        self.TargetQNetwork = QNetwork(self.gamma, self.relative_path, self.model_name)

        self.QNetwork.create_tensorflow_graph(self.num_inputs, self.num_outputs, num_rewards, hidden_layers_sizes,load_example_model)
        self.TargetQNetwork.create_tensorflow_graph(self.num_inputs, self.num_outputs, num_rewards, hidden_layers_sizes,load_example_model)
        self.update_targetNetwork()
        self.store_weights()


    def update_targetNetwork(self):
        parameters = self.QNetwork.get_trainable_variables()
        self.TargetQNetwork.set_new_parameters(parameters, 0.9)

    def set_services(self):
        rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        rospy.Service('policy_Search', PolicySearch, self.policy_search)
        self.start_demo = rospy.ServiceProxy('/demo_learning/start_demo', Empty)

    def set_actions(self):
        possible_actions = [-0.5, 0.5]
        # possible_actions = [-5,0,5]
        self.possible_actions = []
        for roll in product(possible_actions, repeat = self.num_actions):
            self.possible_actions.append(list(roll))

    def set_bookkeeping_files(self):

        dir_path = self.relative_path+'data/trial_'+str(self.num_trial)
        create_directory(dir_path)

        self.param_file = create_file(dir_path+'/params.txt')

        self.reward_file_name = create_file(dir_path+'/rewards.txt')
        self.disc_reward_file_name = create_file(dir_path+'/discounted_rewards.txt')

        self.actions_file_name = create_file(dir_path+'/actions.txt')
        self.eval_action_file_name = create_file(dir_path+'/eval_actions.txt')
        
        self.explored_states_file_name = create_file(dir_path+'/explored_states.txt')

        self.evaluated_states_file_name = create_file(dir_path+'/evaluated_states.txt')

        self.task_measure_file_name = create_file(dir_path+'/task_measure.txt')

        self.neural_network_param_file_name = create_file(dir_path+'/weights.txt')

        self.loss_file_name = create_file(dir_path+'/losses.txt')

        self.neural_network_output_file_name = create_file(dir_path+'/output.txt')

        self.eval_rewards_file_name = create_file(dir_path+'/eval_rewards.m')
        self.tdyn_file_name = create_file(dir_path+'/tdyn.m')

    def store_weights(self):

        var = self.QNetwork.get_network_parameters()
        save_data_to_file(self.neural_network_param_file_name, [var])

    def store_episode_data_to_file(self, episode_data):
        for key in episode_data:
            save_matrix_data_to_file(key, episode_data[key])

    def store_batch_data_to_file(self, batch_data):
        for key in batch_data:
            save_vector_data_to_file(key, batch_data[key])

        self.store_weights()

    def get_undiscounted_reward(self):
        self.task_measure.pop(0)
        return self.calculate_return(self.task_measure)


    def reset_episode(self):
        self.task_measure[:] = []
        self.states[:] = []
        self.actions[:] = []
        self.eval_actions[:] = []
        self.all_outputs[:] = []

        #Not all data are stored after an evaluation episode
    def reset_eval_episode(self, curr_eval_return):
        self.prev_eval_mean_return = curr_eval_return
        self.eval_episode = False
        self.reset_episode()

        # Reset the data belonging to a batch, i.e. empty the lists storing all data used for that particular batch
    def reset_batch(self):

        self.all_returns[:]      = []
        self.all_actions[:]      = []
        self.all_states[:]  = []
        if self.num_train_episode % self.batch_size==0:
            self.eval_episode = True

        #This function calculates the reward recieved from doing an action. The reward is calculated as
        # total reward = -w1*dist+w2*exp(-w3*dist) where dist is the vector distance from the current point
        # to the constraint, i.e. the task error.
    def calculate_return(self, curr_reward):

        squared_points = np.square(np.asarray(curr_reward))
        dist_abs = np.abs(np.asarray(curr_reward))

        rollout_return = 0
        
    	dist = np.sqrt(np.sum(squared_points,axis=1))
    	dist_l1 = np.sum(dist_abs,axis=1)
        dist_square = np.sum(squared_points,axis=1)
        alpha = 0.2#1e-17
        # alpha = 1e-10
        # rollout_return = -(5*dist+np.log(alpha+dist_square))
        # rollout_return = -(np.sqrt(np.square(dist)+np.square(alpha))-alpha)
        rollout_return = -np.log(dist_square)
        return rollout_return

    def compute_episode_data(self):

        self.all_actions.append(np.asarray(self.actions))

        self.episode_reward = self.calculate_return(self.states)
        self.all_returns.append(self.episode_reward)
        
        self.all_states.append(np.asarray(self.states))
        self.num_states = len(self.states)-1

    def add_training_data_to_replay(self):
        for i in xrange(self.num_states):
            self.replay_buffer.add(np.reshape(self.states[i],(1,self.num_inputs)), self.actions[i], self.episode_reward[i], 0,
                                   np.reshape(self.states[i+1],(1,self.num_inputs)))

        self.replay_buffer.add(np.reshape(self.states[self.num_states],(1,self.num_inputs)), self.actions[self.num_states], self.episode_reward[self.num_states], 1, 0)

    def check_convergence(self):
        dist_norm = np.linalg.norm(self.states[-1])
        print "Dist norm ", dist_norm
        if dist_norm < self.min_dist_norm:
            print "Policy Converged"
            self.train = False
        else:
            self.train = True

    def get_training_data(self):
        states, actions, rewards, terminal, nextstate = self.replay_buffer.sample_batch(self.num_samples)

        Qtarget = []
        for i in xrange(len(rewards)):
            action = self.QNetwork.get_best_action(states[i])
            Qtarget.append(self.TargetQNetwork.calculate_target_Q(action, rewards[i],terminal[i],nextstate[i]))

        states = np.reshape(states,(len(states),self.num_inputs))

        return states, actions, Qtarget

    def get_episode_data(self):
        episode_data = {}

        if self.eval_episode == True:
            episode_data[self.eval_action_file_name] = vec_2_mat(self.eval_actions)
            episode_data[self.evaluated_states_file_name] = self.states
        else:
            episode_data[self.evaluated_states_file_name] = self.states
            episode_data[self.reward_file_name] = vec_2_mat(self.episode_reward)

        episode_data[self.neural_network_output_file_name] = self.all_outputs
        return episode_data

    def converged_episode(self):
        #clean up output file
        self.num_eval_episode += 1
        print "Converged policy  "+str(self.num_eval_episode)+" finished!"
        self.reset_eval_episode(self.prev_eval_mean_return)
        return PolicySearchResponse(not self.train)


    def evaluation_episode(self):
        self.num_eval_episode += 1
        print "Evaluation episode number "+str(self.num_eval_episode)+" finished!"
        # Calculates the mean return from the evaluation episode 
        curr_eval_return = self.get_undiscounted_reward().sum()
        print "Average return from evaluation is " + str(curr_eval_return)
        # The difference between the returns of the current and previous evaluation episode
        diff = curr_eval_return-self.prev_eval_mean_return

        # If the difference is positive meaning that the policy is improving than we increase the learning rate.
        if diff>0:
            # self.kl += 0.01
            print "Policy improved by", diff
        else:
            # self.kl = 0.01 
            print "Policy got worse by", diff

        print "Final task error " + str(self.states[-1])
        self.check_convergence()

        # eval_episode_data = self.get_episode_data()

        # self.store_episode_data_to_file(eval_episode_data)

        self.reset_eval_episode(curr_eval_return)

    def training_episode(self):
        # First we compute the necessary data based on one episode
        self.compute_episode_data()
        self.add_training_data_to_replay()
        # Here we store the data computed for that episode
        episode_data = self.get_episode_data()

        self.store_episode_data_to_file(episode_data)

        self.num_train_episode +=1
        print "Training episode number "+str(self.num_train_episode)+" finished with a reward of " + str(self.episode_reward.sum())# + " and final task error " +  str(self.states[-1]) 

        # The policy is only updated if the number of training episodes match the batch size
        # if self.num_train_episode % self.batch_size == 0 and self.train:
        if self.train:
            self.update_policy()

        self.reset_episode()

    def reduce_exploration(self):
        self.epsilon -= 1.0/1000
        if self.epsilon <0.05:
            self.epsilon = 0.05
        print "Epsilon " + str(self.epsilon)


    def plot_policy(self):
        if self.num_eval_episode == 0:
            print "HEEJ"
            plt.ion()
            self.f = plt.figure(dpi=200)
            self.ax = self.f.gca()
            print self.ax
            self.f.show()

        states = discretize_data(-1,1,50)

        actions = []
        for state in states:
            actions.append(self.possible_actions[self.QNetwork.get_best_action(np.reshape(state,(1,1)))])

        actions = [item for sublist in actions for item in sublist]
        # plt.cla()
        self.ax.clear()
        # ax = self.f.gca()

        self.ax.bar(states, actions)
        self.f.canvas.draw()
        # self.f.canvas.flush_events()
        # plt.figure(dpi=200)
        # plt.bar(states, actions)
        # plt.draw()
        # plt.show()
        # plot_policy(states,actions)

    def update_policy(self):

        print "Updating policy"

        #Concatenating all relevant data for each training episode into one 1D vector
        states, actions, Qtarget = self.get_training_data()

        # Reshape the states to be the same shape as their individual placeholder
        # states = states.reshape(states.shape[0],self.state_placeholder.shape[1])

        # Calculate the mean of all returns for the batch. It is only done to see if the return between batches increases
        curr_batch_mean_return = np.mean([ret.sum() for ret in self.all_returns])

        # print "mean of batch rewards is "+ str(curr_batch_mean_return)

        # As long as the policy has not converged or the episode is an evaluation episode then update the policy

        # Load all batch data into a dictionary
        unum = int (self.num_train_episode / self.batch_size)

        # run i numbers of gradient descent updates
        # setup run options for profiling
        error = self.QNetwork.train(states, Qtarget, actions, self.lr)        
        #add summaries for this training run

        # This dictionary holds all the batch data for one batch. 
        batch_dic = {}
        batch_dic[self.loss_file_name] = [error]
        # batch_dic[self.loss_grads_file_name] = loss_grads

        # Function that stores the batch data in corresponding files
        # self.store_batch_data_to_file(batch_dic)
        self.store_weights()
        self.reset_batch()
        self.reduce_exploration()
        self.update_targetNetwork()

    # This function updates the policy according to the policy gradient. 
    def policy_search(self, req):
            
        #self.plot_policy()
        if self.train==False:
            self.converged_episode()
            return PolicySearchResponse(True)

        # Do an evaluation episode (episode with no noise) every self.eval_episode
        elif self.eval_episode:
            self.evaluation_episode()
            return PolicySearchResponse(not self.train)

        else:

            self.training_episode()
            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        
        self.task_measure.append(req.task_measures)
        self.states.append(list(req.task_measures))
        Qvalues = self.QNetwork.predict(np.array([req.task_measures]))#   self.sess.run(self.QNetwork, feed_dict)
        self.all_outputs.append(list(Qvalues.flatten()))
        if self.train and not self.eval_episode:
            if np.random.rand(1) < self.epsilon:
                idx = random.randint(0, self.num_outputs-1)
            else:
                idx = np.argmax(Qvalues)

            task_dynamics=np.asarray(self.possible_actions[idx])
        else:
            idx = np.argmax(Qvalues)
            task_dynamics=np.asarray(self.possible_actions[idx])
            self.eval_actions.append(task_dynamics[0])

        # Store the output from the neural network which is the action with no noise added to it
        self.actions.append(idx)
        return  np.asarray(task_dynamics)


    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('QLearning')
        QLearning = QLearning()
        QLearning.main()
    except rospy.ROSInterruptException:
        pass47
