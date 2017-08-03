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
from std_msgs.msg import Empty as emp
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
from std_msgs.msg import String
import time
class PISquare(object):

    def __init__(self):

        self.num_inputs = rospy.get_param('~num_inputs', ' ')
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.relative_path = rospy.get_param('~relative_path', ' ')
        self.max_num_trials = rospy.get_param('~max_num_trials', '1')
        self.min_dist_norm = rospy.get_param('~min_dist_norm', '0.04')
        self.num_samples = rospy.get_param('~num_samples', '10')
        self.num_parameters = rospy.get_param('~num_params', '50')
        self.Lambda = 1
        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.c = 10
        self.train = True
        self.eval_episode = False
        self.demonstration = True

        self.start_time = 0
        self.current_time = 0

        self.discretized_time = []

        self.noise = []
        
        self.policy_params = []

        self.all_returns = []
        
        self.all_noises = []
        
        self.states = []
        self.all_states = []

        self.time_indexes = []
        self.all_time_indexes = []

        self.prev_eval_mean_return = 0

        self.init = True

        self.num_trial = 0
        
        self.set_services()
        self.set_covariance_matrix()
        self.set_dist_mean()
        self.set_bookkeeping_files()
        # self.store_weights()

    def set_services(self):
        rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        rospy.Service('policy_Search', PolicySearch, self.policy_search)
        self.start_demo = rospy.ServiceProxy('/demo_learning/start_demo', Empty)
        rospy.Subscriber("/run_new_episode", emp, self.sample_noise)


    def set_covariance_matrix(self):
        A2 = np.zeros([self.num_parameters+2,self.num_parameters])
        for r in xrange(self.num_parameters+2):
            for c in xrange(self.num_parameters):
                if r == c:
                    A2[r,c] = 1
                if (r-1) == c:
                    A2[r,c] = -2
                if (r-2) == c:
                    A2[r,c] = 1

        self.R = np.dot(np.transpose(A2),A2)
        self.covar_mat = np.linalg.inv(self.R)

    def set_dist_mean(self):
        self.mean = np.zeros(self.num_parameters)

    def set_bookkeeping_files(self):

        dir_path = self.relative_path+'data/trial_'+str(self.num_trial)
        create_directory(dir_path)

        self.reward_file_name = create_file(dir_path+'/rewards.txt')

        self.actions_file_name = create_file(dir_path+'/actions.txt')
        
        self.param_file_name = create_file(dir_path+'/params.txt')

        self.converged_in_file_name = create_file(dir_path+'/converged.txt')

        self.states_file_name = create_file(dir_path+'/evaluated_states.txt')

    def store_weights(self):
        save_data_to_file(self.param_file_name, self.policy_params)

    def store_episode_data_to_file(self, episode_data):
        for key in episode_data:
            save_matrix_data_to_file(key, episode_data[key])

    def reset_episode(self):
        self.states[:] = []
        self.noise = []
        self.time_indexes[:] = []
        #Not all data are stored after an evaluation episode
    def reset_eval_episode(self):
        # self.prev_eval_mean_return = curr_eval_return
        self.eval_episode = False
        self.reset_episode()

        # Reset the data belonging to a batch, i.e. empty the lists storing all data used for that particular batch
    def reset_batch(self):

        self.all_returns[:]      = []
        self.all_noises[:]       = []
        # self.all_actions[:]      = []
        self.all_states[:]       = []
        if self.num_train_episode % self.batch_size==0:
            self.eval_episode = True

        #This function calculates the reward recieved from doing an action. The reward is calculated as
        # total reward = -w1*dist+w2*exp(-w3*dist) where dist is the vector distance from the current point
        # to the constraint, i.e. the task error.
    def calculate_return(self, states):

        squared_points = np.square(np.asarray(states))
        control_cost = 0
        running_cost = np.zeros([self.num_inputs,self.num_parameters])
        for i in xrange(self.num_inputs):
            for j in xrange(len(self.time_indexes)):
                control_cost = np.dot(np.dot(self.policy_params[i,:],self.R),self.policy_params[i,:].T)
                running_cost[i,self.time_indexes[j]] += squared_points[j,i]+control_cost

        return running_cost

    def compute_episode_data(self):

        self.episode_reward = self.calculate_return(self.states)
        self.all_returns.append(self.episode_reward)
        self.all_states.append(np.asarray(self.states))
        self.num_states = len(self.states)-1

        self.all_noises.append(self.noise)

    def check_convergence(self):
        dist_norm = np.linalg.norm(self.states[-1])
        print "Dist norm ", dist_norm
        if dist_norm < self.min_dist_norm:
            print "Policy Converged"
            self.train = False
        else:
            self.train = True

    def get_total_rollouts(self):
        return self.num_train_episode+self.num_eval_episode

    def get_episode_data(self):
        episode_data = {}
        if self.train == False:
            episode_data[self.converged_in_file_name] = self.get_total_rollouts()
        elif self.eval_episode == True:
            episode_data[self.states_file_name] = self.states
        else:
            episode_data[self.states_file_name] = self.states
            episode_data[self.reward_file_name] = vec_2_mat(self.episode_reward)
    
        return episode_data

    def converged_episode(self):
        #clean up output file
        self.num_eval_episode += 1
        print "Converged policy  "+str(self.num_eval_episode)+" finished!"
        if self.num_eval_episode == 1:
            converged_episode_data = self.get_episode_data()
            self.store_episode_data_to_file(converged_episode_data)

        if self.num_eval_episode >5:
            self.reset_node('')
        else:
            self.reset_eval_episode()
        
        return PolicySearchResponse(not self.train)


    def evaluation_episode(self):
        self.num_eval_episode += 1
        print "Evaluation episode number "+str(self.num_eval_episode)+" finished!"
        # Calculates the mean return from the evaluation episode 
        curr_eval_return = self.calculate_return(self.states).sum()
        # print "Average return from evaluation is " + str(curr_eval_return)
        # The difference between the returns of the current and previous evaluation episode
        diff = curr_eval_return-self.prev_eval_mean_return

        # If the difference is positive meaning that the policy is improving than we increase the learning rate.
        if diff>0:
            print "Policy got worse by", diff
        else:
            print "Policy improved by", diff

        print "Final task error " + str(self.states[-1])
        self.check_convergence()

        # eval_episode_data = self.get_episode_data()

        # self.store_episode_data_to_file(eval_episode_data)

        self.reset_eval_episode()

    def training_episode(self):
        # First we compute the necessary data based on one episode
        self.compute_episode_data()
        # Here we store the data computed for that episode
        # episode_data = self.get_episode_data()

        # self.store_episode_data_to_file(episode_data)

        self.num_train_episode +=1
        print "Training episode number "+str(self.num_train_episode)+" finished with a cost of " + str(self.episode_reward.sum())# + " and final task error " +  str(self.states[-1]) 

        # The policy is only updated if the number of training episodes match the batch size
        if self.num_train_episode % self.batch_size == 0 and self.train:
            self.update_policy()

        self.reset_episode()

    def PI_Square(self, rewards, noises):
        for j in xrange(self.num_inputs):
            self.policy_params
            S = np.zeros([self.batch_size, self.num_parameters])
            P = np.zeros([self.batch_size, self.num_parameters])
            for k in xrange(self.batch_size):
                for i in xrange(self.num_parameters):
                    S[k,i] = np.sum(rewards[k][j,i:])
            for k in xrange(self.batch_size):
                for i in xrange(self.num_parameters):
                    # num = np.exp((-1/self.Lambda)*S[k,i])
                    # denom = np.sum(np.exp((-1/self.Lambda)*S[:,i]))
                    # P[k,i] = num/denom
                    num = S[k,i]-np.min(S[k,:])
                    denom = np.max(S[k,:])-np.min(S[k,:])
                    P[k,i] = np.exp(-self.c*(num/denom))
            delta_theta = np.zeros([self.batch_size+1, self.num_parameters])
            for i in xrange(self.num_parameters):
                g = np.zeros([self.num_parameters,1])
                g[i,0] = 1
                num = np.dot(np.dot(self.covar_mat,g),g.T)
                denom = np.dot(np.dot(g.T,self.covar_mat),g)
                M = num/denom
                print P.shape
                print M.shape
                print self.all_noises[k].shape
                for k in xrange(self.batch_size):
                    print np.dot(P[k,i],M).shape
                    delta_theta += np.dot(np.dot(P[k,i],M),self.all_noises[k][j,i]) # TODO check the M-matrix
            # print P
            num = 0
            denom = 0
            for i in range(self.num_parameters-1):
                num += (self.num_parameters-i)*delta_theta[i,:]
                denom += (self.num_parameters-i)
            # print num
            # print denom
            delta_theta = num/denom
            self.policy_params[j,:] += delta_theta

    def update_policy(self):

        print "Updating policy"

        # Concatenating all relevant data for each training episode into one 1D vector
        rewards = self.all_returns
        noises = self.all_noises
        self.PI_Square(rewards, noises)


        # Function that stores the batch data in corresponding files
        self.store_weights()
        self.reset_batch()

    def sample_noise(self, req):
        if not self.eval_episode and self.train:
            self.noise = 0.01*np.random.multivariate_normal(self.mean, self.covar_mat, self.num_inputs)

    def set_policy_params(self):
        self.policy_params = np.zeros([self.num_inputs, self.num_parameters])

    def demonstration_episode(self):
        print self.current_time-self.start_time
        self.discretized_time = discretize_data(0, self.current_time-self.start_time, self.num_parameters)
        self.set_policy_params()
        self.reset_episode()
        self.demonstration = False
        
    def get_time_index(self, time_diff):
        i,= np.where(self.discretized_time==self.discretized_time[self.discretized_time<time_diff][-1])
        return i[0]

    # This function updates the policy according to the policy gradient. 
    def policy_search(self, req):
            
        self.init = True
        if self.demonstration:
            self.demonstration_episode()
            return PolicySearchResponse(not self.train)

        elif self.train==False:
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
        
        self.states.append(list(req.task_measures))
        if self.init:
            self.start_time = time.time() * 1000
            self.init = False
        self.current_time = time.time() * 1000

        if self.demonstration:
            task_dynamics = [0] * self.num_inputs
        else:
            time_idx = self.get_time_index(self.current_time-self.start_time)
            self.time_indexes.append(time_idx)
            if self.eval_episode or not self.train:
                # print self.policy_params[:,time_idx]
                task_dynamics = self.policy_params[:,time_idx]
            elif self.train:
                # print self.policy_params[:,time_idx]+self.noise[:,time_idx]
                task_dynamics = self.policy_params[:,time_idx]+self.noise[:,time_idx]

        return  np.asarray(task_dynamics)

    def reset_node(self, req):
        if self.num_trial<self.max_num_trials:
            print "Resets the node, i.e. learning start over"
            self.num_trial+=1
            self.reset_episode()
            self.reset_batch()
            self.set_bookkeeping_files()
            self.num_train_episode = 0
            self.num_eval_episode = 0
            self.train = True
            self.eval_episode = True
            self.prev_eval_mean_return = 0
            self.start_demo()
        else:
            print "Max number of trials reached"
        return []


    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('PISquare')
        PISquare = PISquare()
        PISquare.main()
    except rospy.ROSInterruptException:
        pass47
