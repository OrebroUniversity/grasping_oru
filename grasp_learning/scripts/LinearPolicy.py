#!/usr/bin/env python
import roslib

import rospy
import sys
import itertools
from utils import *
from value_function import *
from grasp_learning.srv import *
from std_msgs.msg import Empty
from math import sqrt
from math import pow
import numpy as np
import os
from std_srvs.srv import Empty
import csv

class LinearPolicy(object):

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.relative_path = rospy.get_param('~relative_path', ' ')
        self.max_num_trials = rospy.get_param('~max_num_trials', '1')
        self.min_dist_norm = rospy.get_param('~min_dist_norm', '0.04')

        self.sigma = np.zeros(num_outputs)
        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.num_trial=1

        self.train = True
        self.eval_episode = True

        self.all_returns = []
        self.all_disc_returns = []
        self.exploration = []

        self.all_actions = []
        self.actions = []
        self.eval_actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []

        self.prev_eval_mean_return = 0

        self.set_services()
        self.set_bookkeeping_files()


    def set_services(self):
        rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        rospy.Service('policy_Search', PolicySearch, self.policy_search)
        self.start_demo = rospy.ServiceProxy('/demo_learning/start_demo', Empty)

    # Parses the training data stored in parameter input_file into corresponding input and output data 
    def parse_input_output_data(self, input_file):
        input_data = []
        output_data = []
        input_ = []
        output_ = []
        i = 0
        with open(input_file, 'rU') as f:
            for line in f:
                #Skip first two lines of the file
                if (i==0 or i==1):
                    i+=1
                    continue
                line = line.split()
                for string in xrange(len(line)):
                    if string%2==0:
                        input_data.append(float(line[string])+np.random.normal(0, 0.1))
                    else:
                        output_data.append(float(line[string]))

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)

    def set_bookkeeping_files(self):

        dir_path = self.relative_path+'data/trial_'+str(self.num_trial)
        self.create_directory(dir_path)

        self.param_file = dir_path+'/params.txt'

        self.reward_file_name = dir_path+'/rewards.txt'
        
        self.evaluated_states_file_name = dir_path+'/evaluated_states.txt'

    def create_directory(self, newpath):
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    # Opens and closes a file to empty its content
    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()

    def save_matrix_data_to_file(self, filename, data):
        f_handle = file(filename,'a')
        for inner_list in data:
            for elem in inner_list:
                f_handle.write(str(elem)+" ")
        f_handle.write("\n")
        f_handle.close()


    def save_vector_data_to_file(self, filename, data):
        f_handle = file(filename,'a')
        for elem in data:
            f_handle.write(str(elem)+" ")
        f_handle.write("\n")
        f_handle.close()

    def store_episode_data_to_file(self, episode_data):
        
        for key in episode_data:
            self.save_matrix_data_to_file(key, episode_data[key])


    def reset_episode(self):
        self.task_measure[:] = []
        self.states[:] = []
        self.actions[:] = []
        self.mean_action[:] = []
        self.exploration[:]  = []
        self.eval_actions[:] = []

        #Not all data are stored after an evaluation episode
    def reset_eval_episode(self, curr_eval_return):
        self.prev_eval_mean_return = curr_eval_return
        self.eval_episode = False
        self.reset_episode()

        # Reset the data belonging to a batch, i.e. empty the lists storing all data used for that particular batch
    def reset_batch(self):

        self.eval_episode = True
        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_actions[:]      = []
        self.all_states[:]  = []

        #This function calculates the reward recieved from doing an action. The reward is calculated as
        # total reward = -w1*dist+w2*exp(-w3*dist) where dist is the vector distance from the current point
        # to the constraint, i.e. the task error.
    def calculate_return(self, curr_reward):
        return 0

    def check_convergence(self):
        dist_norm = np.linalg.norm(self.states[-1])
        print "Dist norm ", dist_norm
        if dist_norm < self.min_dist_norm:
            print "Policy Converged"
            self.train = False
        else:
            self.train = True

        # The variance of the Gaussian distribution is set as the mean of the actions from the evaluation episode.
        # In this way the noise is high enough to actually impact the mean of the neural network
    def set_exploration(self):
        return 0


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
            print "Policy improved by", diff
        else:
            print "Policy got worse by", diff

        self.set_exploration()

        print "Final task error " + str(self.states[-1])
        self.check_convergence()


    def training_episode(self):
        return 0
    def update_policy(self):
        return 0
        # This function updates the policy according to the policy gradient. 
    def policy_search(self, req):
        
            
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


            self.actions.append(task_dynamics[0])
            return  task_dynamics.flatten()


    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('LinearPolicy')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass47
