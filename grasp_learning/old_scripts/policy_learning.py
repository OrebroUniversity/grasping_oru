#!/usr/bin/env python
import roslib

import rospy
import sys
import bisect
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
import numpy as np

class Policy(object):
    """docstring for Policy"""
    def __init__(self):
        self.net = 0
        self.input_data = 0
        self.output_data = 0
        input_data_file = rospy.get_param('~input_training_data', ' ')
        output_data_file = rospy.get_param('~output_training_data', ' ')
        self.parse_input_data(input_data_file)
        self.parse_output_data(output_data_file)
        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)
        self.sample_and_rewight_sub_ = rospy.Subscriber('sample_and_rewight', Empty, self.sample_and_rewight)        
        self.NN_params_ = 0
        self.noises = np.zeros((80,0))
        self.returns = []
        self.mu = 0
        self.sigma = 0.3 
        self.max_iter_ = 10
        self.burn_in_trials_ = 0
        self.curr_rollout_noise = np.zeros((80,1))
        self.num_rollout = 0
        self.best_rollouts = []

        self.train_policy()

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
        self.input_data = np.asarray(data)

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
        self.output_data = np.asarray(data)

    def train_policy(self):
        ds = SupervisedDataSet(7, 1)
        for i in range(self.input_data.shape[0]-1):
            ds.appendLinked(self.input_data[i,:],self.output_data[i])

        self.net = buildNetwork(7, 10, 1, bias=True, hiddenclass=TanhLayer)
        trainer = BackpropTrainer(self.net, ds)
        trainer.train()
        NetworkWriter.writeToFile(self.net, '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/policy/policy.xml')
        self.NN_params_ = deepcopy(self.net.params)
        self.sigma = np.std(self.NN_params_)
        print "Initial std is: "+ str(self.sigma)

    def sample_noise(self):
        noise = np.random.normal(self.mu, self.sigma, 80)
        self.curr_rollout_noise = np.reshape(noise,(noise.shape[0],1))
        return noise

    def set_weights(self,added_weights):
        new_params = deepcopy(self.NN_params_)
        new_params[11:] += added_weights
        self.net._setParameters(new_params)

    def update_params(self, delta_theta):
        print "Updating policy parametres"
        temp = deepcopy(self.NN_params_)
        temp[11:] += delta_theta[:,0]
        self.NN_params_ = deepcopy(temp)
        self.net._setParameters(self.NN_params_)

    def imp_sampling_order(self,reward):
        print "Storing trajectory"
        pos = bisect.bisect_left(sorted(self.returns), reward)
        self.returns.append(reward)
        self.best_rollouts.insert(pos, self.num_rollout)

        reshaped_noise = self.curr_rollout_noise.reshape((self.curr_rollout_noise.shape[0],1))
        if (self.num_rollout==0):
            self.noises = self.curr_rollout_noise
        else:
            self.noises = np.append(self.noises,self.curr_rollout_noise,axis=1)
        self.num_rollout+=1

    def policy_search(self, req):
        self.imp_sampling_order(req.reward)
        numerator = np.zeros((1,80))
        denominator = 0
        num_iter = self.max_iter_ if len(self.returns) >= 10 else len(self.returns)
        for i in range(num_iter):
            curr_best_rollout = self.best_rollouts[-(i+1)]
            numerator += self.returns[curr_best_rollout]*self.noises[:,curr_best_rollout]
            denominator += self.returns[curr_best_rollout]

        new_weights = numerator/denominator
        if self.num_rollout >= self.burn_in_trials_:
            self.update_params(new_weights.T)
            self.sigma = sqrt(1/denominator)
            print "new variance is " + str(self.sigma)
        success = True
        return success

    def sample_and_rewight(self, req):
        print "Sampling noise"
        noise = self.sample_noise()
        self.set_weights(noise)

    def handle_query_NN_(self,req):
        task_dynamics = self.net.activate(req.joint_angles)
        return  task_dynamics

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_learner')
        policy = Policy()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass