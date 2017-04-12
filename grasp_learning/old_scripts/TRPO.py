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
        self.sigma = 1
        self.max_iter_ = 10
        self.burn_in_trials_ = 15
        self.num_episode = 0
        self.num_inputs = 7
        self.num_outputs = 1
        self.gamma = 0.9
        self.batch_size = 1
        self.g = tf.Graph()
        self.train = True
        self.max_kl = 0.001
        self.all_returns = []
        self.all_disc_returns = []

        self.all_actions = []
        self.actions = []

        self.action_dist = []
        self.all_action_dist = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        self.all_task_measure = []

        self.sess = tf.InteractiveSession(graph=self.g)

        self.state_placeholder = obs = tf.placeholder(tf.float32, [None, 7])
        self.action_placeholder = action = tf.placeholder(tf.float32, [None, 1])

        self.action_dist_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.old_action_dist_placeholder = oldaction_dist = tf.placeholder(tf.float32, [None, 1])

        self.advant = advant = tf.placeholder(tf.float32,[None,1])

        self.train_weights = [ ]
        self.eval_weights = [ ]
        input_data = self.parse_input_data(input_data_file)
        output_data = self.parse_output_data(output_data_file)

        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards.txt'
        self.disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards.txt'
        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'
        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.state_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/joint_states/joint_states_python.txt'
        self.actions_temp_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.actions_dist_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/action_dist.txt'

        self.reset_files([self.actions_dist_name, self.reward_file_name, self.actions_file_name, self.state_file_name, self.actions_temp_name, self.disc_reward_file_name])

        # self.saver = tf.train.Saver()

        ff_NN_train, w1_train, w2_train, w3_train = self.construct_ff_NN(self.state_placeholder, self.action_placeholder,input_data, output_data)

        self.ff_NN_train = ff_NN_train

        self.train_weights.append(w1_train)
        self.train_weights.append(w2_train)
        self.train_weights.append(w3_train)

        np.savetxt(self.weights_file_name, [np.append(w1_train.eval(session=self.sess),w2_train.eval(session=self.sess))], delimiter='\t')

        ratio = (self.ff_NN_train/self.old_action_dist_placeholder)

        surr = -tf.reduce_mean(ratio*advant) 


        var_list = tf.trainable_variables()

        batch_size_float = tf.cast(self.batch_size, tf.float32)
        eps = 1e-8
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (ff_NN_train + eps))) / batch_size_float
        ent = tf.reduce_sum(-ff_NN_train * tf.log(ff_NN_train + eps)) / batch_size_float

        self.losses = [surr, kl, ent]

        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            ff_NN_train) * tf.log(tf.stop_gradient(ff_NN_train + eps) / (ff_NN_train + eps))) / batch_size_float

        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.session.run(tf.initialize_all_variables())


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

    def construct_ff_NN(self ,state_placeholder, action_placeholder, joints, task_dyn, HIDDEN_UNITS_L1 = 15, HIDDEN_UNITS_L2 = 3):
        with self.g.as_default():

            # weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            # biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            # layer_1_outputs = tf.nn.softplus(tf.matmul(state_placeholder, weights_1) + biases_1)

            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, 1]))
            # biases_2 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_1_outputs, weights_2) + biases_2

            weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]),name="w1")
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]),name="b1")
            layer_1_outputs = tf.nn.softplus(tf.matmul(state_placeholder, weights_1) + biases_1)


            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]),name="w2")
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]),name="b2")
            layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)

            weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, 1]),name="w3")
            biases_3 = tf.Variable(tf.zeros([1]),name="b3")
            logits = tf.matmul(layer_2_outputs, weights_3)

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, action_placeholder)))

            train_step = tf.train.GradientDescentOptimizer(0.02).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            for i in range(1000):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={state_placeholder: joints,
                                      action_placeholder: task_dyn})
            # print weights_2.eval(session=self.sess)
            print loss
            return logits, weights_1, weights_2, weights_3


    def discount_rewards(self, reward):
        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        discounted_r -= np.mean(discounted_r)
        discounted_r /= (np.std(discounted_r)+1e-8)

        return discounted_r


    def calculate_return(self, curr_reward):

        rollout_return = -10*np.square(np.asarray(curr_reward))
        return rollout_return

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
        f_handle = file(self.actions_temp_name,'a')
        np.savetxt(f_handle, [self.actions], delimiter='\t')
        f_handle.close()

        self.actions[:] = []

        self.action_dist.pop()
        self.all_action_dist.append(np.asarray(self.action_dist))

        f_handle = file(self.actions_dist_name,'a')
        np.savetxt(f_handle, [list(itertools.chain.from_iterable(self.action_dist))], delimiter='\t')
        f_handle.close()

        self.action_dist = []



    def store_states(self):
        print "Storing states"

        self.states.pop()
        self.all_states.append(np.asarray(self.states))
        
        f_handle = file(self.state_file_name,'a')
        np.savetxt(f_handle, [list(itertools.chain.from_iterable(self.states))], delimiter='\t')
        f_handle.close()

        self.states[:] = []

    def var_shape(self,x):
        out = [k.value for k in x.get_shape()]
        assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
        return out


    def numel(self, x):
        return np.prod(self.var_shape(x))


    def flatgrad(self, loss, var_list):
        grads = tf.gradients(loss, var_list)
        grads.pop()
        return tf.concat(0, [tf.reshape(grad, [self.numel(v)])
                         for (v, grad) in zip(var_list, grads)])


    # def fisher_vector_product(self):

    # def linesearch(self):

    # def conjugate_gradient(self):


    def policy_search(self, req):
        self.num_episode +=1
        print "Episode number "+str(self.num_episode)+" finished!" 

        with self.g.as_default():
            
            self.store_actions()
            self.store_rewards()
            self.store_states()


            if self.num_episode % self.batch_size == 0:
                print "Updating policy"

             

                batch_lower_idx = self.num_episode-self.batch_size
                batch_upper_idx = self.num_episode

                #Normalize the rewards
             
                rewards = np.concatenate((self.all_disc_returns[batch_lower_idx:batch_upper_idx]), axis=0)
                states = np.concatenate((self.all_states[batch_lower_idx : batch_upper_idx]), axis=0)
                actions = np.concatenate((self.all_actions[batch_lower_idx : batch_upper_idx]), axis=0)
                action_dist_n = np.concatenate((self.all_action_dist[batch_lower_idx : batch_upper_idx]), axis=0)

                actions= actions.reshape(actions.shape[0],1)
                rewards= rewards.reshape(rewards.shape[0],1)
                action_dist_n = action_dist_n.reshape(action_dist_n.shape[0],1)
                # print self.all_disc_returns
                # print rewards
                # print states
                # print actions 
#                   
                # print self.all_returns
                # print self.all_disc_returns
                # print rewards
                batch_mean_return = np.mean([ self.all_returns[i].sum() for i in range(batch_lower_idx,batch_upper_idx)])
                print "mean of batch rewards is "+ str(batch_mean_return)
                if batch_mean_return > -100:
                    self.train = False
                else:
                    self.train = True
                if not self.train:
                    print "Policy seems to have converged"
                if self.train:
                    # loglik = (tf.constant(1.0/self.sigma)*(self.action_placeholder - self.ff_NN_train))

                    
                    feed = {self.state_placeholder : states,
                            self.action_placeholder : actions,
                            self.advant : rewards,
                            self.old_action_dist_placeholder: action_dist_n}

                    pg = self.flatgrad(surrogat_loss, var_list)
                    print var_list
                    print pg
                    pg.pop()
                    # stepdir = conjugate_gradient(fisher_vector_product, -g)

                    
                    # kl = tf.reduce_sum(self.old_action_dist_placeholder * tf.log((self.old_action_dist_placeholder + 1e-8) / (self.ff_NN_train + 1e-8))) / Nf

                    # losses = [surrogat_loss, kl]

                    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(surrogat_loss)
                    self.sess.run(tf.global_variables_initializer())

                    g = self.sess.run(pg, feed_dict=feed)
                    for i in g:
                        print i
                    # print g.eval(session=self.sess)
                    # self.sess.run(losses, feed_dict=feed)

                    # for i in range(10):
                    #     _, error, ll, advant, actions, states = self.sess.run([train_step, temp, loglik, self.advant, self.action_placeholder, self.state_placeholder],
                    #             feed_dict=feed)
                        # print error
                        # print actions
                        # print states
                    # print error
            success = True
            return success

    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures[0]);
            self.states.append(req.joint_angles)
            mean = self.sess.run(self.ff_NN_train, feed_dict={self.state_placeholder: np.asarray([req.joint_angles])})
            self.action_dist.append(mean)
            if self.train:
                task_dynamics = np.random.normal(mean, self.sigma)
            else:
                task_dynamics = mean
            # print mean
            # print task_dynamics
            self.actions.append(task_dynamics)
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('TRPO')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass