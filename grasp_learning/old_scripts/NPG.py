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

        self.mu = 0
        self.sigma = 5
        self.num_episode = 0
        self.num_inputs = 7
        self.num_outputs = 1
        self.gamma = 0.95#0.9
        self.batch_size = 5
        self.g = tf.Graph()
        self.train = True

        self.all_returns = []
        self.all_disc_returns = []

        self.all_actions = []
        self.actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        self.all_task_measure = []

        self.noises = []
        self.all_noises = []

        self.sess = tf.InteractiveSession(graph=self.g)

        self.joint_placeholder = tf.placeholder(tf.float64, [None, 7])
        self.task_dyn_placeholder = tf.placeholder(tf.float64, [None, 1])

        input_data = self.parse_input_data(input_data_file)
        output_data = self.parse_output_data(output_data_file)

        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/rewards.txt'
        self.disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/reward/discounted_rewards.txt'
        self.weights_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights/weights.txt'
        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/actions.txt'
        self.state_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/joint_states/joint_states_python.txt'
        self.actions_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/action.txt'
        self.noise_file = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/noise/noises.txt'
        self.fisher_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions/fisher.txt'
        self.gradient_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients/gradients.txt'
        self.eligability_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/eligability/eligability.txt'
        self.learning_rate_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients/learning_rate.txt'

        self.reset_files([self.learning_rate_file_name, self.weights_file_name,self.eligability_file_name, self.noise_file, self.gradient_file_name, self.fisher_name, self.reward_file_name, self.actions_file_name, self.state_file_name, self.actions_name, self.disc_reward_file_name])


        # self.ff_NN_train, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3 = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data, )
        self.ff_NN_train, weights_1, biases_1, weights_2, biases_2 = self.construct_ff_NN(self.joint_placeholder, self.task_dyn_placeholder,input_data, output_data, )

        self.train_weights = [ ]
        self.train_weights.append(weights_1)
        self.train_weights.append(biases_1)
        self.train_weights.append(weights_2)
        self.train_weights.append(biases_2)
        # self.train_weights.append(weights_3)
        # self.train_weights.append(biases_3)

        self.num_var = self.count_number_trainable_params()
        self.store_weights()

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

    def construct_ff_NN(self ,joint_placeholder, task_dyn_placeholder, joints, task_dyn, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 1 , name=''):
        with self.g.as_default():

            # weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1]))
            # biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            # layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)

            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, 1]))
            # biases_2 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_1_outputs, weights_2) + biases_2

            weights_1 = tf.Variable(tf.truncated_normal([7, HIDDEN_UNITS_L1],dtype=tf.float64), name="netw1",dtype=tf.float64)
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1],dtype=tf.float64), name ="netb1",dtype=tf.float64)
            layer_1_outputs = tf.nn.softplus(tf.matmul(joint_placeholder, weights_1) + biases_1)


            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2],dtype=tf.float64), name="netw2",dtype=tf.float64)
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2],dtype=tf.float64),name="netb2",dtype=tf.float64)
            # layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)
            logits = tf.matmul(layer_1_outputs, weights_2)+biases_2

            # weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, 1],dtype=tf.float64),name="netw3",dtype=tf.float64)
            # biases_3 = tf.Variable(tf.zeros([1],dtype=tf.float64),name="netb3",dtype=tf.float64)
            # logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(logits, task_dyn_placeholder)))
            # error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=task_dyn_placeholder)) 
            # 0.5 * tf.reduce_sum(tf.subtract(logits, task_dyn_placeholder) * tf.subtract(logits, task_dyn_placeholder))

            train_step = tf.train.GradientDescentOptimizer(0.02).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            for i in range(1000):
                _, loss = self.sess.run([train_step, error_function],
                           feed_dict={joint_placeholder: joints,
                                      task_dyn_placeholder: task_dyn})
            print loss
            return logits, weights_1, biases_1, weights_2, biases_2#, weights_3, biases_3

    def print_var(self):

        for trainable_variable in tf.trainable_variables():
            print trainable_variable

    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        print "Number of trainalbe variables are "+ str(tot_nb_params)
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 



    def store_rewards(self):

        print "Storing reward"

        self.task_measure.pop(0)
        curr_rollout_return = self.calculate_return(self.task_measure)
        self.all_returns.append(curr_rollout_return)

        curr_rollout_disc_return = self.discount_rewards(curr_rollout_return) 
        self.all_disc_returns.append(curr_rollout_disc_return)
        
        print "Sum of discounted reward " + str(curr_rollout_disc_return.sum())


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
        f_handle = file(self.actions_name,'a')
        np.savetxt(f_handle, [self.actions], delimiter='\t')
        f_handle.close()

        self.actions[:] = []

    def store_noises(self):
        print "Storing noises"

        self.noises.pop()
        self.all_noises.append(np.asarray(self.noises))
        f_handle = file(self.noise_file,'a')
        np.savetxt(f_handle, [self.noises], delimiter='\t')
        f_handle.close()

        self.noises[:] = []



    def store_states(self):
        print "Storing states"

        self.states.pop()
        self.all_states.append(np.asarray(self.states))
        
        f_handle = file(self.state_file_name,'a')
        np.savetxt(f_handle, [list(itertools.chain.from_iterable(self.states))], delimiter='\t')
        f_handle.close()

        self.states[:] = []

    def store_weights(self):

        var = np.array([])
        for trainable_variable in tf.trainable_variables():
            if "net" in trainable_variable.name:
                var = np.append(var, trainable_variable.eval(session=self.sess))

        f_handle = file(self.weights_file_name,'a')
        np.savetxt(f_handle, [var] , delimiter='\t')
        f_handle.close()


    def store_gradients(self, gradients, learning_rate):
        f_handle = file(self.gradient_file_name,'a')
        np.savetxt(f_handle, [gradients], delimiter='\t')
        f_handle.close()

        f_handle = file(self.learning_rate_file_name,'a')
        np.savetxt(f_handle, [learning_rate], delimiter='\t')
        f_handle.close()

    def store_eligability(self, eligability):
        f_handle = file(self.eligability_file_name,'a')
        np.savetxt(f_handle, [eligability], delimiter='\t')
        f_handle.close()

    # def concatenate_gradiants(self, grad):
    #     res = grad[0]
    #     for i in xrange(len(grad)-1):
    #         res = np.append(res,grad[i+1])
    #     return res

    def concatenate_gradiants(self, grads):

        concat_grads = []
        for grad in grads:
            res= grad[0]
            for i in xrange(len(grad)-1):
                res = np.append(res,grad[i+1])
            # print res
            concat_grads.append(res)
        return np.sum(concat_grads,axis=0)

    
    def fisherMatrix(self, pg):
        pg = np.reshape(pg, (pg.shape[0], 1))
        return pg.dot(pg.T)

    def convertGradients(self, grad, shapes):

        res = []
        lower = 0
        upper = 0
        for i in xrange(len(shapes)):
            if i>0:
                lower += shapes[i-1].size

            upper = lower+shapes[i].size
            res.append(grad[lower:upper].reshape(shapes[i].shape))

        return res

    def gradientAscent(self, direction, learning_rate):
        with self.g.as_default():
            i = 0
            for param in self.train_weights:
                temp = tf.Variable(-learning_rate*direction[i],dtype=tf.float64)
                new_weight_train =  param.assign_add(temp)
                i+=1
                self.sess.run(temp.initializer)
                new_weight_train.eval(session=self.sess)



    #Calculate the baseline of the reward
    # def calculateBaseline(self, gradients, rewards):
    #     baseline = 0
    #     denom = np.zeros(gradients[0].shape[0])
    #     num = np.zeros(gradients[0].shape[0])

    #     for idx in xrange(len(gradients)):
    #         denom += np.square(gradients[idx])
    #         num += np.square(gradients[idx])*rewards[idx].sum()
         
    #     return num/denom

    #         #Calculate the baseline of the reward
    # def calculateBaseline(self, gradients, rewards):
    #     baseline = 0
    #     denom = np.zeros(gradients[0].shape[0])
    #     num = np.zeros(gradients[0].shape[0])

    #     for idx in xrange(len(gradients)):
    #         denom += np.square(gradients[idx])
    #         num += np.square(gradients[idx])*rewards[idx].sum()
         
    #     return num/denom

    def calculateBaseline(self, fisher, VG, eligability, avg_reward, N):
        baseline = 0
        eps = np.eye(self.num_var)*1e-5
        inner = np.linalg.inv(N*fisher-eligability.dot(eligability.T)+eps)
        Q = (1+eligability.T.dot(inner).dot(eligability))/N
        baseline = Q*(avg_reward-eligability.T.dot(np.linalg.inv(fisher)).dot(VG))

        return baseline

    # def calculateEligibility(self, grad, noise):


    def discount_rewards(self, reward):
        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        # discounted_r -= np.mean(discounted_r)
        # discounted_r /= (np.std(discounted_r)+1e-8)

        return discounted_r

    def calculate_return(self, curr_reward):


        dist_square = np.square(np.asarray(curr_reward))
        alpha = 1e-10
        rollout_return = 10*dist_square + 0.4*np.log(dist_square+alpha)
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
        return res.reshape(res.shape[0],1)

    def policy_search(self, req):
        self.num_episode +=1
        print "Episode number "+str(self.num_episode)+" finished!" 

        with self.g.as_default():
            
            self.store_actions()
            self.store_noises()
            self.store_rewards()
            self.store_states()

            if self.num_episode % self.batch_size == 0:
                print "Updating policy"

             

                batch_lower_idx = self.num_episode-self.batch_size
                batch_upper_idx = self.num_episode



                batch_mean_return = np.mean([ self.all_returns[i].sum() for i in range(batch_lower_idx,batch_upper_idx)])
                print "mean of batch rewards is "+ str(batch_mean_return)
                # if batch_mean_return<50:
                #     learning_rate =0.8
                #     self.sigma = 3
                # else:
                #     self.sigma = 2
                #     learning_rate =0.2
                # if batch_mean_return < 10:
                #     self.train = False
                # else:
                #     self.train = True
                # if not self.train:
                #     print "Policy seems to have converged"
                if self.train:

                    loglik = (tf.constant(1.0/self.sigma,dtype=tf.float64)*(self.task_dyn_placeholder - self.ff_NN_train))
                    var_list = tf.trainable_variables()

                    pg = tf.gradients(self.ff_NN_train,var_list)
                    pg = [x for x in pg if x is not None]

                    concat_grads = []
                    all_gradients = []
                    fisher = np.zeros([self.num_var,self.num_var])
                    eps = np.eye(self.num_var)*1e-5
                    VG = np.zeros(self.num_var)
                    avg_reward = 0
                    N = float(self.batch_size)

                    for i in range(batch_lower_idx, batch_upper_idx):

                        action = np.asarray(self.all_actions[i])
                        policy_gradient=[]
                        for j in xrange(action.size): 
                            feed={self.joint_placeholder : self.all_states[i][j].reshape(1,7),
                                  self.task_dyn_placeholder : action[j].reshape(1,1),
                                  }

                            g, error = self.sess.run([pg, loglik], feed_dict=feed)
                            # asd = self.sess.run(train_op2, feed_dict=feed)
                            policy_gradient.append(g*error[0])
                            # gradient.append(g)
                        all_policy_gradients.append(self.concatenate_gradiants(policy_gradient))

                    # baseline = self.calculate_reward_baseline(self.all_disc_returns[batch_lower_idx:batch_upper_idx])

                    for idx,i in zip(xrange(len(all_policy_gradients)),xrange(batch_lower_idx, batch_upper_idx)):
                        fisher += self.fisherMatrix(all_policy_gradients[idx])+eps

                        # VG += all_gradients[idx]*(self.all_disc_returns[i].sum()-baseline)
                        VG += all_policy_gradients[idx]*(self.all_disc_returns[i].sum())

                        avg_reward += self.all_disc_returns[i].sum()
                    

                    fisher /= N
                    VG /= N
                    avg_reward /= N

                    # NG = np.linalg.inv(fisher).dot(VG)

                    eligability = np.sum(all_policy_gradients,axis=0)
                    baseline = self.calculateBaseline(fisher, VG, eligability, avg_reward, N)

                    NG = np.linalg.inv(fisher+eps).dot(VG-eligability*baseline)
                    NG_tf = self.convertGradients(NG, g)

                    learning_rate = 0.1
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    train_op = optimizer.apply_gradients(zip(NG_tf, var_list))
                    self.sess.run(train_op)
                    self.store_gradients(NG, learning_rate)
                    self.store_weights()
                    # self.store_eligability(eligability)

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
            self.noises.append(task_dynamics-mean)
            # print mean
            # print task_dynamics
            self.actions.append(task_dynamics)
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('NPG')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass