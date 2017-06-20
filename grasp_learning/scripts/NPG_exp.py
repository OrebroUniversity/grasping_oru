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
import tensorflow as tf
import numpy as np
import os
from std_srvs.srv import Empty
import csv
import random
from FileHandler import *
from itertools import product

class Policy(object):

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        load_example_model = rospy.get_param('~load_model', ' ')
        self.model_name = rospy.get_param('~model_name', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        hidden_layers_sizes  =  rospy.get_param('~hidden_units', ' ')
        self.static_lr = rospy.get_param('~use_static_learning_rate', '1')
        self.lr = rospy.get_param('~learning_rate', '1')
        self.subtract_baseline = rospy.get_param('~subtract_baseline', '0')
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.gamma = rospy.get_param('~discount_factor', '0.99') #0.99 #TSV: testing a much more local approach
        self.use_normalized_advantages = rospy.get_param('~use_normalized_advantages', '0')
        self.relative_path = rospy.get_param('~relative_path', ' ')
        self.read_params_from_file = rospy.get_param('~read_params_from_file', '0')
        self.max_num_trials = rospy.get_param('~max_num_trials', '1')
        self.min_dist_norm = rospy.get_param('~min_dist_norm', '0.04')
        self.epsilon = rospy.get_param('~epsilon', '0.1')
        self.num_actions = rospy.get_param('~num_actions', '1')

        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.num_trial=1

    	self.kl = 0.01

        self.train = True
        self.eval_episode = True
        self.NN_output = np.zeros(num_outputs)
        self.all_outputs = []

        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_disc_returns = []
        
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

        self.create_tensorflow_graph(num_inputs, num_outputs, num_rewards, input_output_data_file, hidden_layers_sizes,load_example_model)

        self.store_weights()

        self.VN = NeuralNetworkValueFunction(num_inputs, num_outputs, self.relative_path)

        if self.read_params_from_file==True:
            self.all_params = self.import_params()
            self.set_next_params(self.num_trial)

    def set_services(self):
        rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        rospy.Service('policy_Search', PolicySearch, self.policy_search)
        rospy.Service('reset_node', Empty, self.reset_node)
        self.start_demo = rospy.ServiceProxy('/demo_learning/start_demo', Empty)

    def set_actions(self):
        possible_actions = [-5,0,5]
        self.possible_actions = []
        for roll in product(possible_actions, repeat = self.num_actions):
            self.possible_actions.append(list(roll))


    def create_tensorflow_graph(self,num_inputs, num_outputs, num_rewards, input_output_data_file, hidden_layers_sizes,load_example_model):

        self.g = tf.Graph()

        self.sess = tf.InteractiveSession(graph=self.g) #,config=tf.ConfigProto(log_device_placement=True))
        with tf.name_scope('input'):
            # Placeholder for the states i.e. task errors e 
            self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs],name="Task_error_placeholder") 
            # Placeholder for the softmax output layer 
            self.output_placeholder = tf.placeholder(tf.float32, [None, self.num_outputs],name="Task_error_placeholder") 
            # Placeholder for the actions i.e. task dynamics e_dot_star 
            self.action_placeholder = tf.placeholder(tf.int32, [None], name="Task_dynamics_placeholder")
            # Placeholder for the advantage function
            self.advantage = tf.placeholder(tf.float32,[None,num_rewards], name="Advantage_placeholder")
        with tf.name_scope("Hyper_params"):
            # Placeholder for the learning rate
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate_placeholder")

        # List that will containg the placehoalder for the fisher matrices
        self.fisher_matrix = []
        self.NPG = []

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.ff_NN_train = self.construct_Neural_Network(self.state_placeholder, self.num_outputs, hidden_layers_sizes)

        self.indexes = tf.range(0, tf.shape(self.ff_NN_train)[0]) * tf.shape(self.ff_NN_train)[1] + self.action_placeholder
        self.responsible_outputs = tf.gather(tf.reshape(self.ff_NN_train, [-1]), self.indexes)

        self.var_list = tf.trainable_variables()

        with tf.name_scope("NPG"):
            # The output from the neural network is the mean of a Gaussian distribution. This variable is simply the
            # log likelihood of that Gaussian distribution
            # self.loglik = self.action_placeholder*tf.log(self.ff_NN_train) + (1 - self.action_placeholder)*tf.log(1-self.ff_NN_train)
            self.loglik = tf.log(self.responsible_outputs)# + (1 - self.action_placeholder)*tf.log(1-self.ff_NN_train)

            self.loss = -tf.reduce_mean(self.loglik*self.advantage, name='loss_reduce_mean')#/self.batch_size            
            # self.loss = -tf.reduce_mean(tf.multiply(self.loglik,self.advantage,'loss_prod'), 0, name='loss_reduce_mean')#/self.batch_size            
            # Get the list of all trainable variables in our tensorflow graph
            # Compute the analytic gradients of the loss function given the trainable variables in our graph
            #with tf.device('/cpu:0'):
            self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list)
            # Calculate the gradients of the policy
            self.pg = tf.gradients(self.loglik, self.var_list,name="Policy_gradients")
            self.lg = tf.gradients(self.loss, self.var_list, name="Loss_gradients")
            self.set_natural_policy_gradients()
        
            self.train_op = self.optimizer.apply_gradients(self.NPG)
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if load_example_model:
            self.saver.restore(self.sess, self.relative_path+'models/'+self.model_name)
        else:
            self.saver.save(self.sess, self.relative_path+'models/'+self.model_name)

        self.setup_tensorboard()

        print "Network initialized"

    def construct_Neural_Network(self, input_placeholder, num_outputs, hidden_units):
        
            prev_layer = input_placeholder
            layer = 0
            for units in hidden_units:
                prev_layer = self.create_layer(prev_layer, units, tf.nn.tanh, layer , "Hidden_"+str(layer))
                layer+=1

            # last layer is a linear output layer
            num_outputs = int(num_outputs)
            return self.create_layer(prev_layer, num_outputs, None, layer, "Output")

    def create_layer(self, prev_layer, layer_size, activation_fun, num_layer, name):
        with tf.name_scope(name):
                with tf.name_scope('weights'):
                    weights = tf.Variable(tf.truncated_normal([int(prev_layer.shape[1]), layer_size]),name="NN/w"+str(num_layer)+"/t")
                with tf.name_scope('biases'):
                    biases = tf.Variable(tf.truncated_normal([layer_size]), name="NN/b"+str(num_layer)+"/t")
                if activation_fun == None:
                    layer_output = tf.nn.softmax(tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases,name="NN/output/t"))
                else:
                    layer_output = activation_fun(tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases),name="NN/L"+str(num_layer)+"/t")

                return layer_output

    # The natural policy gradient (NPG) is the inverse of the fisher matrix times the gradient.
    # In this function the NPG is set by muliplying a placeholder storing the inverse of the fisher
    # matrix with the corresponding policy gradient. The policy gradient is the gradient of the neural network (NN)
    # and in tensorflow the gradient of the NN is split into a list where each entry corresponds to the parameters
    # in each layer
    def set_natural_policy_gradients(self):
        i=1
        for g,v in self.loss_grads:
            self.fisher_matrix.append(tf.placeholder(tf.float32, g.shape,name="Fisher_/"+str(i)))
            grad = tf.multiply(self.fisher_matrix[-1], g,name="PG_Fisher_Mul_"+str(i))
            self.NPG.append((grad, v))
            i+=1

    def setup_tensorboard(self):
        with tf.name_scope('summaries'):
            tf.summary.histogram('loglik_hist', self.loglik)
            tf.summary.histogram('loss_hist', self.loss)
            tf.summary.histogram('advantages_hist', self.advantage)
            tf.summary.histogram('task_errors_hist', self.state_placeholder)
            tf.summary.histogram('task_dynamics_hist', self.action_placeholder)

            tf.summary.scalar('learning_rate',self.learning_rate)

            self.store_containers_in_tensorboard(self.lg)
            self.store_containers_in_tensorboard(self.pg)
            self.store_containers_in_tensorboard(self.var_list)

            self.merged_summary = tf.summary.merge_all()
            self.set_tensorflow_summary_writer()

    def store_containers_in_tensorboard(self, container):

        if type(container[0]) is tuple:
            for grad,var in container:
                name = grad.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , grad)
        elif type(container) is list:
            for var in container:
                name = var.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , var)

    def set_tensorflow_summary_writer(self):
        dir_path = self.relative_path+'/graphs/trial_'+str(self.num_trial)
        create_directory(dir_path)

        #setup tensor board writers
        self.train_writer = tf.summary.FileWriter(dir_path,self.g)

    def import_params(self):
        file_name = self.relative_path+'../parameter_data/parameters.csv'
        return read_csv_file(file_name)

    def set_next_params(self, current_trial_num):
        curr_param = self.all_params[current_trial_num]
        self.lr = float(curr_param[0])
        self.subtract_baseline = int(curr_param[1])
        self.batch_size = int(curr_param[2])
        self.gamma = float(curr_param[3])
        self.use_normalized_advantages = int(curr_param[4])
        self.min_dist_norm = float(curr_param[5])
        data = ["Learning_rate", "Subtract_baseline","Batch_Size","Discount_factor","Normalize_advantage","Min_dist_until_convergance"]
        self.save_vector_data_to_file(self.param_file, data)
        self.save_vector_data_to_file(self.param_file, curr_param)

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

        self.baseline_file_name = create_file(dir_path+'/baseline.txt')
        self.advantages_file_name = create_file(dir_path+'/advantages.txt')
        self.unnorm_advantageages_file_name = create_file(dir_path+'/unnorm_advantages.txt')

        self.loss_file_name = create_file(dir_path+'/losses.txt')
        self.log_likelihood_file_name = create_file(dir_path+'/log_likelihood.txt')

        self.PG_file_name = create_file(dir_path+'/PG.txt')

        self.loss_grads_file_name = create_file(dir_path+'/loss_grads.txt')
        self.NPG_file_name = create_file(dir_path+'/NPG.txt')

        self.fisher_file_name = create_file(dir_path+'/fisher.txt')

        self.neural_network_output_file_name = create_file(dir_path+'/output.txt')

        self.eval_rewards_file_name = create_file(dir_path+'/eval_rewards.m')
        self.tdyn_file_name = create_file(dir_path+'/tdyn.m')

    def store_weights(self):

        var = np.array([])
        for trainable_variable in self.var_list:
            var = np.append(var, trainable_variable.eval(session=self.sess))
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

    def compute_episode_data(self):

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))

        self.task_measure.pop(0)

        self.episode_reward = self.calculate_return(self.task_measure)
        self.all_returns.append(self.episode_reward)

        self.episode_disc_reward = self.discount_rewards(self.episode_reward) 
        self.all_disc_returns.append(self.episode_disc_reward)

        self.states.pop()
        self.all_states.append(np.asarray(self.states))

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

        self.eval_episode = True
        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_actions[:]      = []
        self.all_states[:]  = []

        # Calculates the disounted returns from one episode. The cumulative disounted return (R) starts summing from the last reward (r) at time T.
        # At time T-1 the return is R(T-1)=r(t-1)+gamma*r(T),  at time T-2 it is R(T-2)=r(t-2)+gamma*r(t-1), and so on until R(0)=r(0)+gamma*r(1).
    def discount_rewards(self, reward):

        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in xrange(0, len(reward)):
            discounted_r[t] = running_add
        discounted_r.append(reward[-1])

        return discounted_r

        #This function calculates the reward recieved from doing an action. The reward is calculated as
        # total reward = -w1*dist+w2*exp(-w3*dist) where dist is the vector distance from the current point
        # to the constraint, i.e. the task error.
    def calculate_return(self, curr_reward):

        squared_points = np.square(np.asarray(curr_reward))
        dist_abs = np.abs(np.asarray(curr_reward))

        rollout_return = 0
        
    	dist = np.sqrt(np.sum(squared_points,axis=1))
    	dist_l1 = np.sum(dist_abs,axis=1)

        alpha = 0.2#1e-17

        rollout_return = -(np.sqrt(np.square(dist)+np.square(alpha))-alpha)

        return rollout_return

        #Normalizes the data to have mean 0 and variance 1
    def normalize_data(self, data):
        return (data-np.mean(data,axis=0))/(np.std(data,axis=0)+1e-8)


        # Calculates the inverse of the fisher matrix used for the natural policy gradient
        # http://www.scholarpedia.org/article/Policy_gradient_methods
    def calculate_inverse_Fisher_Matrix(self, states, actions):
        
            feed_dict={self.state_placeholder:states, self.action_placeholder : actions}
            pg = self.sess.run(self.pg, feed_dict)
            fisher = []
            eps = 1e-8
            for g in pg:
                fisher.append((np.asarray(1/(np.square(g.flatten())+eps)).reshape(g.shape))/self.batch_size)
            return fisher, pg

        # Concatenates a list of lists to one vector
    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec

        # Calculates the step size for the gradient descent suggesed by Jan Peters and Stefan Schaal in their paper
        # "Reinforcement learning of motor skills with policy gradients" 
        # http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf 
    def calculate_learning_rate(self, fisher, input_data, output_data, rewards):
        
            feed_dict={self.state_placeholder  : input_data,
                        self.action_placeholder : output_data,
                        self.advantage          : rewards}

            lg = np.asarray(self.sess.run(self.lg, feed_dict=feed_dict))
            flatten_fisher = self.flattenVectors(fisher)
            flatten_lg = self.flattenVectors(lg)
            flatten_lg = flatten_lg.reshape(flatten_lg.shape[0],1)
            flatten_fisher = flatten_fisher.reshape(flatten_fisher.shape[0],1)
            eps = 1e-5
            denom = eps+np.square(flatten_lg).T.dot(flatten_fisher)
            step_size = np.sqrt(self.kl/denom)

            print "STEPSIZE"
            print step_size
            return step_size

        # Calculates the advantage function by substracting a baseline from the return at each time step. 
        # The baseline predicts a value representing how good that state is and is approximated as a neural
        # network. Advanage functions are explained in https://arxiv.org/abs/1506.02438
    def calculate_advantages(self, rewards, states):
        advantages = np.zeros_like(rewards)
        baseline = np.zeros_like(rewards)
        for i in xrange(len(states)):
            baseline[i] = self.VN.predict(states[i,:]).flatten()
            if self.subtract_baseline:
                advantages[i] = rewards[i]-np.abs(baseline[i])
            else:
                advantages[i] = rewards[i]

        if self.use_normalized_advantages:
            advantages = advantages.reshape(advantages.shape[0],1)      
            advantages = self.normalize_data(advantages)
        else:
            advantages = advantages.reshape(advantages.shape[0],1)
        return advantages, baseline

    def check_convergence(self):
        dist_norm = np.linalg.norm(self.states[-1])
        print "Dist norm ", dist_norm
        if dist_norm < self.min_dist_norm:
            print "Policy Converged"
            self.train = False
        else:
            self.train = True

    def get_training_data(self):
        # actions = np.concatenate(self.all_actions)
        # num_NN_outputs = self.action_placeholder.shape[1]
        # all_actions = []
        # for i in range(len(actions)):
        #     temp = np.zeros(num_NN_outputs)
        #     temp[actions[i]] = 1
        #     all_actions.append(temp)

        return np.concatenate(self.all_disc_returns), np.concatenate(self.all_states), np.concatenate(self.all_actions)

    def get_episode_data(self):
        episode_data = {}

        if self.eval_episode == True:
            episode_data[self.eval_action_file_name] = vec_2_mat(self.eval_actions)
            episode_data[self.evaluated_states_file_name] = self.states
        else:
            episode_data[self.task_measure_file_name] = self.task_measure
            episode_data[self.explored_states_file_name] = self.states
            episode_data[self.evaluated_states_file_name] = self.states
            episode_data[self.reward_file_name] = vec_2_mat(self.episode_reward)
            episode_data[self.disc_reward_file_name] = vec_2_mat(self.episode_disc_reward)
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

        eval_episode_data = self.get_episode_data()

        self.store_episode_data_to_file(eval_episode_data)

        self.reset_eval_episode(curr_eval_return)

    def training_episode(self):
        # First we compute the necessary data based on one episode
        self.compute_episode_data()

        # Here we store the data computed for that episode
        episode_data = self.get_episode_data()

        self.store_episode_data_to_file(episode_data)

        self.num_train_episode +=1
        print "Training episode number "+str(self.num_train_episode)+" finished with a reward of " + str(self.episode_reward.sum())# + " and final task error " +  str(self.states[-1]) 

        # The policy is only updated if the number of training episodes match the batch size
        if self.num_train_episode % self.batch_size == 0 and self.train:
            self.update_policy()

        self.reset_episode()

    def update_policy(self):

        print "Updating policy"

        #Concatenating all relevant data for each training episode into one 1D vector
        rewards, states, actions = self.get_training_data()

        # Reshape the states to be the same shape as their individual placeholder
        states = states.reshape(states.shape[0],self.state_placeholder.shape[1])

        # Calculate the advantages. The unnormalized advantages are not used but stored in a file such that it can be analyzed afterhand 
        advantage, baseline = self.calculate_advantages(rewards, states)

        # Calculate the mean of all returns for the batch. It is only done to see if the return between batches increases
        curr_batch_mean_return = np.mean([ret.sum() for ret in self.all_returns])

        print "mean of batch rewards is "+ str(curr_batch_mean_return)
        fisher, pg = self.calculate_inverse_Fisher_Matrix(states, actions)
        if self.static_lr:
            learning_rate = self.lr
        else:
            learning_rate = float(self.calculate_learning_rate(fisher, states, actions, advantage))

        # As long as the policy has not converged or the episode is an evaluation episode then update the policy

        # Load all batch data into a dictionary
        feed_dict={self.state_placeholder  : states,
                   self.action_placeholder : actions,
                   self.advantage          : advantage,
                   self.learning_rate      : learning_rate}

        # The fisher matrix is loaded differently due to the fact that it is split up into a list
        i=0
        for f in fisher:
            feed_dict[self.fisher_matrix[i]] = f
            i+=1

        # Here the NPG are calculated. This is only done for the sole purpose of storing the gradients such that
        # they can be plotted and analyzed in afterhand
        loss_grads = self.flattenVectors(self.sess.run([g for g,v in self.loss_grads],feed_dict))
        NPG = self.flattenVectors(self.sess.run([grad[0] for grad in self.NPG],feed_dict))
        unum = int (self.num_train_episode / self.batch_size)

        # run i numbers of gradient descent updates
        # setup run options for profiling
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range(1):
            _, error, ll, summary = self.sess.run([self.train_op, self.loss, self.loglik, self.merged_summary], feed_dict,
            run_options,
            run_metadata)
        
        #add summaries for this training run
        self.train_writer.add_summary(summary,unum)
        self.train_writer.add_run_metadata(run_metadata, 'step%03d' % unum)
        # print('Adding run metadata for', unum)


        # This dictionary holds all the batch data for one batch. 
        batch_dic = {}
        batch_dic[self.baseline_file_name] = baseline
        batch_dic[self.advantages_file_name] = advantage.flatten()
        batch_dic[self.log_likelihood_file_name] = [ll]
        batch_dic[self.loss_file_name] = [error]
        batch_dic[self.NPG_file_name] = NPG
        batch_dic[self.loss_grads_file_name] = loss_grads
        batch_dic[self.PG_file_name] = self.flattenVectors(pg)
        batch_dic[self.fisher_file_name] = self.flattenVectors(fisher)

        # Function that stores the batch data in corresponding files
        self.store_batch_data_to_file(batch_dic)

        self.VN.train(states, rewards, self.batch_size)

        self.reset_batch()
        self.epsilon /= 1.1


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
            self.states.append(req.task_measures)
            feed_dict = {self.state_placeholder: np.array([req.task_measures])}
            self.NN_output = self.sess.run(self.ff_NN_train, feed_dict)
            self.all_outputs.append(list(self.NN_output.flatten()))
            if self.train and not self.eval_episode:
                if np.random.rand(1) < self.epsilon:
                    idx = random.randint(0, self.num_outputs-1)
                else:
                    idx = np.argmax(self.NN_output)

                task_dynamics=np.asarray(self.possible_actions[idx])
                # task_dynamics=np.asarray(list(req.task_measures))+np.asarray(self.possible_actions[idx])
            else:
                idx = np.argmax(self.NN_output)
                task_dynamics=np.asarray(self.possible_actions[idx])
                # task_dynamics=np.asarray(list(req.task_measures))+np.asarray(self.possible_actions[idx])
                self.eval_actions.append(task_dynamics[0])
            # Store the output from the neural network which is the action with no noise added to it
            self.actions.append(idx)
            return  np.asarray(task_dynamics)


    def reset_node(self, req):
        if self.num_trial<self.max_num_trials:
            self.num_trial+=1
            self.saver.restore(self.sess, self.relative_path+'models/'+self.model_name)
            self.reset_episode()
            self.reset_batch()
            self.set_bookkeeping_files()
            self.set_tensorflow_summary_writer()
            self.num_train_episode = 0
            self.num_eval_episode = 0
            self.set_next_params(self.num_trial)
            self.train = True
            self.start_demo()
        else:
            print "Max number of trials reached"
        return []

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('NPG_exp')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass47
