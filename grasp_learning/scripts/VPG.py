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



class Policy(object):

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        load_example_model = rospy.get_param('~load_model', ' ')
        model_name = rospy.get_param('~model_name', ' ')
        hidden_layer_size  =  rospy.get_param('~hidden_layer_size', ' ') 
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.relative_path =        rospy.get_param('~relative_path', ' ')
        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.sigma = np.zeros(num_outputs)
        self.random_bias = np.zeros(num_outputs)
        self.num_train_episode = 0
        self.num_eval_episode = 0
    	self.gamma = 0.99 #0.99 #TSV: testing a much more local approach

        self.g = tf.Graph()
        self.train = True
        self.eval_episode = True
        self.epsilon = 1e-5
        self.mean = np.zeros(num_outputs)
        self.ffnn_mean = np.zeros(num_outputs)
        self.lr = 0.001
        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_disc_returns = []
        self.exploration = []
        self.init = True
        self.all_actions = []
        self.actions = []
        self.eval_actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []
        self.all_task_measure = []

        self.mean_action = []

        self.network_params = []
        # This list holds the natural policy gradients which is composed of the inverse of the fisher
        # matrix mulitpled by the policy gradient. For more info check http://www.scholarpedia.org/article/Policy_gradient_methods

        self.prev_eval_mean_return = 0

        # self.VN = LinearValueFunction()
        self.sess = tf.InteractiveSession(graph=self.g) #,config=tf.ConfigProto(log_device_placement=True))

        # Placeholder to which the future task errors and task dynamics are loaded into 
        self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs],name="Task_error_placeholder") 
        self.action_placeholder = tf.placeholder(tf.float32, [None, num_outputs], name="Task_dynamics_placeholder")
        # Placeholder for the learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate_placeholder")
    	tf.summary.scalar('learning_rate',self.learning_rate)

        # Placeholder for the advantage function
        self.advantage = tf.placeholder(tf.float32,[None,num_rewards], name="Advantage_placeholder")

        # Placehoalder for the variance of the added noise
        self.var = tf.placeholder(tf.float32, [num_outputs], name="Variance_placeholder")

        # Create the optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        input_data, output_data = self.parse_input_output_data(input_output_data_file)

        self.set_bookkeeping_files(self.relative_path)

        self.construct_ff_NN(self.state_placeholder, self.action_placeholder , input_data, output_data, num_inputs, num_outputs, hidden_layer_size)
                
        # The output from the neural network is the mean of a Gaussian distribution. This variable is simply the
        # log likelihood of that Gaussian distribution

        self.loglik = gauss_log_prob(self.ff_NN_train, self.var, self.state_placeholder)
        # This is the most important function of them all. It is the loss function and by taking the gradient of it we obtain the
        # policy gradient telling us in what direction we should change our parameters, in this case the weights of the neural network, as to
        # increase future rewards.
        # self.loss = -tf.reduce_mean(tf.multiply(self.loglik,self.advantage,'loss_prod'),name='loss_reduce_mean') 
    	#self.loss = -tf.reduce_mean(self.loglik * self.advantage)
        # Get the list of all trainable variables in our tensorflow graph
        # self.var_list = tf.trainable_variables()
        # Compute the analytic gradients of the loss function given the trainable variables in our graph
        # self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list, name="Loss_function_gradients")

    	tf.summary.histogram('loglik', self.loglik)
    	tf.summary.histogram('adv', self.advantage)

        self.loss = -tf.reduce_mean(tf.multiply(self.loglik,self.advantage,'loss_prod'),name='loss_reduce_mean')/self.batch_size
	#self.loss_prod = tf.multiply(self.loglik,self.advantage,'mrNasty')
	#self.loss = -tf.reduce_mean(self.loss_prod,name="Loss_function")/self.batch_size
            
        tf.summary.histogram('loss', self.loss)
    	self.merged_summary = tf.summary.merge_all()
    	tf.summary.histogram('loglik_hist', self.loglik)
    	tf.summary.histogram('loss_hist', self.loss)
    	tf.summary.histogram('adv_hist', self.advantage)
        tf.summary.tensor_summary('loglikelihood', self.loglik)
        tf.summary.tensor_summary('advantages', self.advantage)
        tf.summary.tensor_summary('loss', self.loss)

        # Get the list of all trainable variables in our tensorflow graph
        self.var_list = tf.trainable_variables()
        # Compute the analytic gradients of the loss function given the trainable variables in our graph
    	#with tf.device('/cpu:0'):
    	self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list)
        # Calculate the gradients of the policy
        self.pg = tf.gradients(self.loglik, self.var_list,name="Policy_gradients")
        self.lg = tf.gradients(self.loss, self.var_list, name="Loss_gradients")
    	#tf.summary.tensor_summary('loss_gradients_summary', self.loss_grads) #didn't work. why?
    	#tf.summary.histogram('loss_grad_hist', self.loss_grads)
            
    	#tf.summary.tensor_summary('policy_gradients', self.pg)
    	#tf.summary.histogram('policy_grad_hist', self.pg)
        
        self.train_op = self.optimizer.apply_gradients(self.loss_grads)

        saver = tf.train.Saver()

        if load_example_model:
            saver.restore(self.sess, self.relative_path+'models/'+model_name)
        else:
            save_path = saver.save(self.sess, self.relative_path+'models/'+model_name)

        self.store_weights()
                    
    	#setup tensor board writers
    	self.train_writer = tf.summary.FileWriter(self.relative_path+'/graphs',self.g)
    	self.merged_summary = tf.summary.merge_all()
    	# self.sess.run(tf.global_variables_initializer())
    	#freezing the main graph
    	# self.g.finalize()

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
                # if (i==0):
                    i+=1
                    continue
                line = line.split()
                for string in xrange(len(line)):
                    if string%2==0:
                        input_data.append(float(line[string])+np.random.normal(0, 0.1))
                    #Every second column contains the output data
                    else:
                        # if string == 1:
                        output_data.append(float(line[string]))
                        # output_data.append(float(line[string])+np.random.normal(0, 0.5))

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)


    def construct_ff_NN(self, task_error_placeholder, task_dyn_placeholder, states, task_dyn, NUM_INPUTS = 1, NUM_OUTPUTS = 1, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 10):
        with self.g.as_default():

            weights_1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, HIDDEN_UNITS_L1]),name="w1")
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]), name="b1")
            layer_1_outputs = tf.nn.tanh(tf.matmul(task_error_placeholder, weights_1,name="Input_W1_Mul") + biases_1,name="L1")

            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, NUM_OUTPUTS]),name="w2")
            biases_2 = tf.Variable(tf.zeros([NUM_OUTPUTS]),name="b2")
            self.ff_NN_train = tf.add(tf.matmul(layer_1_outputs, weights_2,name="L1_W2_Mul"),biases_2,name="output")


            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]),name="w2")
            # biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]),name="b2")
            # layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)

            # weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, NUM_OUTPUTS]))
            # biases_3 = tf.Variable(tf.zeros([NUM_OUTPUTS]))
            # self.ff_NN_train = tf.add(tf.matmul(layer_2_outputs, weights_3,name="L1_W2_Mul"),biases_3,name="output")

            # logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(self.ff_NN_train, task_dyn_placeholder)),0)

            train_op = self.optimizer.minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],NUM_OUTPUTS)
            states= states.reshape(states.shape[0],NUM_INPUTS)

            feed_dict={task_error_placeholder: states,
                        task_dyn_placeholder: task_dyn,
                        self.learning_rate : 0.001} 

            for i in range(1100):
                _, loss = self.sess.run([train_op, error_function],feed_dict)

            print "Network trained"

    def set_bookkeeping_files(self, relative_path):

        self.reward_file_name = relative_path+'data/rewards.txt'
        self.disc_reward_file_name = relative_path+'data/discounted_rewards.txt'

        self.actions_file_name = relative_path+'data/actions.txt'
        self.action_dist_mean_file_name = relative_path+'data/action_dist_mean.txt'
        self.exploration_file_name = relative_path+'data/exploration.txt'
        self.eval_action_file_name = relative_path+'data/eval_actions.txt'
        
        self.explored_states_file_name = relative_path+'data/explored_states.txt'

        self.evaluated_states_file_name = relative_path+'data/evaluated_states.txt'

        self.task_measure_file_name = relative_path+'data/task_measure.txt'

        self.neural_network_param_file_name = relative_path+'data/weights.txt'

        self.baseline_file_name = relative_path+'data/baseline.txt'
        self.advantageageages_file_name = relative_path+'data/advantages.txt'
        self.unnorm_advantageages_file_name = relative_path+'data/unnorm_advantages.txt'

        self.loss_file_name = relative_path+'data/losses.txt'
        self.log_likelihood_file_name = relative_path+'data/log_likelihood.txt'

        self.PG_file_name = relative_path+'data/PG.txt'

        self.loss_grads_file_name = relative_path+'data/loss_grads.txt'

        self.gradients_file_name = relative_path+'data/gradients.txt'
        self.nn_mean_file_name = relative_path+'data/means.m'
        self.tdyn_file_name = relative_path+'data/tdyn.m'

        self.reset_files([self.neural_network_param_file_name, self.advantageageages_file_name, self.unnorm_advantageages_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.explored_states_file_name, self.evaluated_states_file_name, self.disc_reward_file_name, self.action_dist_mean_file_name,
                             self.task_measure_file_name, self.exploration_file_name, self.loss_file_name, self.log_likelihood_file_name, self.gradients_file_name, self.nn_mean_file_name, self.tdyn_file_name,
                             self.PG_file_name, self.loss_grads_file_name])
        f_handle = file(self.nn_mean_file_name,'a')
    	f_handle.write("mean%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode));
    	f_handle.close()
    	f_handle = file(self.tdyn_file_name,'a')
    	f_handle.write("tdyn%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode));
    	f_handle.close()

    # Opens and closes a file to empty its content
    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()


    def store_weights(self):

        var = np.array([])
        for trainable_variable in self.var_list:
            var = np.append(var, trainable_variable.eval(session=self.sess))

        self.network_params.append(var)

        f_handle = file(self.neural_network_param_file_name,'a')
        np.savetxt(f_handle, [var] , delimiter='\t')
        f_handle.close()


    def store_rewards(self):
        f_handle = file(self.reward_file_name,'a')
        np.savetxt(f_handle, [self.episode_reward], delimiter='\t')
        f_handle.close()

        f_handle = file(self.disc_reward_file_name,'a')
        np.savetxt(f_handle, [self.episode_disc_reward], delimiter='\t')
        f_handle.close()

    def save_matrix_data_to_file(self, filename, data):
        f_handle = file(filename,'a')
        for inner_list in data:
            for elem in inner_list:
                f_handle.write(str(elem)+" ")
        f_handle.write("\n")
        f_handle.close()

    def store_mean_action(self):
        self.save_matrix_data_to_file(self.action_dist_mean_file_name, self.mean_action)

    def store_task_measure(self):
        self.save_matrix_data_to_file(self.task_measure_file_name, self.task_measure)

    def store_exploration_states(self):
        self.save_matrix_data_to_file(self.explored_states_file_name, self.states)

    def store_evaluation_states(self):
        self.save_matrix_data_to_file(self.evaluated_states_file_name, self.states)

    def store_actions(self):
        self.save_matrix_data_to_file(self.actions_file_name, self.actions)

    def store_exploration(self):
        self.save_matrix_data_to_file(self.exploration_file_name, np.concatenate(self.exploration))

    def store_eval_action(self):
        self.save_matrix_data_to_file(self.eval_action_file_name, self.eval_actions)

    def save_vector_data_to_file(self, filename, data):
        f_handle = file(filename,'a')
        for elem in data:
            f_handle.write(str(elem)+" ")
        f_handle.write("\n")
        f_handle.close()


    def get_undiscounted_reward(self):
        self.task_measure.pop(0)
        return self.calculate_return(self.task_measure)

    def compute_episode_data(self):
        self.exploration.pop()

        self.mean_action.pop()

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))

	#print "Task measures shapes are "
	#print self.task_measure
        self.task_measure.pop(0)
	#print self.task_measure
        self.episode_reward = self.calculate_return(self.task_measure)
	#print "Episode rewards ",self.episode_reward
        self.all_returns.append(self.episode_reward)

        self.episode_disc_reward = self.discount_rewards(self.episode_reward) 
        self.all_disc_returns.append(self.episode_disc_reward)

        self.states.pop()
        self.all_states.append(np.asarray(self.states))

    def store_episode_data_to_file(self):
        self.store_mean_action()
        self.store_actions()
        self.store_task_measure()
        self.store_rewards()
        self.store_exploration_states()
        self.store_exploration()
    
    def store_batch_data_to_file(self, batch_data):
        for key in batch_data:
            self.save_vector_data_to_file(key, batch_data[key])

        self.store_weights()

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
        self.store_eval_action()
        self.store_evaluation_states()
        self.reset_episode()
        self.eval_episode = False
        self.init = True
        # Reset the data belonging to a batch, i.e. empty the lists storing all data used for that particular batch
    def reset_batch(self):

        self.eval_episode = True

        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_actions[:]      = []
        self.all_states[:]  = []
        self.all_task_measure[:] = []
        self.network_params.pop(0)
        # Calculates the average of the outputs generated by the neural network. In this case it resembles
        # the mean value for all the actions. They are used to set the variance for the exploration
    def get_action_mean(self):
        # print "Mean of actions: " + str(abs(np.mean(self.actions,axis=0)))
        return abs(np.mean(np.abs(self.actions),axis=0))

        # Calculates the disounted returns from one episode. The cumulative disounted return (R) starts summing from the last reward (r) at time T.
        # At time T-1 the return is R(T-1)=r(t-1)+gamma*r(T),  at time T-2 it is R(T-2)=r(t-2)+gamma*r(t-1), and so on until R(0)=r(0)+gamma*r(1).
    def discount_rewards(self, reward):

        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        return discounted_r

        #This function calculates the reward recieved from doing an action. The reward is calculated as
        # total reward = -w1*dist+w2*exp(-w3*dist) where dist is the vector distance from the current point
        # to the constraint, i.e. the task error.
    def calculate_return(self, curr_reward):

        # squared_points = np.square(np.asarray(curr_reward))
        dist_abs = np.abs(np.asarray(curr_reward))

        rollout_return = 0
        
        dist = (np.sum(dist_abs,axis=1))
        rollout_return = -np.log(dist)
        
        return rollout_return

        # Concatenates a list of lists to one vector
    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec

        # Calculates the advantage function by substracting a baseline from the return at each time step. 
        # The baseline predicts a value representing how good that state is and is approximated as a neural
        # network. Advanage functions are explained in https://arxiv.org/abs/1506.02438
    def calculate_advantages(self, rewards):
        advantages = np.zeros_like(rewards)
        for i in xrange(len(rewards)):
            advantages[i] = rewards[i]

        advantages = advantages.reshape(advantages.shape[0],1)
        return advantages

    def check_convergence(self):
        param_diff = np.linalg.norm(self.network_params[0]-self.network_params[1])
        print "Param diff ", param_diff
        if param_diff < self.epsilon:
            return True
        else:
            return False

        # This function updates the policy according to the natural policy gradient. 
    def policy_search(self, req):
        with self.g.as_default():
            

            # Do an evaluation episode (episode with no noise) every self.eval_episode
            if self.eval_episode:
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
                    self.lr += 0.001
                # If the policy is not improving reset the learning rate to a base value.
                else:
                    self.lr = 0.001
                    print "Policy got worse by", diff

                # The variance of the Gaussian distribution is set as the mean of the actions from the evaluation episode.
                # In this way the exploration is high enough

                # self.sigma = 15*self.get_action_mean()
        		# self.sigma = self.get_action_mean()/3.
                # if self.num_eval_episode == 1 :
                self.sigma = 5*self.get_action_mean()#/4
                # else:
        		    # self.sigma = 0.95*self.sigma
        		# print "SIGMA is ",self.sigma
                        # However, the variance is limited upwards otherwise the expolarion can be too high and damage the real robot

        		#self.sigma[self.sigma>0.4]=0.4
                print "Final task error " + str(self.states[-1])

                self.reset_eval_episode(curr_eval_return)

        		#clean up output file
                f_handle = file(self.nn_mean_file_name,'a')
                f_handle.write("];\n")
                f_handle.write("mean%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode))
                f_handle.close()
                f_handle = file(self.tdyn_file_name,'a')
                f_handle.write("];\n")
                f_handle.write("tdyn%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode))
                f_handle.close()
                return PolicySearchResponse(not self.train)

            # First we compute the necessary data based on one episode
            self.compute_episode_data()
            # Here we store the data computed for that episode
            self.store_episode_data_to_file()

            self.num_train_episode +=1
            print "Training episode number "+str(self.num_train_episode)+" finished with a reward of " + str(self.episode_reward.sum())# + " and final task error " +  str(self.states[-1]) 
    	    self.random_bias = np.random.normal(self.mean, self.sigma)
    	    # print "Random bias for next rollout is" + str(self.random_bias)

            # The policy is only updated if the number of training episodes match the batch size
            if self.num_train_episode % self.batch_size == 0 and self.train:

                print "Updating policy"

                #Concatenating all relevant data for each training episode into one 1D vector
                rewards = np.concatenate(self.all_disc_returns) 
                states = np.concatenate(self.all_states)
                actions = np.concatenate(self.all_actions)

                # Reshape the actions and states to be the same shape as their individual placeholder
                actions = actions.reshape(actions.shape[0],self.action_placeholder.shape[1])
                states = states.reshape(states.shape[0],self.state_placeholder.shape[1])

                # Calculate the advantages. The unnormalized advantages are not used but stored in a file such that it can be analyzed afterhand 
                advantage = self.calculate_advantages(rewards)

                # Calculate the mean of all returns for the batch. It is only done to see if the return between batches increases
                curr_batch_mean_return = np.mean([ret.sum() for ret in self.all_returns])

                print "mean of batch rewards is "+ str(curr_batch_mean_return)

                # If the policy has not converged or it is an evaluation episode than train the policy
                if self.train:


                    # Load all batch data into a dictionary
                    feed_dict={self.state_placeholder  : states,
                               self.action_placeholder : actions,
                               self.advantage          : advantage,
                               self.var                : np.power(self.sigma,2),
                               self.learning_rate      : self.lr}


                    # Here the NPG are calculated. This is only done for the sole purpose of storing the gradients such that
                    # they can be plotted and analyzed in afterhand
                    loss_grads = self.flattenVectors(self.sess.run([g for g,v in self.loss_grads],feed_dict))
                    pg = self.sess.run(self.pg, feed_dict)
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


                    # This dictionary holds all the batch data recently used. In this way the dictionary can be passed
                    # to a function which in turn saves all the data to files.
                    batch_dic = {}
                    # batch_dic[self.baseline_file_name] = baseline
                    batch_dic[self.advantageageages_file_name] = advantage.flatten()
                    # batch_dic[self.unnorm_advantageages_file_name] = unnorm_advantages.flatten()
                    batch_dic[self.log_likelihood_file_name] = [ll]
                    batch_dic[self.loss_file_name] = [error]
                    batch_dic[self.loss_grads_file_name] = loss_grads
                    batch_dic[self.PG_file_name] = self.flattenVectors(pg)
                    # Function that stores the batch data in corresponding files

                    self.store_batch_data_to_file(batch_dic)

                    if self.check_convergence():
                        self.train = False
                    else:
                        self.train = True
                    self.reset_batch()

                    # Trains the value net with the task errors (e) as input and the cumulated rewards from that state onwards as output.
                    # The training is done after the batch update to not add bias
                    print "Training Value Network"


            self.reset_episode()
    	    #clean up output file
    	    f_handle = file(self.nn_mean_file_name,'a')
    	    f_handle.write("];\n")
    	    f_handle.write("mean%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode))
    	    f_handle.close()
    	    f_handle = file(self.tdyn_file_name,'a')
    	    f_handle.write("];\n")
    	    f_handle.write("tdyn%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode))
    	    f_handle.close()

            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures)
            self.states.append(req.task_measures)
            feed_dict = {self.state_placeholder: np.array([req.task_measures])}
            self.ffnn_mean = self.sess.run(self.ff_NN_train, feed_dict)
           
    	    f_handle = file(self.nn_mean_file_name,'a')
    #f_handle.write("mean%i (:,:) = [" % (self.num_train_episode+1));
    # f_handle.write("];")
    	    np.savetxt(f_handle, self.ffnn_mean, fmt='%.5f', delimiter=',')
    	    f_handle.close()
		
            if self.train and not self.eval_episode:
                # Sample the noise
                noise = np.random.normal(np.zeros_like(self.mean), self.sigma)
                # random_noise = np.random.normal(self.mean, [0.002,0.002]) # should find some good way to choose this: can't be more than the controller handles in a single time step
                # The new task dynamics is the mean of the output from the NN plus the noise
                if self.init:
                    self.prev_action = self.ffnn_mean
                    init= False
                task_dynamics = 0.9*self.prev_action+0.1*(noise)#(mean+noise)
                # task_dynamics = mean+noise
                # self.exploration.append(mean-task_dynamics)

                # task_dynamics = self.ffnn_mean+random_noise+self.random_bias  #0.2*self.prev_action+0.8*(self.ffnn_mean+noise)#(mean+noise)
                self.exploration.append(self.ffnn_mean-task_dynamics)
                self.prev_action = task_dynamics
                self.mean_action.append(self.ffnn_mean.flatten())
            else:
                # task_dynamics = mean
                # self.eval_actions.append(mean.flatten())
                # Store the output from the neural network which is the action with no noise added to it
                task_dynamics = self.ffnn_mean.flatten()
    	    #plot task dynamics
    	    f_handle = file(self.tdyn_file_name,'a')
    	    np.savetxt(f_handle, task_dynamics, fmt='%.5f', delimiter=',')
    	    f_handle.close()

            # Store the output from the neural network which is the action with no noise added to it
            self.mean_action.append(self.ffnn_mean.flatten())
            self.actions.append(task_dynamics)
            return  task_dynamics.flatten()

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('VPG')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass47
