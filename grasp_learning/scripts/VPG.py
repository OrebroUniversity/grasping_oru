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
from math import sqrt
from math import pow
import tensorflow as tf
import numpy as np
import os



class Policy(object):

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        load_example_model = rospy.get_param('~load_model', ' ')
        model_name = rospy.get_param('~model_name', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        num_hidden_layers = rospy.get_param('~num_hidden_layers', ' ')
        hidden_layers_sizes  =  rospy.get_param('~hidden_layers_sizes', ' ')
        self.lr = rospy.get_param('~learning_rate', '1')
        self.subtract_baseline = rospy.get_param('~subtract_baseline', '0')
        self.batch_size = rospy.get_param('~batch_size', '5')
        self.gamma = rospy.get_param('~discount_factor', '0.99') #TSV: testing a much more local approach
        self.use_normalized_advantages = rospy.get_param('~use_normalized_advantages', '0')
        self.relative_path = rospy.get_param('~relative_path', ' ')

        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.sigma = np.zeros(num_outputs)
        self.random_bias = np.zeros(num_outputs)
        self.num_train_episode = 0
        self.num_eval_episode = 0

        self.g = tf.Graph()
        self.train = True
        self.eval_episode = True
        self.min_dist_norm = 0.04
        self.mean = np.zeros(num_outputs)
        self.ffnn_mean = np.zeros(num_outputs)

        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_disc_returns = []
        self.exploration = []

        self.all_actions = []
        self.actions = []
        self.eval_actions = []
        
        self.states = []
        self.all_states = []

        self.task_measure = []

        self.mean_action = []

        # This list holds the natural policy gradients which is composed of the inverse of the fisher
        # matrix mulitpled by the policy gradient. For more info check http://www.scholarpedia.org/article/Policy_gradient_methods

        self.prev_eval_mean_return = 0

        self.VN = NeuralNetworkValueFunction(num_inputs, num_outputs, self.relative_path)

        self.sess = tf.InteractiveSession(graph=self.g) #,config=tf.ConfigProto(log_device_placement=True))

        # Placeholder for the states i.e. task errors e 
        self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs],name="Task_error_placeholder") 
        # Placeholder for the actions i.e. task dynamics e_dot_star 
        self.action_placeholder = tf.placeholder(tf.float32, [None, num_outputs], name="Task_dynamics_placeholder")

        # Placeholder for the learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate_placeholder")

        # Placeholder for the advantage function
        self.advantage = tf.placeholder(tf.float32,[None,num_rewards], name="Advantage_placeholder")

        # Placehoalder for the variance of the added noise
        self.var = tf.placeholder(tf.float32, [num_outputs], name="Variance_placeholder")

        # Create the optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        input_data, output_data = self.parse_input_output_data(input_output_data_file)

        self.set_bookkeeping_files(self.relative_path)

        self.ff_NN_train = self.construct_Neural_Network(self.state_placeholder, self.action_placeholder, num_hidden_layers, hidden_layers_sizes)
        self.train_Neural_Network(self.state_placeholder, self.action_placeholder, input_data, output_data)
        # The output from the neural network is the mean of a Gaussian distribution. This variable is simply the
        # log likelihood of that Gaussian distribution
        self.loglik = gauss_log_prob(self.ff_NN_train, self.var, self.state_placeholder)

        self.loss = -tf.reduce_mean(tf.multiply(self.loglik,self.advantage,'loss_prod'), 0, name='loss_reduce_mean')/self.batch_size            

        # Get the list of all trainable variables in our tensorflow graph
        self.var_list = tf.trainable_variables()

        # Compute the analytic gradients of the loss function given the trainable variables in our graph
    	#with tf.device('/cpu:0'):
    	self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list)
        # Calculate the gradients of the policy
        self.pg = tf.gradients(self.loglik, self.var_list,name="Policy_gradients")
        self.lg = tf.gradients(self.loss, self.var_list, name="Loss_gradients")

        with tf.name_scope("VPG"):
            self.train_op = self.optimizer.apply_gradients(self.loss_grads)

        saver = tf.train.Saver()

        if load_example_model:
            saver.restore(self.sess, self.relative_path+'models/'+model_name)
        else:
            save_path = saver.save(self.sess, self.relative_path+'models/'+model_name)

        self.store_weights()
                    
        self.setup_tensor_board()


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

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)


    def construct_Neural_Network(self, input_placeholder, output_placeholder, num_layers, layer_sizes):
        with self.g.as_default():
            prev_layer = input_placeholder
            for i in range(num_layers):
                prev_layer = self.create_layer(prev_layer, layer_sizes[i], tf.nn.tanh,i , "Hidden")

            # last layer is a linear output layer
            num_outputs = int(output_placeholder.shape[1])
            return self.create_layer(prev_layer, num_outputs, None, num_layers, "Output")

    def train_Neural_Network(self, input_placeholder, output_placeholder, input_training_data, output_training_data):

        error_function = tf.reduce_mean(tf.square(tf.subtract(self.ff_NN_train, output_placeholder)),0)

        with tf.name_scope("Train"):
            train_op = self.optimizer.minimize(error_function)

        feed_dict={input_placeholder: input_training_data,
                    output_placeholder: output_training_data,
                    self.learning_rate : 0.001} 

        self.sess.run(tf.global_variables_initializer())

        for i in range(2000):
            _, loss = self.sess.run([train_op, error_function],feed_dict)

        print "Network trained"


    def create_layer(self, prev_layer, layer_size, activation_fun, num_layer, name):
        with tf.name_scope(name):
            with self.g.as_default():
                weights = tf.Variable(tf.truncated_normal([int(prev_layer.shape[1]), layer_size]),name="NN/w"+str(num_layer)+"/t")
                biases = tf.Variable(tf.truncated_normal([layer_size]), name="NN/b"+str(num_layer)+"/t")
                if activation_fun == None:
                    layer_output = tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases,name="NN/output/t")
                else:
                    layer_output = activation_fun(tf.add(tf.matmul(prev_layer, weights,name="NN/Input_W"+str(num_layer)+"_Mul/t"), biases),name="NN/L"+str(num_layer)+"/t")

                return layer_output


    def setup_tensor_board(self):

        tf.summary.histogram('loglik_hist', self.loglik)
        tf.summary.histogram('loss_hist', self.loss)
        tf.summary.histogram('advantages_hist', self.advantage)
        tf.summary.histogram('variance_hist', self.var)
        tf.summary.histogram('task_errors_hist', self.state_placeholder)
        tf.summary.histogram('task_dynamics_hist', self.action_placeholder)

        tf.summary.scalar('learning_rate',self.learning_rate)

        tf.summary.tensor_summary('task_errors', self.state_placeholder)
        tf.summary.tensor_summary('task_dynamics', self.action_placeholder)
        tf.summary.tensor_summary('variance', self.var)
        tf.summary.tensor_summary('loglikelihood', self.loglik)
        tf.summary.tensor_summary('advantages', self.advantage)
        tf.summary.tensor_summary('loss', self.loss)

        self.store_containers_in_tensorboard(self.lg)
        self.store_containers_in_tensorboard(self.pg)
        self.store_containers_in_tensorboard(self.var_list)

        #setup tensor board writers
        self.train_writer = tf.summary.FileWriter(self.relative_path+'/graphs',self.g)
        self.merged_summary = tf.summary.merge_all()
        #freezing the main graph
        self.g.finalize()

    def store_containers_in_tensorboard(self, container):

        if type(container[0]) is tuple:
            for grad,var in container:
                name = grad.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , grad)
                tf.summary.tensor_summary(name[0]+"_"+name[1], grad)
        elif type(container) is list:
            for var in container:
                name = var.name.split("/",2)
                tf.summary.histogram(name[0]+"_"+name[1]+"_hist" , var)
                tf.summary.tensor_summary(name[0]+"_"+name[1], var)

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
        self.eval_rewards_file_name = relative_path+'data/eval_rewards.m'
        self.tdyn_file_name = relative_path+'data/tdyn.m'

        self.reset_files([self.neural_network_param_file_name, self.advantageageages_file_name, self.unnorm_advantageages_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.explored_states_file_name, self.evaluated_states_file_name, self.disc_reward_file_name, self.action_dist_mean_file_name,
                             self.task_measure_file_name, self.exploration_file_name, self.loss_file_name, self.log_likelihood_file_name, self.gradients_file_name, self.eval_rewards_file_name, self.tdyn_file_name,
                             self.PG_file_name, self.loss_grads_file_name])
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

        self.task_measure.pop(0)

        self.episode_reward = self.calculate_return(self.task_measure)
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
        # Reset the data belonging to a batch, i.e. empty the lists storing all data used for that particular batch
    def reset_batch(self):

        self.eval_episode = True

        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_actions[:]      = []
        self.all_states[:]  = []
        # Calculates the average of the outputs generated by the neural network. In this case it resembles
        # the mean value for all the actions. They are used to set the variance for the exploration
    def get_action_mean(self):
        print "Mean of actions: " + str(abs(np.mean(self.actions,axis=0)))
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

        squared_points = np.square(np.asarray(curr_reward))
        dist_abs = np.abs(np.asarray(curr_reward))

        alpha = 1e-5#1e-17
        rollout_return = 0
        
        dist = np.sqrt(np.sum(squared_points,axis=1))
        dist_l1 = np.sum(dist_abs,axis=1)
        #dist = 0.5*(np.sum(dist_abs,axis=1))
        # dist_square = np.sum(squared_points,axis=1)
        # rollout_return = -10*dist-1.5*np.log(alpha+10*dist)
        # rollout_return = -50*dist+30*np.exp(-10*dist)
        # rollout_return = -1*(10*dist+1*np.log(alpha+dist))
    
        #print "Dist square ", dist_square, "\n Dist abs", dist_abs
        # dist = np.sqrt(np.sum(dist_square))
        # dist = 0.5*np.sum(dist_abs,axis=1)
        #print "Dist ", dist 
        #rollout_return = -10*dist-1.5*np.log(alpha+10*dist)

        #rollout_return = -50*dist+25*np.exp(-10*dist)

        #rollout_return = -100*dist-10*np.exp(dist) #TSV original
        
        delta = 0.2
        sq_factor = 10
        lin_factor = 0.05
        #rollout_return = -sq_factor*np.square(dist) #simple quadratic
        rollout_return = -sq_factor*np.square(dist) #TSV new
        rollout_return[dist_l1 > delta] = -sq_factor*delta*delta + lin_factor*(delta-dist_l1[dist_l1 > delta])
        # rollout_return = -100*dist-10*np.square(dist)

        # rollout_return += -10*np.log(alpha+10*dist_abs[:,e])#-10000*dist_abs[:,e]-10*np.log(alpha+15*dist_abs[:,e])#0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
        
        return rollout_return


        # Concatenates a list of lists to one vector
    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec

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
            advantages = self.normalize_data(advantages)
        else:
            advantages = advantages.reshape(advantages.shape[0],1)
        return advantages, baseline

        # This is a naive 
    def check_convergence(self):
        dist_norm = np.linalg.norm(self.states[-1])
        print "Dist norm ", dist_norm
        if dist_norm < self.min_dist_norm:
            return True
        else:
            return False

    def store_files_to_matlab(self):
        #clean up output file
        f_handle = file(self.tdyn_file_name,'a')
        f_handle.write("];\n")
        f_handle.write("tdyn%i_%i (:,:) = [\n" % (self.num_train_episode,self.num_eval_episode))
        f_handle.close()

    def get_batch_data(self):
        return np.concatenate(self.all_disc_returns), np.concatenate(self.all_states), np.concatenate(self.all_actions)

        # This function updates the policy according to the policy gradient. 
    def policy_search(self, req):
        with self.g.as_default():
            
            if self.train==False:
                #clean up output file
                self.num_eval_episode += 1
                print "Converged policy  "+str(self.num_eval_episode)+" finished!"
                self.store_files_to_matlab()
                curr_eval_return = self.get_undiscounted_reward().sum()
                self.reset_eval_episode(curr_eval_return)
                return PolicySearchResponse(not self.train)


            # Do an evaluation episode (episode with no noise) every self.eval_episode
            if self.eval_episode:
                self.num_eval_episode += 1
                print "Evaluation episode number "+str(self.num_eval_episode)+" finished!"
                # Calculates the mean return from the evaluation episode 
                curr_eval_return = self.get_undiscounted_reward().sum()
                print "Average return from evaluation is " + str(curr_eval_return)
                # The difference between the returns of the current and previous evaluation episode
                diff = curr_eval_return-self.prev_eval_mean_return
                f_handle = file(self.eval_rewards_file_name,'a')
                f_handle.write("reward(%i) = %.5f\n" % (self.num_eval_episode, curr_eval_return));
                f_handle.close()

                # If the difference is positive meaning that the policy is improving than we increase the learning rate.
                if diff>0:
                    print "Policy improved by", diff
                    # self.lr += 0.001
                else:
                    # self.lr = 0.001
                    print "Policy got worse by", diff

                # The variance of the Gaussian distribution is set as the mean of the actions from the evaluation episode.
                # In this way the noise is high enough to actually impact the mean of the neural network

                if(self.num_eval_episode == 1) :
                    self.sigma = self.get_action_mean()/3
                else:
                    self.sigma = 0.95*self.sigma
                print "SIGMA is ",self.sigma

                print "Final task error " + str(self.states[-1])
                if self.check_convergence():
                    print "Policy converged"
                    self.train = False
                else:
                    self.train = True

                self.store_files_to_matlab()
                self.reset_eval_episode(curr_eval_return)

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
                rewards, states, actions = self.get_batch_data()

                # Reshape the actions and states to be the same shape as their individual placeholder
                actions = actions.reshape(actions.shape[0],self.action_placeholder.shape[1])
                states = states.reshape(states.shape[0],self.state_placeholder.shape[1])

                # Calculate the advantages. The unnormalized advantages are not used but stored in a file such that it can be analyzed afterhand 
                advantage, baseline = self.calculate_advantages(rewards, states)

                # Calculate the mean of all returns for the batch. It is only done to see if the return between batches increases
                curr_batch_mean_return = np.mean([ret.sum() for ret in self.all_returns])

                print "mean of batch rewards is "+ str(curr_batch_mean_return)

                # As long as the policy has not converged or the episode is an evaluation episode then update the policy

                # Load all batch data into a dictionary
                feed_dict={self.state_placeholder  : states,
                           self.action_placeholder : actions,
                           self.advantage          : advantage,
                           self.var                : np.power(self.sigma,2),
                           self.learning_rate      : self.lr}


                # Here the VPG are calculated. This is only done for the sole purpose of storing the gradients such that
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


                # This dictionary holds all the batch data for one batch. 
                batch_dic = {}
                batch_dic[self.baseline_file_name] = baseline
                batch_dic[self.advantageageages_file_name] = advantage.flatten()
                # batch_dic[self.unnorm_advantageages_file_name] = unnorm_advantages.flatten()
                batch_dic[self.log_likelihood_file_name] = [ll]
                batch_dic[self.loss_file_name] = [error]
                batch_dic[self.loss_grads_file_name] = loss_grads
                batch_dic[self.PG_file_name] = self.flattenVectors(pg)

                # Function that stores the batch data in corresponding files
                self.store_batch_data_to_file(batch_dic)

                self.VN.train(states, rewards, self.batch_size)

                self.reset_batch()


            self.reset_episode()
    	    #clean up output file
            self.store_files_to_matlab()
            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures)
            self.states.append(req.task_measures)
            feed_dict = {self.state_placeholder: np.array([req.task_measures])}
            self.ffnn_mean = self.sess.run(self.ff_NN_train, feed_dict)
		
            if self.train and not self.eval_episode:
                # Sample the noise
                random_noise = np.random.normal(self.mean, [0.02,0.02]) # should find some good way to choose this: can't be more than the controller handles in a single time step
                task_dynamics = 0.3*self.prev_action+0.7*(self.ffnn_mean+random_noise+self.random_bias)  #0.2*self.prev_action+0.8* 0.2*self.prev_action+0.8*(self.ffnn_mean+noise)#(mean+noise)

                self.exploration.append(self.ffnn_mean-task_dynamics)
                self.prev_action = task_dynamics
                self.mean_action.append(self.ffnn_mean.flatten())
            else:
                task_dynamics = self.ffnn_mean
    	    #plot task dynamics
    	    f_handle = file(self.tdyn_file_name,'a')
    	    np.savetxt(f_handle, task_dynamics, fmt='%.5f', delimiter=',')
    	    f_handle.close()

            # Store the output from the neural network which is the action with no noise added to it
            self.mean_action.append(self.ffnn_mean.flatten())
            self.actions.append(task_dynamics[0])
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
