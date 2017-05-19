#!/usr/bin/env python
import roslib

import rospy
import sys
import bisect
import itertools
from utils import *
from grasp_learning.srv import QueryNN
from grasp_learning.srv import *
from std_msgs.msg import Empty
from copy import deepcopy
from math import sqrt
from math import pow
import tensorflow as tf
import numpy as np
import os


class ValueNet(object):

    def __init__(self, num_inputs , num_outputs, path):

        self.vg = tf.Graph()
        self.session = tf.InteractiveSession(graph=self.vg)
        self.net = None

        self.num_inputs = num_inputs
        self.num_outputs = 1
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_inputs], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.batch_size = tf.placeholder(tf.float32, name="batch_size")
        self.construct_ValueNet()
	self.vg_writer = tf.summary.FileWriter(path+'vgraphs',self.vg)
        self.initialize = True

    def construct_ValueNet(self):
        with self.vg.as_default():

            hidden_size = 64

            weight_init = tf.random_uniform_initializer(-0.05, 0.05)
            bias_init = tf.constant_initializer(0)

            h1 = tf.nn.tanh(fully_connected(self.x, self.num_inputs, hidden_size, weight_init, bias_init, "h1"))
            h2 = tf.nn.tanh(fully_connected(h1, hidden_size, hidden_size, weight_init, bias_init, "h2"))
            h3 = fully_connected(h2, hidden_size, 1, weight_init, bias_init, "h3")
            self.net = tf.reshape(h3, (-1,))
            self.l2 = tf.nn.l2_loss(self.net - self.y)/self.batch_size
            self.optimizer = tf.train.AdamOptimizer().minimize(self.l2)
	    #freezing the value graph
            self.session.run(tf.global_variables_initializer())
	    self.vg.finalize()

    def reshape_input(self, input_):
        return input_.reshape(1,self.num_inputs)

    def reshape_output(self, output_):
        return output_.reshape(output_.shape[0],self.num_outputs)


    def predict(self, states):
        with self.vg.as_default():
            if self.initialize:
                return np.asarray([0])
            else:
                ret = self.session.run(self.net, {self.x: self.reshape_input(states)})
                return ret

    def train(self, input_data, output_data, batch_size = 1.0, n_iter=50):
        with self.vg.as_default():
            self.initialize = False
            for _ in range(1000):
                self.session.run(self.optimizer, {self.x: input_data, self.y: output_data, self.batch_size: batch_size})


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
        self.num_train_episode = 0
        self.num_eval_episode = 0
    	self.gamma = 0.99
    	self.kl = 0.01

        self.g = tf.Graph()
        self.train = True
        self.eval_episode = True
        self.max_rew_before_convergence = 15000
        self.mean = np.zeros(num_outputs)
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
        self.all_task_measure = []

        self.mean_action = []

        # This list holds the natural policy gradients which is composed of the inverse of the fisher
        # matrix mulitpled by the policy gradient. For more info check http://www.scholarpedia.org/article/Policy_gradient_methods
        self.NPG = []

        self.prev_eval_mean_return = 0

        self.VN = ValueNet(num_inputs, num_outputs, self.relative_path)

        self.sess = tf.InteractiveSession(graph=self.g) #,config=tf.ConfigProto(log_device_placement=True))

        # Placeholder to which the future task errors and task dynamics are loaded into 
        self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs],name="Task_error_placeholder") 
        self.action_placeholder = tf.placeholder(tf.float32, [None, num_outputs], name="Task_dynamics_placeholder")
        # Placeholder for the learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate_placeholder")

        # Placeholder for the advantage function
        self.advantage = tf.placeholder(tf.float32,[None,num_rewards], name="Advantage_placeholder")

        # Placehoalder for the variance of the added noise
        self.var = tf.placeholder(tf.float32, [num_outputs], name="Variance_placeholder")

        # List that will containg the placehoalder for the fisher matrices
        self.fisher_matrix = []

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
        # self.loss = -tf.reduce_mean(self.loglik * self.advantage, name="Loss function")
        # Get the list of all trainable variables in our tensorflow graph
        # self.var_list = tf.trainable_variables()
        # Compute the analytic gradients of the loss function given the trainable variables in our graph
        # self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list, name="Loss_function_gradients")

    	tf.summary.histogram('loglik', self.loglik)
    	tf.summary.histogram('adv', self.advantage)

    	with tf.device('/cpu:0'):
    	    self.loss_prod = tf.multiply(self.loglik,self.advantage,'mrNasty')
            self.loss = -tf.reduce_mean(self.loss_prod,name="Loss_function")/self.batch_size
            
            tf.summary.histogram('loss', self.loss)
    	self.merged_summary = tf.summary.merge_all()

        # Get the list of all trainable variables in our tensorflow graph
        self.var_list = tf.trainable_variables()
        # Compute the analytic gradients of the loss function given the trainable variables in our graph
    	#with tf.device('/cpu:0'):
    	self.loss_grads = self.optimizer.compute_gradients(self.loss, self.var_list)
        # Calculate the gradients of the policy
        self.pg = tf.gradients(self.loglik, self.var_list,name="Policy_gradients")

        self.set_natural_policy_gradients()
        
        self.train_op = self.optimizer.apply_gradients(self.loss_grads)

        saver = tf.train.Saver()

        if load_example_model:
            saver.restore(self.sess, self.relative_path+'models/'+model_name)
        else:
            save_path = saver.save(self.sess, self.relative_path+'models/'+model_name)

        self.store_weights()
                    
    	#setup tensor board writers
    	self.train_writer = tf.summary.FileWriter(self.relative_path+'/graphs',self.g)
    	# self.sess.run(tf.global_variables_initializer())
    	#freezing the main graph
    	# self.g.finalize()


    # The natural policy gradient (NPG) is the inverse of the fisher matrix times the gradient.
    # In this function the NPG is set by muliplying a placeholder storing the inverse of the fisher
    # matrix with the corresponding policy gradient. The policy gradient is the gradient of the neural network (NN)
    # and in tensorflow the gradient of the NN is split into a list where each entry corresponds to the parameters
    # in each layer
    def set_natural_policy_gradients(self):
        i=1
        for g,v in self.loss_grads:
            self.fisher_matrix.append(tf.placeholder(tf.float32, g.shape,name="Fisher_"+str(i)))
            grad = tf.multiply(self.fisher_matrix[-1], g,name="PG_Fisher_Mul_"+str(i))
            self.NPG.append((grad, v))
            i+=1
    # Opens and closes a file to empty its content
    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()

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
            layer_1_outputs = tf.nn.softplus(tf.matmul(task_error_placeholder, weights_1,name="Input_W1_Mul") + biases_1,name="L1")

            # weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, NUM_OUTPUTS]),name="w2")
            # biases_2 = tf.Variable(tf.zeros([NUM_OUTPUTS]),name="b2")
            # self.ff_NN_train = tf.add(tf.matmul(layer_1_outputs, weights_2,name="L1_W2_Mul"),biases_2,name="output")


            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, HIDDEN_UNITS_L2]),name="w2")
            biases_2 = tf.Variable(tf.zeros([HIDDEN_UNITS_L2]),name="b2")
            layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)

            weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, NUM_OUTPUTS]))
            biases_3 = tf.Variable(tf.zeros([NUM_OUTPUTS]))
            self.ff_NN_train = tf.add(tf.matmul(layer_2_outputs, weights_3,name="L1_W2_Mul"),biases_3,name="output")

            # logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(self.ff_NN_train, task_dyn_placeholder)),0)

            train_op = self.optimizer.minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],NUM_OUTPUTS)
            states= states.reshape(states.shape[0],NUM_INPUTS)

            feed_dict={task_error_placeholder: states,
                        task_dyn_placeholder: task_dyn,
                        self.learning_rate : 0.01} 

            for i in range(2000):
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

        self.NPG_file_name = relative_path+'data/NPG.txt'

        self.PG_file_name = relative_path+'data/PG.txt'

        self.fisher_file_name = relative_path+'data/fisher.txt'

        self.loss_grads_file_name = relative_path+'data/loss_grads.txt'

        self.reset_files([self.neural_network_param_file_name, self.advantageageages_file_name, self.unnorm_advantageages_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.explored_states_file_name, self.evaluated_states_file_name, self.disc_reward_file_name, self.action_dist_mean_file_name,
                             self.task_measure_file_name, self.exploration_file_name, self.loss_file_name, self.log_likelihood_file_name, self.NPG_file_name, self.PG_file_name, self.PG_file_name,
                             self.fisher_file_name, self.loss_grads_file_name, self.eval_action_file_name])


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
        self.all_task_measure[:] = []

        # Calculates the average of the outputs generated by the neural network. In this case it resembles
        # the mean value for all the actions. They are used to set the variance for the exploration
    def get_action_mean(self):
        print "Mean of actions: " + str(abs(np.mean(self.actions,axis=0)))
        return abs(np.mean(self.actions,axis=0))

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
        dist_square = np.sum(squared_points,axis=1)
        # rollout_return = -10*dist-1.5*np.log(alpha+10*dist)
        # rollout_return = -50*dist+30*np.exp(-10*dist)
        rollout_return = -1*(10*dist+1*np.log(alpha+dist))

        # rollout_return += -10*np.log(alpha+10*dist_abs[:,e])#-10000*dist_abs[:,e]-10*np.log(alpha+15*dist_abs[:,e])#0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
        
        return rollout_return

        #Normalizes the data to have mean 0 and variance 1
    def normalize_data(self, data):
        print "Mean and variance of data is " + str(np.mean(data,axis=0)) + " " + str(np.std(data,axis=0))
        return (data-np.mean(data,axis=0))/(np.std(data,axis=0)+1e-8)


        # Calculates the inverse of the fisher matrix used for the natural policy gradient
        # http://www.scholarpedia.org/article/Policy_gradient_methods
    def calculate_inverse_Fisher_Matrix(self, input_data, output):
        with self.g.as_default():
            pg = np.asarray(self.sess.run(self.pg, feed_dict={self.state_placeholder:input_data, self.action_placeholder : output, self.var : np.power(self.sigma,2)}))
            fisher = []
            eps = 1e-8
            for g in pg:
                fisher.append((np.asarray(1/(np.square(g.flatten())+eps)).reshape(g.shape)))#/self.batch_size)

            return fisher, pg

        # Concatenates a list of lists to one vector
    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec
        # Calculates the step size for the gradient descent suggesed by Jan Peters and Stefan Schaal in their paper
        # "Reinforcement learning of motor skills with policy gradients" 
        # http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf 
    def calculate_learning_rate(self, fisher, pg):
        flatten_fisher = self.flattenVectors(fisher)
        flatten_pg = self.flattenVectors(pg)
        flatten_pg = flatten_pg.reshape(flatten_pg.shape[0],1)
        flatten_fisher = flatten_fisher.reshape(flatten_fisher.shape[0],1)
        eps = 1e-8
        denom = eps+np.square(flatten_pg).T.dot(flatten_fisher)
        step_size = np.sqrt(self.kl/denom)

        print "STEPSIZE"
        print step_size
        return step_size

        # Calculates the advantage function by substracting a baseline from the return at each time step. 
        # The baseline predicts a value representing how good that state is and is approximated as a neural
        # network. Advanage functions are explained in https://arxiv.org/abs/1506.02438
    def calculate_advantages_with_VN(self, rewards, states):
        advantages = np.zeros_like(rewards)
        baseline = np.zeros_like(rewards)
        for i in xrange(len(states)):
            baseline[i] = self.VN.predict(states[i,:]).flatten()
            advantages[i] = rewards[i]#-np.abs(baseline[i])

        norm_advantages = self.normalize_data(advantages)
        advantages = advantages.reshape(advantages.shape[0],1)
        norm_advantages = norm_advantages.reshape(norm_advantages.shape[0],1)
        return norm_advantages, advantages, baseline


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
                # If the return from the evaluation episode is higher than a threshold value the policy has converged
                if curr_eval_return>self.max_rew_before_convergence:
                    print "Policy converged in " +str(self.num_train_episode+self.num_eval_episode)+" episodes!"
                    self.train = False
                # If the difference is positive meaning that the policy is improving than we increase the kl divergence.
                # This corresponds to a higher learning rate and that the new policy will differ more than the current
                elif diff>0:
                    print "Policy improved by", diff
                    # self.kl += 0.01
                # If the policy is not improving reset the kl divergence to a base value.
                else:
                    # self.kl = 0.01 
                    print "Policy got worse by", diff

                # The variance of the Gaussian distribution is set as the mean of the actions from the evaluation episode.
                # In this way the exploration is high enough
                self.sigma = 15*self.get_action_mean()
                # However, the variance is limited upwards otherwise the expolarion can be too high and damage the real robot
                # self.sigma[self.sigma<0.4]=0.4
                print "Final task error " + str(self.states[-1])
                self.reset_eval_episode(curr_eval_return)
                return PolicySearchResponse(not self.train)

            # First we compute the necessary data based on one episode
            self.compute_episode_data()
            # Here we store the data computed for that episode
            self.store_episode_data_to_file()

            self.num_train_episode +=1
            print "Training episode number "+str(self.num_train_episode)+" finished with a reward of " + str(self.episode_reward.sum()) 

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
                advantage, unnorm_advantages, baseline = self.calculate_advantages_with_VN(rewards, states)

                # Calculate the mean of all returns for the batch. It is only done to see if the return between batches increases
                curr_batch_mean_return = np.mean([ ret.sum() for ret in self.all_returns])

                print "mean of batch rewards is "+ str(curr_batch_mean_return)

                # If the policy has not converged or it is an evaluation episode than train the policy
                if self.train:

                    fisher, pg = self.calculate_inverse_Fisher_Matrix(states, actions)
                    learning_rate = float(self.calculate_learning_rate(fisher, pg))


                    # Load all batch data into a dictionary
                    feed_dict={self.state_placeholder  : states,
                               self.action_placeholder : actions,
                               self.advantage          : advantage,
                               self.var                : np.power(self.sigma,2),
                               self.learning_rate      : learning_rate}

                    i=0
                    # The fisher matrix is loaded differently due to that it is split up into a list
                    for f in fisher:
                        feed_dict[self.fisher_matrix[i]] = f
                        i+=1

                    # Here the NPG are calculated. This is only done for the sole purpose of storing the gradients such that
                    # they can be plotted and analyzed in afterhand
                    NPG = self.flattenVectors(self.sess.run([grad[0] for grad in self.NPG],feed_dict))
                    loss_grads = self.flattenVectors(self.sess.run([g for g,v in self.loss_grads],feed_dict))

                    unum = int (self.num_train_episode / self.batch_size)

                    # run i numbers of gradient descent updates
        		    # setup run options for profiling
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    for i in range(1):
                        _, error, ll, summary, loss_prod = self.sess.run([self.train_op, self.loss, self.loglik, self.merged_summary, self.loss_prod], feed_dict,
        				run_options,
        				run_metadata)
                    
                    #add summaries for this training run
                    self.train_writer.add_summary(summary,unum)
                    self.train_writer.add_run_metadata(run_metadata, 'step%03d' % unum)
                    print('Adding run metadata for', unum)


                    # This dictionary holds all the batch data recently used. In this way the dictionary can be passed
                    # to a function which in turn saves all the data to files.
                    batch_dic = {}
                    batch_dic[self.baseline_file_name] = baseline
                    batch_dic[self.advantageageages_file_name] = advantage.flatten()
                    batch_dic[self.unnorm_advantageages_file_name] = unnorm_advantages.flatten()
                    batch_dic[self.log_likelihood_file_name] = [ll]
                    batch_dic[self.loss_file_name] = [error]
                    batch_dic[self.NPG_file_name] = NPG
                    batch_dic[self.loss_grads_file_name] = loss_grads
                    batch_dic[self.PG_file_name] = self.flattenVectors(pg)
                    batch_dic[self.fisher_file_name] = self.flattenVectors(fisher)
                    # Function that stores the batch data in corresponding files
                    self.store_batch_data_to_file(batch_dic)
                    # Trains the value net with the task errors (e) as input and the cumulated rewards from that state onwards as output.
                    # The training is done after the batch update to not add bias
                    print "Training Value Network"
                    self.VN.train(states, rewards, self.batch_size)
                    self.reset_batch() 
		    

            self.reset_episode()

            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures)
            self.states.append(req.task_measures)
            feed_dict = {self.state_placeholder: np.array([req.task_measures])}
            mean = self.sess.run(self.ff_NN_train, feed_dict)
            if self.train and not self.eval_episode:
                # Sample the noise
                noise = np.random.normal(self.mean, self.sigma)
                # The new task dynamics is the mean of the output from the NN plus the noise
                # task_dynamics = 0.8*self.prev_action+0.2*(mean+noise)#(mean+noise)
                task_dynamics = mean+noise
                self.exploration.append(mean-task_dynamics)
                self.prev_action = task_dynamics
                self.mean_action.append(mean.flatten())
            else:
                task_dynamics = mean
                self.eval_actions.append(mean.flatten())
            # Store the output from the neural network which is the action with no noise added to it
            self.actions.append(task_dynamics[0])
            return  task_dynamics.flatten()

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('policy_gradient_OP_exp')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass47
