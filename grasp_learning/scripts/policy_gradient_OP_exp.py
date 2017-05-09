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



class ValueNet(object):

    def __init__(self, num_inputs , num_outputs):

        self.vg = tf.Graph()
        self.session = tf.InteractiveSession(graph=self.vg)
        self.net = None

        self.num_inputs = num_inputs
        self.num_outputs = 1
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_inputs], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.construct_ValueNet()
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
            self.l2 = tf.nn.l2_loss(self.net - self.y)

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
            train = tf.train.AdamOptimizer().minimize(self.l2)
            self.session.run(tf.global_variables_initializer())

            for _ in range(50):
                self.session.run(train, {self.x: input_data, self.y: output_data})


class Policy(object):
    """docstring for Policy"""

    def __init__(self):

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        load_example_model = rospy.get_param('~load_model', ' ')
        example_model_name = rospy.get_param('~model_name', ' ')
        hidden_layer_size  =  rospy.get_param('~hidden_layer_size', ' ')         
        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.sigma = np.zeros(num_outputs)
        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.gamma = 0.99
        self.batch_size = 10
        self.kl = 0.01

        self.g = tf.Graph()
        self.train = True
        self.eval_episode = True
        self.max_rew_before_convergence = 1500
        self.mean = np.zeros(num_outputs)
        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_unnormalized_returns = []
        self.all_disc_returns = []
        self.exploration = []
        
        self.all_actions = []
        self.all_actions_dist = []
        self.actions = []
        
        self.task_errors = []
        self.all_task_errors = []

        self.task_measure = []
        self.all_task_measure = []

        self.NN_output = []

        self.prev_eval_mean_return = 0

        self.VN = ValueNet(num_inputs, num_outputs)

        self.sess = tf.InteractiveSession(graph=self.g)

        self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs])
        self.action_placeholder = tf.placeholder(tf.float32, [None, num_outputs])

        self.advantage = tf.placeholder(tf.float32,[None,num_rewards])

        self.var = tf.placeholder(tf.float32, [num_outputs])

        input_data, output_data = self.parse_input_output_data(input_output_data_file)

        self.set_bookkeeping_files()

        self.construct_ff_NN(self.state_placeholder, self.action_placeholder , input_data, output_data, num_inputs, num_outputs, hidden_layer_size)
                    
        # self.loglik = (1.0/self.var)*(self.action_placeholder - self.ff_NN_train)
        self.loglik = gauss_log_prob(self.ff_NN_train, self.var, self.state_placeholder)

        self.loss = -tf.reduce_mean(self.loglik * self.advantage) 

        saver = tf.train.Saver()

        if load_example_model:
           saver.restore(self.sess, example_model_name)
        else:
             save_path = saver.save(self.sess, "/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/model.ckpt")

        self.store_weights()

    def reset_files(self, file_names):
        for file in file_names:
            open(file, 'w').close()

    def parse_input_output_data(self, input_file):
        input_data = []
        output_data = []
        input_ = []
        output_ = []
        i = 0
        with open(input_file, 'rU') as f:
            for line in f:
                if (i==0 or i==1):
                # if (i==0):
                    i+=1
                    continue
                line = line.split()
                for string in xrange(len(line)):
                    if string%2==0:
                        # input_data.append(float(line[string]))
                        input_data.append(float(line[string])+np.random.normal(0, 0.2))

                    else:
                        # if string == 1:
                        output_data.append(float(line[string]))
                        # output_data.append(float(line[string])+np.random.normal(0, 0.5))
                        # output_data.append(float(line[string]))

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)


    def construct_ff_NN(self, task_error_placeholder, task_dyn_placeholder, task_errors, task_dyn, NUM_INPUTS = 1, NUM_OUTPUTS = 1, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 1):
        with self.g.as_default():

            weights_1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, HIDDEN_UNITS_L1]),name="w1")
            biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]), name="b1")
            layer_1_outputs = tf.nn.tanh(tf.matmul(task_error_placeholder, weights_1) + biases_1,name="L1")

            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, NUM_OUTPUTS]),name="w2")
            biases_2 = tf.Variable(tf.zeros([NUM_OUTPUTS]),name="b2")
            # layer_2_outputs = tf.nn.softplus(tf.matmul(layer_1_outputs, weights_2) + biases_2)
            self.ff_NN_train = tf.add(tf.matmul(layer_1_outputs, weights_2),biases_2,name="output")

            # weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, NUM_OUTPUTS]))
            # biases_3 = tf.Variable(tf.zeros([1]))
            # logits = tf.matmul(layer_2_outputs, weights_3)+biases_3

            error_function = tf.reduce_mean(tf.square(tf.subtract(self.ff_NN_train, task_dyn_placeholder)))

            train_step = tf.train.AdamOptimizer(0.001).minimize(error_function)
            self.sess.run(tf.global_variables_initializer())

            task_dyn= task_dyn.reshape(task_dyn.shape[0],NUM_OUTPUTS)
            task_errors= task_errors.reshape(task_errors.shape[0],NUM_INPUTS)

            feed_dict={task_error_placeholder: task_errors,
                        task_dyn_placeholder: task_dyn} 

            for i in range(1000):
                _, loss = self.sess.run([train_step, error_function],feed_dict)

            print "Network trained"

    def set_bookkeeping_files(self):

        self.reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/rewards.txt'
        self.disc_reward_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/discounted_rewards.txt'

        self.actions_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions.txt'
        self.action_dist_mean_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/action_dist_mean.txt'
        self.exploration_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/exploration.txt'

        self.states_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/task_errors.txt'
        self.task_measure_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/task_measure.txt'

        self.neural_network_param_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights.txt'

        self.baseline_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/baseline.txt'
        self.advantageageages_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/advantages.txt'
        self.unnorm_advantageages_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/unnorm_advantages.txt'

        self.loss_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/losses.txt'
        self.log_likelihood_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/log_likelihood.txt'

        self.gradients_file_name = '/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients.txt'

        self.reset_files([self.neural_network_param_file_name, self.advantageageages_file_name, self.unnorm_advantageages_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.states_file_name, self.disc_reward_file_name, self.action_dist_mean_file_name,
                             self.task_measure_file_name, self.exploration_file_name, self.loss_file_name, self.log_likelihood_file_name, self.gradients_file_name])


    def store_weights(self):

        var = np.array([])
        for trainable_variable in tf.trainable_variables():
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

    def store_NN_output(self):
        self.save_matrix_data_to_file(self.action_dist_mean_file_name, self.NN_output)

    def store_task_measure(self):
        self.save_matrix_data_to_file(self.task_measure_file_name, self.task_measure)

    def store_task_errors(self):
        self.save_matrix_data_to_file(self.states_file_name, self.task_errors)

    def store_actions(self):
        self.save_matrix_data_to_file(self.actions_file_name, self.actions)

    def store_exploration(self):
        self.save_matrix_data_to_file(self.exploration_file_name, np.concatenate(self.exploration))


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

        self.NN_output.pop()
        self.all_actions_dist.append(np.asarray(self.NN_output))

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))

        self.task_measure.pop(0)
        self.episode_reward = self.calculate_return(self.task_measure)
        self.all_returns.append(self.episode_reward)

        self.episode_disc_reward = self.discount_rewards(self.episode_reward) 
        self.all_disc_returns.append(self.episode_disc_reward)

        self.task_errors.pop()
        self.all_task_errors.append(np.asarray(self.task_errors))

    def store_episode_data_to_file(self):
        self.store_NN_output()
        self.store_actions()
        self.store_task_measure()
        self.store_rewards()
        self.store_task_errors()
        self.store_exploration()
    
    def store_batch_data_to_file(self, batch_data):
        for key in batch_data:
            self.save_vector_data_to_file(key, batch_data[key])

        self.store_weights()

    def reset_episode(self):
        self.task_measure[:] = []
        self.task_errors[:] = []
        self.actions[:] = []
        self.NN_output[:] = []
        self.exploration[:]  = []

    def reset_eval_episode(self, curr_eval_return):
        self.prev_eval_mean_return = curr_eval_return
        self.store_NN_output()
        self.store_task_errors()
        self.reset_episode()

        self.eval_episode = False

    def reset_batch(self):

        self.eval_episode = True

        self.all_unnormalized_returns[:] = []
        self.all_returns[:]      = []
        self.all_disc_returns[:] = []
        self.all_actions[:]      = []
        self.all_actions_dist[:] = []
        self.all_task_errors[:]  = []
        self.all_task_measure[:] = []


    def get_NN_output_mean(self):
        print "Mean of actions: " + str(abs(np.mean(self.actions,axis=0)))
        return abs(np.mean(self.actions,axis=0))

    def discount_rewards(self, reward):

        discounted_r = np.zeros_like(reward)
        running_add = 0#np.zeros(reward.shape[1])
        for t in reversed(xrange(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_r[t] = running_add

        # discounted_r = self.normalize_data(discounted_r)
        return discounted_r


    def calculate_return(self, curr_reward):

        dist_square = np.square(np.asarray(curr_reward))
        dist_abs = np.abs(np.asarray(curr_reward))

        alpha = 1e-15#1e-17
        rollout_return = 0

        dist = np.sqrt(np.sum(dist_square,axis=1))
        # rollout_return = -10*dist-1.5*np.log(alpha+10*dist)
        rollout_return = -50*dist+25*np.exp(-15*dist)


        # for e in xrange(dist_square.shape[1]):
        #     if e==0:
        #         rollout_return += -10*np.log(alpha+10*dist_abs[:,e])#-10000*dist_abs[:,e]-10*np.log(alpha+15*dist_abs[:,e])#0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
        #     else:
        #         rollout_return += -10*np.log(alpha+10*dist_abs[:,e])#-10000*dist_abs[:,e]-10*np.log(alpha+15*dist_abs[:,e])#-0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)

            # if e==0:
            #     rollout_return += -100*dist_abs[:,e]-0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
            # else:
            #     rollout_return += -100*dist_abs[:,e]-0.5*np.log(dist_square[:,e]+alpha)#-2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)


        # rollout_return = -2*dist_square-0.4/18*np.log(dist_square+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)

        self.all_unnormalized_returns.append(rollout_return) 

        return rollout_return

    def normalize_data(self, data):
        print "Mean and variance of data is " + str(np.mean(data,axis=0)) + " " + str(np.std(data,axis=0))
        return (data-np.mean(data,axis=0))/(np.std(data,axis=0)+1e-8)


    def calculate_Fisher_Matrix(self, input_data, output):
        with self.g.as_default():
            var_list = tf.trainable_variables()
            pg = tf.gradients(self.loglik,var_list)
            # pg = tf.gradients(tf.log(self.ff_NN_train),var_list)
            pg = np.asarray(self.sess.run(pg, feed_dict={self.state_placeholder:input_data, self.action_placeholder : output, self.var : np.power(self.sigma,2)}))
            fisher = []
            eps = 1e-8
            for g in pg:
                fisher.append((np.asarray(1/np.square(g.flatten())+eps).reshape(g.shape))/self.batch_size)

            return fisher, pg



    def flattenVectors(self, vec):
    	vec = [idx.flatten().tolist() for idx in vec]
    	vec = np.asarray(list(itertools.chain.from_iterable(vec)))
    	return vec

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

    def calculate_advantages_with_VN(self, rewards, states):
        advantages = np.zeros_like(rewards)
        baseline = np.zeros_like(rewards)
        for i in xrange(len(states)):
            baseline[i] = self.VN.predict(states[i,:]).flatten()
            advantages[i] = rewards[i]-np.abs(baseline[i])

        norm_advantages = self.normalize_data(advantages)
        advantages = advantages.reshape(advantages.shape[0],1)
        norm_advantages = norm_advantages.reshape(norm_advantages.shape[0],1)
        return norm_advantages, advantages, baseline


    def policy_search(self, req):
        with self.g.as_default():
            

            #Do an evaluation episode (episode with no noise) every self.eval_episode
            if self.eval_episode:
                self.num_eval_episode += 1
                print "Evaluation episode number "+str(self.num_eval_episode)+" finished!" 
                curr_eval_return = self.get_undiscounted_reward().sum()
                print "Average return from evaluation is " + str(curr_eval_return)
                diff = curr_eval_return-self.prev_eval_mean_return
                if curr_eval_return>self.max_rew_before_convergence:
                    print "Policy converged in " +str(self.num_train_episode+self.num_eval_episode)+" episodes!"
                    self.train = False
                elif diff>0:
                    print "Policy improved by", diff
                    self.kl += 0.01
                else:
                    self.kl = 0.01 
                    print "Policy got worse by", diff


                self.sigma = self.get_NN_output_mean()
                self.sigma[self.sigma<0.2]=0.2
                self.reset_eval_episode(curr_eval_return)
                return PolicySearchResponse(not self.train)


            self.compute_episode_data()
            self.store_episode_data_to_file()

            self.num_train_episode +=1
            print "Training episode number "+str(self.num_train_episode)+" finished with a reward of " + str(self.episode_reward.sum()) 

            if self.num_train_episode % self.batch_size == 0 and self.train:

                print "Updating policy"
             
                rewards = np.concatenate(self.all_disc_returns)
                task_errors = np.concatenate(self.all_task_errors)
                actions = np.concatenate(self.all_actions)

                actions = actions.reshape(actions.shape[0],self.action_placeholder.shape[1])
                task_errors = task_errors.reshape(task_errors.shape[0],self.state_placeholder.shape[1])

                advantage, unnorm_advantages, baseline = self.calculate_advantages_with_VN(rewards, task_errors)

                curr_batch_mean_return = np.mean([ ret.sum() for ret in self.all_returns])

                print "mean of batch rewards is "+ str(curr_batch_mean_return)

                if self.train:
                    var_list = tf.trainable_variables()

                    fisher, pg = self.calculate_Fisher_Matrix(task_errors, actions)
                    learning_rate = float(self.calculate_learning_rate(fisher, pg))
                    temp = set(tf.global_variables())
                    
                    optimizer = tf.train.AdamOptimizer(learning_rate)#.minimize(self.loss)

                    loss_grads = optimizer.compute_gradients(self.loss, var_list)
                    masked_grad_and_vars = []
                    grad_temp = []
                    i = 0
                    for g,v in loss_grads:
                        grad = tf.multiply(tf.constant(fisher[i]), g)
                        masked_grad_and_vars.append((grad, v))
                        grad_temp.append(grad)
                        i+=1
                    # print tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, masked_grad_and_vars)],0)
                    train_op = optimizer.apply_gradients(masked_grad_and_vars)
                    self.sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
                    feed_dict={self.state_placeholder  : task_errors,
                               self.action_placeholder : actions,
                               self.advantage          : advantage,
                               self.var                : np.power(self.sigma,2)}

                    gradients = self.flattenVectors(self.sess.run(grad_temp,feed_dict))

                    for i in range(1):

                        _, error, ll= self.sess.run([train_op, self.loss, self.loglik], feed_dict)

                    batch_dic = {}

                    batch_dic[self.baseline_file_name] = baseline
                    batch_dic[self.advantageageages_file_name] = advantage.flatten()
                    batch_dic[self.unnorm_advantageages_file_name] = unnorm_advantages.flatten()
                    batch_dic[self.log_likelihood_file_name] = [ll]
                    batch_dic[self.loss_file_name] = [error]
                    batch_dic[self.gradients_file_name] = gradients

                    self.store_batch_data_to_file(batch_dic)
                    self.VN.train(task_errors, rewards, self.batch_size)
                    self.reset_batch() 

            self.reset_episode()

            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default():
            self.task_measure.append(req.task_measures)
            self.task_errors.append(req.task_measures)
            feed_dict = {self.state_placeholder: np.array([req.task_measures])}
            mean = self.sess.run(self.ff_NN_train, feed_dict)
            if self.train and not self.eval_episode:
                noise = np.random.normal(self.mean, self.sigma)
                task_dynamics = (mean+noise)#0.6*self.prev_action+0.4*(mean+noise)
                self.exploration.append(mean-task_dynamics)
                self.prev_action = task_dynamics

            else:
                task_dynamics = mean
            self.NN_output.append(mean.flatten())
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