#!/usr/bin/env python
import roslib

import rospy
import sys
import bisect
import itertools
from grasp_learning.srv import *
from utils import *
from value_function import *
from std_msgs.msg import Empty
from copy import deepcopy
from math import sqrt
from math import pow
import tensorflow as tf
import numpy as np
import time

class ValueNet(object):

    def __init__(self, num_inputs , num_outputs):

        self.vg = tf.Graph()
        self.session = tf.InteractiveSession(graph=self.vg)
        self.net = None

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
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

        input_output_data_file = rospy.get_param('~input_output_data', ' ')
        num_inputs = rospy.get_param('~num_inputs', ' ')
        num_outputs = rospy.get_param('~num_outputs', ' ')
        num_rewards = rospy.get_param('~num_rewards', ' ')
        load_example_model = rospy.get_param('~load_model', ' ')
        example_model_name = rospy.get_param('~model_name', ' ')
        hidden_layer_size  =  rospy.get_param('~hidden_layer_size', ' ') 
        relative_path =        rospy.get_param('~relative_path', ' ') 

        self.action_dim = num_outputs
        self.s = rospy.Service('query_NN', QueryNN, self.handle_query_NN_)
        
        self.policy_search_ = rospy.Service('policy_Search', PolicySearch, self.policy_search)

        self.sigma_old = self.sigma_new = 0
        self.num_train_episode = 0
        self.num_eval_episode = 0
        self.gamma = 0.9992
        self.batch_size = 10
        self.burn_in_trials = 20
        self.g = tf.Graph()
        self.train = True
        self.eval_episode = True
        self.max_kl = 0.01

        self.max_rew_before_convergence = 3000

        self.prev_action = np.zeros(num_outputs)
        self.all_returns = []
        self.all_unnormalized_returns = []
        self.all_disc_returns = []
        self.all_sum_disc_returns = []

        self.best_rollouts = []

        self.exploration = []

        self.all_actions = []
        self.all_actions_dist = []
        self.actions = []
        
        self.task_errors = []
        self.all_task_errors = []

        self.task_measure = []
        self.all_task_measure = []

        self.NN_output = []
        self.all_action_dist_logstd = []
        self.action_dist_logstd_ = []
        self.prev_eval_mean_return = 0

        self.VN = ValueNet(num_inputs, num_rewards)

        self.sess = tf.InteractiveSession(graph=self.g)

        self.state_placeholder = tf.placeholder(tf.float32, [None, num_inputs])
        self.action_placeholder = tf.placeholder(tf.float32, [None, num_outputs])
        self.oldaction_dist = tf.placeholder(tf.float32, [None, num_outputs])
        self.advantage = tf.placeholder(tf.float32,[None,num_rewards])
        self.std_old = tf.placeholder(tf.float32,[None, num_outputs])
        self.std_new = tf.placeholder(tf.float32,[None, num_outputs])
        self.std = tf.placeholder(tf.float32,[num_outputs])
        self.curr_std = np.zeros(num_outputs)
        input_data, output_data = self.parse_input_output_data(input_output_data_file)



        self.set_bookkeeping_files(relative_path)

        action_dist_logstd_param = tf.Variable((np.asarray([np.ones(num_outputs)])).astype(np.float32), name="policy_logstd")
        self.construct_ff_NN(self.state_placeholder, self.action_placeholder , input_data, output_data, num_inputs, num_outputs)

        self.action_dist_logstd = tf.tile(self.std*action_dist_logstd_param, tf.stack((tf.shape(self.ff_NN_train)[0], 1)))
        log_p_n = gauss_log_prob(self.ff_NN_train, self.action_dist_logstd, self.action_placeholder)
        log_oldp_n = gauss_log_prob(self.oldaction_dist, self.std_old, self.action_placeholder)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)
        # importance sampling of surrogate loss (L in paper)
        surr = -tf.reduce_mean(ratio * self.advantage)
        var_list = tf.trainable_variables()

        eps = 1e-8
        batch_size_float = tf.cast(self.batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = gauss_KL(self.oldaction_dist, self.std_old, self.ff_NN_train, self.action_dist_logstd) / batch_size_float
        ent = gauss_ent(self.ff_NN_train, self.action_dist_logstd) / batch_size_float

        self.losses = [surr, kl, ent]
        # policy gradient
        self.pg = flatgrad(surr, var_list)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = gauss_selfKL_firstfixed(self.ff_NN_train, self.action_dist_logstd) / batch_size_float
        # gradient of KL w/ itself
        grads = tf.gradients(kl_firstfixed, var_list)

        # what vector we're multiplying by
        self.flat_tangent = tf.placeholder(tf.float32, [None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        # gradient of KL w/ itself * tangent
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        self.fvp = flatgrad(gvp, var_list)
        # the actual parameter values
        self.gf = GetFlat(self.sess, var_list)
        # call this to set parameter values
        self.sff = SetFromFlat(self.sess, var_list)
        self.sess.run(tf.global_variables_initializer())

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
                        input_data.append(float(line[string]))
                        # input_data.append(float(line[string])+np.random.normal(0, 0.15))

                    else:
                        # if string == 1:
                        output_data.append(float(line[string]))
                        # output_data.append(float(line[string])+np.random.normal(0, 0.01))
                        # output_data.append(float(line[string]))

                input_.append(input_data)
                output_.append(output_data)
                input_data = []
                output_data = []

        return np.asarray(input_), np.asarray(output_)

    def construct_ff_NN(self, state_placeholder, action_placeholder, task_errors, task_dyn, NUM_INPUTS = 1, NUM_OUTPUTS = 1, HIDDEN_UNITS_L1 = 10, HIDDEN_UNITS_L2 = 5):
        with self.g.as_default():


            weight_init = tf.random_uniform_initializer(-0.05, 0.05)
            bias_init = tf.constant_initializer(0)
            initializer=weight_init

            self.w1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, HIDDEN_UNITS_L1]), name="input_layer")
            self.b1 = tf.Variable(tf.zeros([HIDDEN_UNITS_L1]))
            layer_1_outputs = tf.nn.tanh(tf.matmul(state_placeholder, self.w1) + self.b1)

            self.w2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L1, NUM_OUTPUTS]))
            self.b2 = tf.Variable(tf.zeros([NUM_OUTPUTS]))
            # layer_2_outputs = tf.nn.tanh(tf.matmul(layer_1_outputs, self.w2) + self.b2)
            self.ff_NN_train = tf.matmul(layer_1_outputs, self.w2)+self.b2

            # self.w3 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_L2, NUM_OUTPUTS]))
            # self.b3 = tf.Variable(tf.zeros([NUM_OUTPUTS]))
            # self.ff_NN_train = tf.matmul(layer_2_outputs, self.w3)+self.b3

            # error_function = tf.reduce_mean(tf.square(tf.subtract(self.ff_NN_train, action_placeholder)))

            # train_step = tf.train.AdamOptimizer(0.001).minimize(error_function)
            # self.sess.run(tf.global_variables_initializer())

            # task_dyn= task_dyn.reshape(task_dyn.shape[0],1)
            # task_errors= task_errors.reshape(task_errors.shape[0],1)
            # for i in range(500):
            #     _, loss = self.sess.run([train_step, error_function],
            #                feed_dict={state_placeholder: task_errors,
            #                           action_placeholder: task_dyn})

            print "Network Initialized"
            # print loss

    def set_bookkeeping_files(self,relative_path):

        self.reward_file_name = relative_path+'rewards.txt'
        self.disc_reward_file_name = relative_path+'discounted_rewards.txt'

        self.actions_file_name = relative_path+'actions.txt'
        self.action_dist_mean_file_name = relative_path+'action_dist_mean.txt'
        self.exploration_file_name = relative_path+'exploration.txt'

        self.states_file_name = relative_path+'task_errors.txt'
        self.task_measure_file_name = relative_path+'task_measure.txt'

        self.neural_network_param_file_name = relative_path+'weights.txt'
        
        self.kl_divergence_file_name = relative_path+'kl_div.txt'
        self.surrogate_loss_file_name = relative_path+'surr_loss.txt'
        self.dist_entropy_file_name = relative_path+'dist_ent.txt'

        self.baseline_file_name = relative_path+'baseline.txt'
        self.advantageages_file_name = relative_path+'advantages.txt'
        self.unnorm_advantageages_file_name = relative_path+'unnorm_advantages.txt'
        
        self.reset_files([self.neural_network_param_file_name, self.advantageages_file_name, self.unnorm_advantageages_file_name, self.baseline_file_name,
                             self.reward_file_name, self.actions_file_name, self.states_file_name, self.disc_reward_file_name, self.action_dist_mean_file_name,
                             self.task_measure_file_name, self.exploration_file_name, self.kl_divergence_file_name, self.surrogate_loss_file_name, self.dist_entropy_file_name])

    def store_weights(self):

        var = np.array([])
        for trainable_variable in tf.trainable_variables():
            var = np.append(var, trainable_variable.eval(session=self.sess))

        f_handle = file(self.neural_network_param_file_name,'a')
        np.savetxt(f_handle, [var] , delimiter='\t')
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


    def store_rewards(self):
        f_handle = file(self.reward_file_name,'a')
        np.savetxt(f_handle, [self.episode_reward], delimiter='\t')
        f_handle.close()

        f_handle = file(self.disc_reward_file_name,'a')
        np.savetxt(f_handle, [self.episode_disc_reward], delimiter='\t')
        f_handle.close()

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
        alpha = 1e-15

        rollout_return = np.zeros_like(dist_square[:,0])

        for e in xrange(dist_square.shape[1]):
            if e==0:
                rollout_return += -2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)
            else:
                rollout_return += -2*dist_square[:,e]-0.4/18*np.log(dist_square[:,e]+alpha)#-1*np.sqrt(dist_square[:,e]+alpha)


        self.all_unnormalized_returns.append(rollout_return) 

        return rollout_return


    def normalize_data(self, data):
        print "Mean and variance of data is " + str(np.mean(data,axis=0)) + " " + str(np.std(data,axis=0))
        return (data-np.mean(data,axis=0))/(np.std(data,axis=0)+1e-8)


    def calculate_advantages_with_VN(self, rewards, states):
        advantages = np.zeros_like(rewards)
        baseline = np.zeros_like(rewards)
        for i in xrange(len(states)):
            baseline[i] = self.VN.predict(states[i,:]).flatten()
            advantages[i] = rewards[i]-baseline[i]

        norm_advantages = self.normalize_data(advantages)
        advantages = advantages.reshape(advantages.shape[0],1)
        norm_advantages = norm_advantages.reshape(norm_advantages.shape[0],1)
        return norm_advantages, advantages, baseline


    def get_undiscounted_reward(self):
        self.task_measure.pop(0)
        return self.calculate_return(self.task_measure)

    def compute_episode_data(self):
        self.exploration.pop()

        self.task_measure.pop(0)
        self.episode_reward = self.calculate_return(self.task_measure)
        self.all_returns.append(self.episode_reward)

        self.episode_disc_reward = self.discount_rewards(self.episode_reward)
        self.all_disc_returns.append(self.episode_disc_reward)

        self.NN_output.pop()
        self.all_actions_dist.append(np.asarray(self.NN_output))

        self.action_dist_logstd_.pop()
        self.all_action_dist_logstd.append(np.asarray(self.action_dist_logstd_))

        self.actions.pop()
        self.all_actions.append(np.asarray(self.actions))

        self.task_errors.pop()
        self.all_task_errors.append(np.asarray(self.task_errors))

        self.arrange_by_return(self.episode_disc_reward.sum())

    def arrange_by_return(self, episode_disc_reward):
        pos = bisect.bisect_left(sorted(self.all_sum_disc_returns), episode_disc_reward)

        self.best_rollouts.insert(pos, self.num_train_episode)
        self.all_sum_disc_returns.append(episode_disc_reward)


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
        self.task_errors[:]  = []
        self.actions[:]      = []
        self.NN_output[:]    = []
        self.action_dist_logstd_ = []
        self.exploration[:]  = []

    def reset_eval_episode(self, curr_eval_return):
        self.prev_eval_mean_return = curr_eval_return
        self.store_NN_output()
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
        self.all_action_dist_logstd = []

    def get_NN_output_mean(self):
        print "Mean of actions: ", abs(np.mean(self.actions,axis=0).flatten())[0]
        # return np.mean(abs(np.mean(self.actions,axis=0)))
        return abs(np.mean(self.actions,axis=0).reshape(-1))

    def data_from_best_episodes(self,data):
        return np.concatenate([data[i] for i in self.best_rollouts[::-1][0:self.batch_size]])

    def policy_search(self, req):
        with self.g.as_default():
            

            #Do an evaluation episode (episode with no noise) every self.eval_episode
            if self.eval_episode == True:
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
                else:
                    print "Policy got worse by", diff
                
                self.curr_std = 1*self.get_NN_output_mean()
                self.reset_eval_episode(curr_eval_return)

                return PolicySearchResponse(not self.train)

            self.compute_episode_data()
            self.store_episode_data_to_file()

            self.num_train_episode +=1
            print "Training episode number "+str(self.num_train_episode)+" finished with a disounted return of " + str(self.episode_disc_reward.sum()) 

            if self.num_train_episode >= self.burn_in_trials  and self.train:

                print "Updating policy"
                if self.train:

                    rewards = self.data_from_best_episodes(self.all_disc_returns)
                    task_errors = self.data_from_best_episodes(self.all_task_errors) 
                    actions = self.data_from_best_episodes(self.all_actions) 
                    action_dist = self.data_from_best_episodes(self.all_actions_dist) 
                    action_dist_logstd = self.data_from_best_episodes(self.all_action_dist_logstd)

                    advantage, unnorm_advantages, baseline =  self.calculate_advantages_with_VN(rewards, task_errors)


                    feed_dict = {self.state_placeholder      : task_errors,
                                 self.action_placeholder     : actions,
                                 self.advantage              : advantage,
                                 self.oldaction_dist         : action_dist,
                                 self.std_old                : action_dist_logstd,
                                 self.std                    : self.curr_std}

                    # parameters
                    thprev = self.gf()

                    # computes fisher vector product: F * [self.pg]
                    def fisher_vector_product(p):
                        feed_dict[self.flat_tangent] = p
                        return self.sess.run(self.fvp, feed_dict)# + p * self.args.cg_damping

                    g = self.sess.run(self.pg, feed_dict)

                    # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
                    # stepdir = A_inverse * g = x
                    stepdir = conjugate_gradient(fisher_vector_product, -g)

                    # let stepdir =  change in theta / direction that theta changes in
                    # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
                    # where the [Fisher Information Matrix] acts like a metric
                    # ([Fisher Information Matrix] * stepdir) is computed using the function,
                    # and then stepdir * [above] is computed manually.
                    shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

                    lm = np.sqrt(shs / self.max_kl)
                    # if self.args.max_kl > 0.001:
                    #     self.args.max_kl *= self.args.kl_anneal

                    fullstep = stepdir / lm
                    negative_g_dot_steppdir = -g.dot(stepdir)

                    def loss(th):
                        self.sff(th)
                        # surrogate loss: policy gradient loss
                        return self.sess.run(self.losses[0], feed_dict)

                    batch_dic = {}

                    surrogate_prev, kl_prev, entropy_prev = self.sess.run(self.losses,feed_dict)
                    # finds best parameter by starting with a big step and working backwards
                    success, theta = linesearch(loss, thprev, fullstep, negative_g_dot_steppdir/ lm)
                    print "success", success
                    # i guess we just take a fullstep no matter what
                    # theta = thprev + fullstep
                    self.sff(theta)

                    surrogate_after, kl_after, entropy_after = self.sess.run(self.losses,feed_dict)

                    batch_dic[self.baseline_file_name] = baseline
                    batch_dic[self.advantageages_file_name] = advantage.flatten()
                    batch_dic[self.unnorm_advantageages_file_name] = unnorm_advantages.flatten()
                    batch_dic[self.kl_divergence_file_name] = [kl_prev, kl_after]
                    batch_dic[self.surrogate_loss_file_name] = [surrogate_prev, surrogate_after]
                    batch_dic[self.dist_entropy_file_name] = [entropy_prev, entropy_after]
                    
                    self.VN.train(task_errors, rewards, self.batch_size)
                    self.store_batch_data_to_file(batch_dic)
                    # self.reset_batch() 
                    print "Finished updating"

            self.reset_episode()

            return PolicySearchResponse(not self.train)


    def handle_query_NN_(self,req):
        with self.g.as_default(): 
            self.task_measure.append(req.task_measures)
            self.task_errors.append(req.task_measures)
            # start_time = time.time()
            mean_values, action_dist_logstd = self.sess.run([self.ff_NN_train, self.action_dist_logstd], feed_dict={self.state_placeholder : np.asarray([req.task_measures]),
                                                                                                                    self.std                    : self.curr_std})
            # print("--- %s seconds ---" % (time.time() - start_time))
            if self.train == True and self.eval_episode == False:
                exp = np.exp(action_dist_logstd)*np.random.randn(*action_dist_logstd.shape)
                task_dynamics = mean_values + exp
                # Low-pass filter the action.  
                task_dynamics = np.multiply(0.6,self.prev_action)+np.multiply(0.4,task_dynamics[0])
                self.prev_action = task_dynamics
                self.exploration.append(exp)
            else:
                task_dynamics = mean_values.tolist()

            self.NN_output.append(mean_values.flatten())
            self.actions.append(task_dynamics)
            self.action_dist_logstd_.append(action_dist_logstd.flatten())
            return  task_dynamics

    def main(self):
        rospy.spin()

# Main function.
if __name__ == '__main__':
    try:
        rospy.init_node('TRPO_exp')
        policy = Policy()
        policy.main()
    except rospy.ROSInterruptException:
        pass