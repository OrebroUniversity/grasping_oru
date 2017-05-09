#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_weights():
	return np.loadtxt("/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/weights.txt")

def get_states():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/task_errors.txt')
	
def get_actions():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/actions.txt')
	
def get_rewards():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/rewards.txt')
	
def get_disc_rewards():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/discounted_rewards.txt')
	
def get_advantages():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/advantages.txt')
	
def get_unnorm_advantages():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/unnorm_advantages.txt')

def get_baseline():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/baseline.txt')

def get_mean_action():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/action_dist_mean.txt')

def get_exploration():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/exploration.txt')

def get_surrogate_loss():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/surr_loss.txt')

def get_kl_div():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/kl_div.txt')

def get_dist_ent():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/dist_ent.txt')

def get_gradients():
	return read_file('/home/jejje/grasping_ws/src/grasping_oru/grasp_learning/stored_data/tested_data/gradients.txt')


def read_file(filename):
	with open(filename,"r") as f:
		all_data=[x.split() for x in f.readlines()]
		a=np.array([map(float,x) for x in all_data[:]])
	return a
def get_1d_indicies(batch_size, rollout, states):
	start_row = batch_size*get_start_row(rollout,batch_size)
	lower = 0
	for row in xrange(int(start_row),rollout):
		lower += len(states[row])

	upper = lower + len(states[rollout])-1

	return np.asarray([lower,upper])

def get_episode_data():
	episode_data = {}
	episode_data["advantages"] = get_advantages()
	episode_data["unnorm_advantages"] = get_unnorm_advantages()

	episode_data["baseline"] = get_baseline()
	episode_data["rewards"] = get_rewards()
	episode_data["disc_rewards"] = get_disc_rewards()
	episode_data["states"] = get_states()
	episode_data["actions"] = get_actions()
	episode_data["mean_action"] = get_mean_action()
	episode_data["exploration"] = get_exploration()
	episode_data["weights"] = get_weights()
	episode_data["gradients"] = get_gradients()

	# episode_data["surr_loss"] = get_surrogate_loss()
	# episode_data["kl_div"] = get_kl_div()
	# episode_data["dist_ent"] = get_dist_ent()

	return episode_data

def get_start_row(rollout,batch_size):
	return int(np.floor(rollout/batch_size))

def plot_rollout(num_rollout,batch_size=5, num_figs=1):
	episode_data = get_episode_data()

	indicies = get_1d_indicies(batch_size, num_rollout, episode_data["rewards"])
	start_row = get_start_row(num_rollout, batch_size)
	plt.figure(dpi=200)
	if num_figs==1:
		plt.plot(episode_data["states"][num_rollout+1][:],label='states')
		plt.plot(episode_data["actions"][num_rollout+1][:],label='actions')
		plt.plot(episode_data["mean_action"][num_rollout+1][:],label='mean_actions')
		plt.plot(episode_data["exploration"][num_rollout][:],label='exploration')

		# plt.plot(episode_data["rewards"][num_rollout], label='rewards')
		# plt.plot(episode_data["disc_rewards"][num_rollout], label='disc rewards')
		# print episode_data["advantages"][start_row]
		plt.plot(episode_data["advantages"][start_row][indicies[0]:indicies[1]], label='advantages')
		# plt.plot(episode_data["baseline"][start_row][indicies[0]:indicies[1]], label='baseline')
		plt.axhline(y=0,color='black')

		plt.legend(loc='lower right')
		plt.show()

	else:
		plt.figure(1)
		plt.subplot(211)

		plt.plot(episode_data["states"][num_rollout+1][0::2],label='states')
		# plt.plot(episode_data["actions"][num_rollout+1][0::2],label='actions')
		plt.plot(episode_data["mean_action"][num_rollout+1][0::2],label='mean_actions')
		# plt.plot(episode_data["exploration"][num_rollout][0::2],label='exploration')
		# plt.plot([x/1.0 for x in episode_data["rewards"][num_rollout]], label='rewards')
		# plt.plot([x/1 for x in episode_data["disc_rewards"][num_rollout]], label='disc rewards')
		# print episode_data["advantages"][start_row]
		plt.plot([x/1.0 for x in episode_data["advantages"][start_row][indicies[0]:indicies[1]]], label='advantages')
		# plt.plot([x/1.0 for x in episode_data["unnorm_advantages"][start_row][indicies[0]:indicies[1]]], label='unnorm advantages')

		# plt.plot([x/1.0 for x in episode_data["baseline"][start_row][indicies[0]:indicies[1]]], label='baseline')
		# f1.canvas.set_window_title('Task 1') 
		plt.axhline(y=0,color='black')
		plt.subplot(212)

		plt.plot(episode_data["states"][num_rollout][1::2],label='states')
		# plt.plot(episode_data["actions"][num_rollout+1][1::2],label='actions')
		plt.plot(episode_data["mean_action"][num_rollout+1][1::2],label='mean_actions')
		# plt.plot(episode_data["exploration"][num_rollout][1::2],label='exploration')

		# plt.plot([x/1.0 for x in episode_data["rewards"][num_rollout]], label='rewards')
		# plt.plot([x/1 for x in episode_data["disc_rewards"][num_rollout]], label='disc rewards')
		# print episode_data["advantages"][start_row]
		plt.plot([x/1.0 for x in episode_data["advantages"][start_row][indicies[0]:indicies[1]]], label='advantages')
		# plt.plot([x/1.0 for x in episode_data["unnorm_advantages"][start_row][indicies[0]:indicies[1]]], label='unnorm advantages')

		# plt.plot([x/1.0 for x in episode_data["baseline"][start_row][indicies[0]:indicies[1]]], label='baseline')
		# f2.canvas.set_window_title('Task 2')

		plt.axhline(y=0,color='black')
		plt.legend(loc='lower center')

		plt.show()

def plot_weights_diff():

	episode_data = get_episode_data()
	weight_diff = []
	for i in xrange(len(episode_data["weights"])-1):
		weight_diff.append(np.abs(episode_data["weights"][i]-episode_data["weights"][i+1]))
		plt.plot(weight_diff[i])
		plt.waitforbuttonpress()
	plt.close()

def plot_states(lower, upper, batch_size):
	episode_data = get_episode_data()
	plt.figure(dpi=200)

	for i in np.arange((lower-1)*3,(upper-1)*3,batch_size):
		plt.figure(1)
		plt.subplot(211)

		plt.plot(episode_data["states"][i][1000::2],label='states' + str(i))
		plt.axhline(y=0,color='black')
		plt.legend(loc='lower center')

		plt.subplot(212)
		plt.plot(episode_data["states"][i][1001::2],label='states' + str(i))

		plt.axhline(y=0,color='black')
		plt.legend(loc='lower center')
	plt.grid()

	plt.show()

def plot_gradients(num_rollout):
	episode_data = get_episode_data()
	plt.figure(dpi=200)
	x = range(len(episode_data["gradients"][num_rollout]))
	plt.bar(x,episode_data["gradients"][num_rollout],label='gradients' + str(num_rollout))
	plt.legend(loc='upper right')

	plt.show()

if __name__ == '__main__':

	while True:
		print "Input what data do you want to plot?"
		print "1 for rollout data\n2 for states\n3 for weights\n4 for gradients\n5 to quit"
		plot = int(raw_input(""))
		if plot == 1:
			print "Input rollout number, batch size, and number of outputs"
			rollout, batch_size, figs = map(int, raw_input("").split(" "))
			print "Plotting rollout data"
			plot_rollout(rollout, batch_size, figs)
		elif plot == 2:
			print "Input lower level, upper level and batch size"
			lower, upper, batch_size = map(int, raw_input("").split(" "))
			print "plotting states"
			plot_states(lower, upper, batch_size)
		elif plot == 3:
			print "Plotting the difference in weights from rollout to rollout"
			plot_weights_diff()
		elif plot == 4:
			print "Input rollout number"
			rollout = int(raw_input(""))
			print "plotting gradients"
			plot_gradients(rollout)
		elif plot == 5:
			break;
		else:
			print "Invalid input data"


	# plot_states(int(sys.argv[1]), int(sys.argv[2]))
	# if len(sys.argv)== 4:
	# 	plot_rollout(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
	# else:
	# 	print str(len(sys.argv)) + " arguments provided, but 4 are needed."
	# 	print "Please add " + str(4-len(sys.argv)) + " more"