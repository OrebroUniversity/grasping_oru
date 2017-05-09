#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

d1=np.arange(-2,2,0.01)
d2=np.arange(-2,2,0.01)
dist= np.asarray([d1,d2])
dist = np.sqrt(np.sum(np.square(dist),axis=0))

alpha = 1e-15

# weighted_dist = -10*dist
# # plt.plot(d1,weighted_dist,label='w_log')
# for i in np.arange(1,7,1):
# 	print i

# 	weighted_log  = -i*np.log(alpha+dist)
# 	lab = str(i)
# 	# plt.plot(d1,weighted_log,label=lab)
# 	error=weighted_dist+weighted_log
# 	plt.plot(d1,error,label=lab)
weighted_exp = 50*np.exp(-15*dist)
weighted_dist = -20*dist
error=weighted_dist+weighted_exp

plt.plot(d1,error,label="weighted_exp")
# error[i-1]=weighted_dist+weighted_log[i-1]

# weighted_log[i-1]  = -0.6*np.log(alpha+10*dist)
# weighted_dist = -10*dist
# weighted_log  = -0.6*np.log(alpha+10*dist)
# error=weighted_dist+weighted_log
# print np.amax(weighted_log)
# markers_on = [np.argmax(weighted_log>0)-1]
# print d1[np.argmax(error>0)-1]

# plt.plot(d1,error,'-gD',markevery=markers_on)
# plt.plot(d1,np.zeros_like(d1))
# plt.plot(d1,test)
# plt.plot(d1,weighted_d1_sqrt,label='w_d1^2')
# plt.plot(d1,weighted_log,label='w_log')
# plt.plot(d1,weighted_dist,label='w_log')
# plt.plot(d1,error,label='w_log')

#plt.plot(d1,weighted_sqrt, label='w_sqrt')
# plt.legend(loc='upper left')
plt.axhline(y=0,color='black')

plt.legend(loc='lower center')

plt.show()
