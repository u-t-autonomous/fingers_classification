#!/usr/bin/env python
import struct
import rospy
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from time import sleep
from scipy.signal import lfilter, get_window, butter
import csv
#global gyro
#gyro = []

#with open("08-capture_top_top.bin", "rb") as f:
    #data = f.read()
    #for i in range(int(len(data)/1)):
        #gyro.append(struct.unpack('<b',data[1*i:1*(i+1)])[0])
    #print(gyro)

'''
my_data = pandas.read_csv('data_chan7.txt', sep=', ', nrows=)
y = pandas.DataFrame.as_matrix(my_data)

'''    

my_data = genfromtxt('cvlist_test.dat', delimiter=',')
#my_data = abs(my_data)
#print(my_data)
fs = 125

j = my_data.shape
a = np.mean(my_data, axis=0)
low_hi = np.array((45,55), dtype=float)


'''
if (1):
	my_data = my_data - np.ones((j[0], 1))*a
	b, a = butter(1, low_hi/(fs/2), btype = 'stop')
	my_data = lfilter(b,a,my_data, axis=0)
	b, a = butter(2,45.0/(fs/2))
	my_data = lfilter(b,a,my_data, axis=0)
	#print(my_data)
'''

y = my_data
y = np.asarray(y)
y = np.reshape(y,(my_data.shape[0],1))




#t = np.arange(0,16,.008)
#t = t[0:1997]
#t = np.reshape(t,(t.shape[0],1))
y_uV = ((y*4.5)/((2**23)-1)/24)*(1e6)


Fs = 125.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,len(y_uV)/Fs,Ts) # time vector
#t1 = (t[5576:5702])
#t2 = (t[5703:5828])
#print(t1)
#print(t2)
#y_uV1 = y_uV[5576:5702]
#y_uV2 = y_uV[5703:5828]
#t = t1
#sig = y_uV1
#sig2 = y_uV2
#n2 = len(sig2)
n = len(y_uV) # length of the signal
k = np.arange(n)
T = n/Fs
#k2 = np.arange(n2)
#T2 = n2/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range
#frq2 = k2/T2 # two sides frequency range
#frq2 = frq2[range(n2/2)]

Y = np.fft.rfft(y_uV, axis=0)/n # fft computing and normalization
Y = Y[range(n/2)]
#Y2 = np.fft.rfft(sig2, axis=0)/n
#Y2 = Y2[range(n2/2)]

#print(abs(Y))

fig, ax = plt.subplots(2, 1)
#ax[0].plot(t,sig)
#ax[0].set_ylim((0,.06))
ax[0].plot(t,y_uV)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('y_uV')
#plt.xlim((.5,1.5))
ax[1].set_ylim((0,.06))
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()

alist = [[frq,Y]]
