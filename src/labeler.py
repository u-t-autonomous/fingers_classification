#!/usr/bin/env python
import struct
import math
import rospy
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from time import sleep
from scipy.signal import lfilter, get_window, butter
import pandas

def cleanize(channel):

	idx = len(channel)-1
	idx2 = len(channel)
	lag = channel[0:idx]
	reg = channel[1:idx2]
	cleaned_sig = (1/math.sqrt(2))*(reg-lag)
	return cleaned_sig

def set_interval(filename):
	run_data = pandas.read_csv(filename, sep=', ')
	run_data_matrix = pandas.DataFrame.as_matrix(run_data)
	begin = []
	end = []
	label = []
	for x in range(0,len(run_data_matrix)):
		if x%2 == 0:
			begin.append(run_data_matrix[x][0])
			label.append(run_data_matrix[x][1])
			end.append(run_data_matrix[x+1][0])
			label.append(run_data_matrix[x+1][1])
			assert((label[x] - label[x+1]) == 0)

	return (begin,end,label)		

def set_label_and_clean(label_file, data_file, new_file):
	begin, end, label = set_interval(label_file)
	channel_data = pandas.read_csv(data_file, sep=', ')
	channel_data_matrix = pandas.DataFrame.as_matrix(channel_data)
	jprime = 0
	for i in range(0,len(begin)):
		for j in range(jprime, len(channel_data_matrix)):
			data_time_val = channel_data_matrix[j][0]
			start_time = begin[i]
			stop_time = end[i]
			if data_time_val>=start_time and data_time_val<=stop_time:
				new_label_value = label[i*2]
				channel_data_matrix[j][6] = new_label_value
			elif data_time_val>stop_time:
				jprime = j
				break 
			else:
				continue 

	channel4 = channel_data_matrix[:,1]
	channel5 = channel_data_matrix[:,2]
	channel6 = channel_data_matrix[:,3]
	channel7 = channel_data_matrix[:,4]
	channel8 = channel_data_matrix[:,5]
	
	channel_data_matrix[0:len(cleanize(channel4)),1] = cleanize(channel4)
	channel_data_matrix[0:len(cleanize(channel5)),2] = cleanize(channel5)
	channel_data_matrix[0:len(cleanize(channel6)),3] = cleanize(channel6)
	channel_data_matrix[0:len(cleanize(channel7)),4] = cleanize(channel7)
	channel_data_matrix[0:len(cleanize(channel8)),5] = cleanize(channel8)

	channel_data_intermed = np.matrix(channel_data_matrix)
	channel_data_intermed = channel_data_matrix[0:len(cleanize(channel4)),:]


	header_list = ['time', 'chan4_thumb', 'chan5_index', 'chan6_middle', 'chan7_ring', 'chan8_pinky', 'label']
	new_channel_data = pandas.DataFrame(channel_data_intermed, columns=header_list)
	new_channel_data.to_csv(new_file, sep=',', index=False)




set_label_and_clean('./New_Data/11-11-16/I2/label.txt', './New_Data/11-11-16/I2/data_main.txt', './New_Data/11-11-16/I2/new_data_main.txt')

