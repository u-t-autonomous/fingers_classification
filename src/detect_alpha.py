#!/usr/bin/env python

import rospy
from openbci.msg import BCIuVolts
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from time import sleep
from stft import STFT
from movavg import MovAvg
import matplotlib.pyplot as plt
import numpy as np

'''
class Ignore():
	def __init__(self, count):
		self.reset(count)

	def test(self):
		if self.count > 0:
			self.count -= 1
			return True
		else:
			return False

	def reset(self, count):
		self.count = count

class DetectAlpha():
	def __init__(self):

		# Initialize node
		rospy.init_node('detect_alpha', anonymous=True)

		# Get ros parameters
		fs = rospy.get_param("sampling_rate")
		channel_count = rospy.get_param("eeg_channel_count")

		# Initialize STFT
		self.stft = STFT(fs, 1.0, 0.25, channel_count)
		self.stft.remove_dc()
		self.stft.bandpass(5.0, 15.0)
		self.stft.window('hann')
		self.freq_bins = self.stft.freq_bins
		self.FFT = np.zeros((len(self.freq_bins), channel_count))

		# Choose channels
		self.channel_mask = np.full(channel_count, False, dtype = bool)
		self.channel_mask[5 -1] = True
		self.channel_mask[6 -1] = True
		self.channel_mask[7 -1] = True
		self.channel_mask[8 -1] = True
		self.channel_mask[9 -1] = True

		# Define bands
		self.G1_mask = np.logical_and(5 < self.freq_bins, self.freq_bins < 7.5)
		self.Al_mask = np.logical_and(8.5 < self.freq_bins, self.freq_bins < 11.5)
		self.G2_mask = np.logical_and(12.5 < self.freq_bins, self.freq_bins < 15)

		# Initialize filters
		self.movavg = MovAvg(4)
		self.ignore = Ignore(0)

		# Setup publishers
		self.pub_guard1 = rospy.Publisher('guard1', Float32, queue_size=1)
		self.pub_alpha = rospy.Publisher('alpha', Float32, queue_size=1)
		self.pub_guard2 = rospy.Publisher('guard2', Float32, queue_size=1)
		self.pub_eyes = rospy.Publisher('eyes_closed', Bool, queue_size=1)

		# Subscribe
		rospy.Subscriber("eeg_channels", BCIuVolts, self.newSample)

	def newSample(self, msg):
		newFFT = self.stft.ingestSample(msg.data)
		if newFFT is not None:
			self.FFT = newFFT

			# Mask and average data
			guard1 = np.mean(newFFT[self.G1_mask, :][:, self.channel_mask])
			alpha = np.mean(newFFT[self.Al_mask, :][:, self.channel_mask])
			guard2 = np.mean(newFFT[self.G2_mask, :][:, self.channel_mask])

			detected = self.movavg.step(alpha > (guard1 + guard2)*1.1) > 0.5
			if detected and not self.ignore.test():
				self.movavg.reset()
				self.ignore.reset(4)
			else:
				detected = False
				
			# Publish messages
			msg = Float32()
			msg.data = guard1
			self.pub_guard1.publish(msg)

			msg = Float32()
			msg.data = alpha
			self.pub_alpha.publish(msg)

			msg = Float32()
			msg.data = guard2
			self.pub_guard2.publish(msg)

			msg = Bool()
			msg.data = detected
			self.pub_eyes.publish(msg)

	def updatePlot(self, line):
		line.set_ydata(np.sum(self.FFT[:,self.channel_mask], axis = 1))
		line.figure.canvas.draw()

if __name__ == '__main__':
    try:

		node = DetectAlpha()

		fig, ax = plt.subplots()
		li, = ax.plot(node.freq_bins, np.linspace(0, 10, len(node.freq_bins)))
		ax.set_xlim([0, 40])
		timer = fig.canvas.new_timer(interval=100)
		timer.add_callback(node.updatePlot, li)
		timer.start()
		plt.show()

		rospy.spin()
    except rospy.ROSInterruptException:
        pass
'''

class Ignore():
	def __init__(self, count):
		self.reset(count)

	def test(self):
		if self.count > 0:
			self.count -= 1
			return True
		else:
			return False

	def reset(self, count):
		self.count = count

class DetectBeta():
	def __init__(self):

		# Initialize node
		rospy.init_node('detect_alpha', anonymous=True)

		# Get ros parameters
		sleep(5)
		fs = rospy.get_param("sampling_rate")
		#fs = 125
		print fs
		channel_count = rospy.get_param("eeg_channel_count")
		print channel_count

		self.list4 = [0 for x in range(0,201)]
		self.list5 = [0 for x in range(0,201)]
		self.list6 = [0 for x in range(0,201)]
		self.list7 = [0 for x in range(0,201)]
		self.list8 = [0 for x in range(0,201)]
		

		# Subscribe
		rospy.Subscriber("eeg_channels", BCIuVolts, self.newSample)

	def newSample(self, msg):
	
		channel_data = msg.data

		ydata_chan4 = channel_data[3]
		y_uV_chan4 = ((ydata_chan4*4.5)/((2**23)-1)/24)*(1e6)

		ydata_chan5 = channel_data[4]
		y_uV_chan5 = ((ydata_chan5*4.5)/((2**23)-1)/24)*(1e6)

		ydata_chan6 = channel_data[5]
		y_uV_chan6 = ((ydata_chan6*4.5)/((2**23)-1)/24)*(1e6)

		ydata_chan7 = channel_data[6]
		y_uV_chan7 = ((ydata_chan7*4.5)/((2**23)-1)/24)*(1e6)

		ydata_chan8 = channel_data[7]
		y_uV_chan8 = ((ydata_chan8*4.5)/((2**23)-1)/24)*(1e6)

		self.list4.append(y_uV_chan4)
		self.list5.append(y_uV_chan5)
		self.list6.append(y_uV_chan6)
		self.list7.append(y_uV_chan7)
		self.list8.append(y_uV_chan8)
		
	
	def update_Chan4(self, line4):
		self.list4 = self.list4[-202:-1]
		#print(len(self.list5))
		line4.set_ydata(np.array(self.list4))
		ax[0].relim()
		ax[0].autoscale_view(False,True,True)
		line4.figure.canvas.draw()
	

	def update_Chan5(self, line5):
		self.list5 = self.list5[-202:-1]
		#print(len(self.list5))
		line5.set_ydata(np.array(self.list5))
		ax[1].relim()
		ax[1].autoscale_view(False,True,True)
		line5.figure.canvas.draw()


	def update_Chan6(self, line6):
		self.list6 = self.list6[-202:-1]
		line6.set_ydata(np.array(self.list6))
		ax[2].relim()
		ax[2].autoscale_view(False,True,True)
		line6.figure.canvas.draw()


	def update_Chan7(self, line7):
		
		self.list7 = self.list7[-202:-1]
		line7.set_ydata(np.array(self.list7))
		ax[3].relim()
		ax[3].autoscale_view(False,True,True)
		line7.figure.canvas.draw()


	def update_Chan8(self, line8):

		self.list8 = self.list8[-202:-1]
		line8.set_ydata(np.array(self.list8))
		ax[4].relim()
		ax[4].autoscale_view(False,True,True)
		line8.figure.canvas.draw()
		


if __name__ == '__main__':
    try:

		node = DetectBeta()
		
		fig, ax = plt.subplots(5,1)

		li4, = ax[0].plot(np.linspace(0, 201, 201), np.linspace(0, 201, 201))
		ax[0].set_title("CHANNEL 4")
		
		li5, = ax[1].plot(np.linspace(0, 201, 201), np.linspace(0, 201, 201))
		ax[1].set_title("CHANNEL 5")
		
		li6, = ax[2].plot(np.linspace(0, 201, 201), np.linspace(0, 201, 201))
		ax[2].set_title("CHANNEL 6")
		
		li7, = ax[3].plot(np.linspace(0, 201, 201), np.linspace(0, 201, 201))
		ax[3].set_title("CHANNEL 7")
		
		li8, = ax[4].plot(np.linspace(0, 201, 201), np.linspace(0, 201, 201))
		ax[4].set_title("CHANNEL 8")
		
		#ax[0].set_xlim([0, 201])
		#ax[0].set_ylim([-5000,5000])
		#ax[1].set_xlim([0, 201])
		#ax[1].set_ylim([-5000,5000])
		ax[0].set_autoscale_on(True)
		ax[0].margins(0, .9)

		ax[1].set_autoscale_on(True)
		ax[1].margins(0, .9)

		ax[2].set_autoscale_on(True)
		ax[2].margins(0, .9)

		ax[3].set_autoscale_on(True)
		ax[3].margins(0, .9)

		ax[4].set_autoscale_on(True)
		ax[4].margins(0, .9)


		timer = fig.canvas.new_timer(interval=100)
		timer.add_callback(node.update_Chan4, li4)
		timer.add_callback(node.update_Chan5, li5)
		timer.add_callback(node.update_Chan6, li6)
		timer.add_callback(node.update_Chan7, li7)
		timer.add_callback(node.update_Chan8, li8)

		timer.start()
		plt.show()
		
		rospy.spin()
    except rospy.ROSInterruptException:
        pass
