#!/usr/bin/env python

import rospy
import open_bci_v3 as bci
from openbci.msg import BCIuVolts
from sensor_msgs.msg import Imu
import threading
import rospy
from time import sleep
from std_msgs.msg import String
import psychopy



class Node():
	def __init__(self):
		# Initialize node
		rospy.init_node('publish_measurements', anonymous=False)

		# Get ros parameters
		#port = rospy.get_param("~port")
		#baud = rospy.get_param('~baud')
		#filter_data = rospy.get_param('~filter_data')
		#scaled_output = rospy.get_param('~scaled_output')
		#daisy = rospy.get_param('~daisy_module')
		#log = rospy.get_param('~log')
		#timeout = rospy.get_param('~timeout')
#		
		port = '/dev/ttyUSB0'
		baud = 115200
		filter_data = True
		scaled_output = True
		daisy = False
		log = True
		timeout = 'None'
		try:
			timeout = float(timeout)
		except ValueError:
			timeout = None

		# Initialize OpenBCI board
		self.board = bci.OpenBCIBoard(
	        port = port, 
	        baud = baud,
	        filter_data = filter_data,
	        scaled_output = scaled_output,
	        daisy = daisy,
	        log = log,
	        timeout = timeout)
		
		self.f = open('/home/siddarthkaki/new_ws/src/openbci/src/data_main.txt','w')
		 #thumb - 1, index - 2, middle - 3, ring - 4, pinky - 5
		self.f.write(str('time, chan4_thumb, chan5_index, chan6_middle, chan7_ring, chan8_pinky, label')+'\n')
		self.f2 = open('/home/siddarthkaki/new_ws/src/openbci/src/label.txt', 'w')
		self.f2.write(str('time, label')+'\n')
	

		self.run_time = 0
		self.run = 0
		# Set ros parameters
		rospy.set_param('eeg_channel_count', self.board.getNbEEGChannels())
		rospy.set_param('aux_channel_count', self.board.getNbAUXChannels())
		rospy.set_param('sampling_rate', self.board.getSampleRate())

		# Setup EEG data publisher and message
		self.pub_EEG = rospy.Publisher('eeg_channels', BCIuVolts, queue_size=1)
		self.msg_EEG = BCIuVolts()

		# Setup AUX data publisher and messsage
		self.pub_AUX = rospy.Publisher('eeg_aux', Imu, queue_size=1)
		self.msg_AUX = Imu()
		self.msg_AUX.header.seq = 0
		self.msg_AUX.header.stamp = None
		self.msg_AUX.header.frame_id = ""
		self.msg_AUX.orientation_covariance[0] = -1 # No orientation
		self.msg_AUX.angular_velocity_covariance[0] = -1 # No angular velocity
		self.msg_AUX.linear_acceleration_covariance = [0,0,0,0,0,0,0,0,0] # No accel covariance

	def start_streaming(self):
		boardThread = threading.Thread(
			target=self.board.start_streaming, 
			args=(self.handle_sample, -1))
		boardThread.daemon = True # will stop on exit
		boardThread.start()
	
	def run_callback(self, data):
	 
	    #thumb - 1, index - 2, middle - 3, ring - 4, pinky - 5
	    self.f2.write(str(data.stamp)+', '+str(int(data.data[0]))+'\n')


	def run_listener(self):

	    rospy.Subscriber("chatter", BCIuVolts, self.run_callback)
	
	def stop_streaming(self):
		self.board.stop()

	def handle_sample(self, sample):
		stamp = rospy.get_rostime()

		self.msg_EEG.stamp = stamp
		self.msg_EEG.data = sample.channel_data

		chan4 = sample.channel_data[3]

		chan5 = sample.channel_data[4]

		chan6 = sample.channel_data[5]

		chan7 = sample.channel_data[6]

		chan8 = sample.channel_data[7]
		
		self.f.write(str(stamp)+', '+str(chan4)+', '+str(chan5)+', '+str(chan6)+', '+str(chan7)+', '+str(chan8)+', '+str(0)+'\n')

		self.pub_EEG.publish(self.msg_EEG)
		

		self.msg_AUX.header.seq += 1
		self.msg_AUX.header.stamp = stamp
		self.msg_AUX.linear_acceleration.x = sample.aux_data[0]
		self.msg_AUX.linear_acceleration.y = sample.aux_data[1]
		self.msg_AUX.linear_acceleration.z = sample.aux_data[2]
		self.pub_AUX.publish(self.msg_AUX)

	def __del__(self):
		self.f.close()
		self.f2.close()


if __name__ == '__main__':
    try:
        node = Node()
        node.run_listener()
        node.start_streaming()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
