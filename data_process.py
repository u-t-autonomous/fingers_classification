import glob
import pandas as pd
import matplotlib as plt
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
%pylab

def MAV(points):
    return sum(map(abs,points))/len(points)

# loading the data
path =r'/home/sahabi/Documents/emg' # use your path
allFiles = glob.glob(path + "/new_data*.txt")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_,ignore_index = True)

# data pre-processing
frame.sort(columns='time', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
frame.reset_index(inplace = True)
del frame['index']

ch4 = frame['chan4_thumb']
ch5 = frame['chan5_index']
ch6 = frame['chan6_middle']
ch7 = frame['chan7_ring']
ch8 = frame['chan8_pinky']
label = frame['label']

X = [ch4.tolist(), ch5.tolist(), ch6.tolist(), ch7.tolist(), ch8.tolist()]
X = np.array(X)
X = np.array(X).reshape((shape(X)[1],shape(X)[0]))
Y = label
current_label = 9
start_window = -1
start_end = []
for i,x in enumerate(frame.label):
    if current_label != x:
        current_label = x
        if start_window == -1:
            start_window = i
        else:
            start_end.append((start_window,i,frame.label[i-1]))
            start_window = i
data_points = []
for tup in start_end:
    data_points.append((frame.chan4_thumb[tup[0]:tup[1]].tolist(),frame.chan5_index[tup[0]:tup[1]].tolist(),frame.chan6_middle[tup[0]:tup[1]].tolist(),frame.chan7_ring[tup[0]:tup[1]].tolist(),frame.chan8_pinky[tup[0]:tup[1]].tolist(),tup[2]))

# feature extraction
data = []

for data_point in data_points:
    data.append((map(MAV,data_point[0:5]),data_point[5]))
    #(sum(map(abs,data_point[0]))/len(data_point[0]),sum(map(abs,data_point[1]))/len(data_point[1]),sum(map(abs,data_point[2]))/len(data_point[2]),sum(map(abs,data_point[3]))/len(data_point[3]),sum(map(abs,data_point[4]))/len(data_point[4]),data_point[5])
    
# ML step  
# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(X,Y)
# scores = cross_val_score(clf, X[:20000], Y[:20000], cv=8,n_jobs = 8)
# scores

# TS plot
axes1 = plt.subplot(321)
axes1.set_ylim([-600,600])
axes1.plot(ch4[0:60000])
axes1.set_title('Ch4: Thumb')

axes2 = plt.subplot(322)
axes2.plot(ch5[0:60000])
axes2.set_ylim([-400,400])
axes2.set_title('Ch5: Index')

axes3 = plt.subplot(323)
axes3.plot(ch6[0:60000])
axes3.set_ylim([-400,400])
axes3.set_title('Ch6: Middle')

axes4 = plt.subplot(324)
axes4.plot(ch7[0:60000])
axes4.set_ylim([-400,400])
axes4.set_title('Ch7: Ring')

axes5 = plt.subplot(325)
axes5.plot(ch8[0:60000])
axes5.set_ylim([-400,400])
axes5.set_title('Ch8: Pinky')

axes6 = plt.subplot(326)
axes6.plot(label[0:60000]*80)
axes6.set_ylim([-400,400])
axes6.set_title('Label')


plt.show()

# 3D plot
# ch4_1 = frame[(frame.label == 3)]['chan4_thumb']
# ch5_1 = frame[(frame.label == 3)]['chan5_index']
# ch6_1 = frame[(frame.label == 3)]['chan6_middle']
# ch4_2 = frame[(frame.label == 2)]['chan4_thumb']
# ch5_2 = frame[(frame.label == 2)]['chan5_index']
# ch6_2 = frame[(frame.label == 2)]['chan6_middle']
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ch4_1, ch5_1, ch6_1, c='r', marker='o')
# ax.scatter(ch4_2, ch5_2, ch6_2, c='b', marker='^')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()