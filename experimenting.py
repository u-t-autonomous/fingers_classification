# Name: Mohammed
import operator
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from mpl_toolkits.mplot3d import Axes3D
import itertools
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn import decomposition
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def accuracy(y_pred, y_test):
    return 1 - np.linalg.norm(np.array(y_pred) - np.array(y_test),ord = 0)/float(len(y_pred))

def relabel(Y):
    for i, label_ in enumerate(Y):
        if label_ == 0:
            if i != 0:
                if Y[i-1] == 1:
                    Y[i] = 6
                if Y[i-1] == 2:
                    Y[i] = 7
                if Y[i-1] == 3:
                    Y[i] = 8
                if Y[i-1] == 4:
                    Y[i] = 9
                if Y[i-1] == 5:
                    Y[i] = 10
            else:
                Y[i] = f
    return Y

def load_data(path):
    allFiles = glob.glob(path + "/new_data*.txt")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_,ignore_index = True)
    return frame

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def TD(channel_data):
    features = [MAV(channel_data)] + MAV_segments(channel_data) + diff_MAV(channel_data) + [ZC(channel_data)] +\
    [SSC(channel_data)] + [WL(channel_data)]
    return features

def MAV(channel_data):
    return sum(map(abs,channel_data))/len(channel_data)

def segment_window(channel_data, n_segments = 5.):
    seg_length = int(ceil(len(channel_data)/n_segments))
    segmented_data = [channel_data[x:x+seg_length] for x in range(0,len(channel_data),seg_length)]
    return segmented_data

def MAV_segments(channel_data):
    segmented_data = segment_window(channel_data)
    return map(MAV,segmented_data)

def diff_MAV(channel_data):
    segmented_data = segment_window(channel_data)
    prev_segments = segmented_data[:-1]
    next_segments = segmented_data[1:]
    prev_segments_MAV = map(MAV,prev_segments)
    next_segments_MAV = map(MAV,next_segments)
    return map(operator.sub, next_segments_MAV, prev_segments_MAV)

def ZC(channel_data, threshold = 10):
    prev_sample = channel_data[:-1]
    next_sample = channel_data[1:]
    left_side = map(abs,map(operator.sub, next_sample, prev_sample))
    right_side = map(abs,map(operator.add, next_sample, prev_sample))
    res = map(operator.sub, left_side, right_side)
    res = [1 if (x >= 0 and left > threshold) else 0 for x,left in zip(res,left_side)]
    return sum(res)/float(len(prev_sample)+1)

def SSC(channel_data, threshold = 10):
    the_sample = channel_data[1:-1]
    prev_sample = channel_data[:-2]
    next_sample = channel_data[2:]
    res = [1 if ((x > max(x_prev,x_next) or x < min(x_prev,x_next)) and max(abs(x_next - x),abs(x - x_prev)) > threshold)\
           else 0 for x,x_prev,x_next in zip(the_sample,prev_sample,next_sample)]
    return sum(res)/float(len(prev_sample)+2)

def WL(channel_data):
    prev_sample = channel_data[:-1]
    next_sample = channel_data[1:]
    diff = map(abs,map(operator.sub, next_sample, prev_sample))
    return sum(diff)/float(len(prev_sample)+1)
    
def fft_features(data,domain,req_freq,width):
    new_data = []
    for freq in req_freq:
        data_agg = []
        for i in range(0,len(data)):
            if domain[i] >= (freq - width/2.) and  domain[i] <= (freq + width/2.):
                data_agg.append(data[i])
        new_data.append(sum(data_agg)/float(len(data_agg)))
    return np.abs(new_data).tolist()
        
def fft(channel_data, freq = 125):
    width = 2.
    z = np.fft.rfft(channel_data) # FFT
    y = np.fft.rfftfreq(len(channel_data), d = 1./freq) # Frequency data
    z = zscore(z)
    req_freq = np.arange(5,65,width)
    return fft_features(z,y,req_freq,width)

def make_features(channel_data):
    return TD(channel_data)+fft(channel_data)
    

# loading the data
path =r'data' # use your path
frame = load_data(path)

# data pre-processing
# frame = frame[frame['label'] != 0]
frame.sort(columns='time', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
frame.reset_index(inplace = True)
del frame['index']

label = frame['label']

X = np.array([frame['chan4_thumb'].tolist(), frame['chan5_index'].tolist(), frame['chan6_middle'].tolist(),\
 frame['chan7_ring'].tolist(), frame['chan8_pinky'].tolist()])

shape = np.shape(X)
X = np.array(X).reshape((shape[1],shape[0]))
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
            # print i
            # start_end.append((frame.label[i-1]))
            start_window = i

data_points = []

for tup in start_end:
    data_points.append([frame.chan4_thumb[tup[0]:tup[1]].tolist(),frame.chan5_index[tup[0]:tup[1]].tolist(),\
        frame.chan6_middle[tup[0]:tup[1]].tolist(),frame.chan7_ring[tup[0]:tup[1]].tolist(),frame.chan8_pinky[tup[0]:tup[1]].tolist(),tup[2]])

# feature extraction
data = []
Y = []
for i, data_point in enumerate(data_points):
    data.append(map(make_features,data_point[0:5]))
    Y.append(data_point[5])
    #(sum(map(abs,data_point[0]))/len(data_point[0]),sum(map(abs,data_point[1]))/len(data_point[1]),sum(map(abs,data_point[2]))/len(data_point[2]),sum(map(abs,data_point[3]))/len(data_point[3]),sum(map(abs,data_point[4]))/len(data_point[4]),data_point[5])

# relabeling data
Y = relabel(Y)
class_names = [1,2,3,4,5,6,7,8,9,10]

data = np.array(data)
shape = np.shape(data)
data = np.reshape(data,(shape[0],shape[1]*shape[2]))
# pca = decomposition.PCA(n_components=35)
# pca.fit(data)
# data = pca.transform(data)
# ML step
# classifier = RandomForestClassifier(n_estimators=200) # best is 200
# classifier = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree') # best is 1
# classifier = LogisticRegression(C=0.25) # best is 0.25
# classifier = DecisionTreeClassifier(max_depth = 3) # best is 3
# scores = cross_val_score(classifier, np.array(data), Y, cv=5)
# print scores

# # TS plot
# axes1 = plt.subplot(321)
# axes1.set_ylim([-600,600])
# axes1.plot(ch4[0:100000])
# axes1.set_title('Ch4: Thumb')

# axes2 = plt.subplot(322)
# axes2.plot(ch5[0:100000])
# axes2.set_ylim([-400,400])
# axes2.set_title('Ch5: Index')

# axes3 = plt.subplot(323)
# axes3.plot(ch6[0:100000])
# axes3.set_ylim([-400,400])
# axes3.set_title('Ch6: Middle')

# axes4 = plt.subplot(324)
# axes4.plot(ch7[0:100000])
# axes4.set_ylim([-400,400])
# axes4.set_title('Ch7: Ring')

# axes5 = plt.subplot(325)
# axes5.plot(ch8[0:100000])
# axes5.set_ylim([-400,400])
# axes5.set_title('Ch8: Pinky')

# axes6 = plt.subplot(326)
# axes6.plot(label[0:100000]*80)
# axes6.set_ylim([-400,400])
# axes6.set_title('Label')

# plt.show()

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

X = data
y = Y

acc_log = []
trees = [100,200,300,400,500,600,700]
neighbors = [1,2,3,4,5,6]
Cs = [.1,.25,.5,1,1.5,2,3,5,8,10,20]
depth = [3,3,3,3,3,3,3,3]
i = 5
for lambda_ in depth:
    i += 1
    # Split the data into a training set and a test set
#     classifier = KNeighborsClassifier(n_neighbors=lambda_, algorithm='ball_tree')
#     classifier = RandomForestClassifier(n_estimators=lambda_)
#     classifier = LogisticRegression(C=lambda_)
#     classifier = svm.SVC(C=lambda_,kernel='poly')
    classifier = DecisionTreeClassifier(max_depth = lambda_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
    # Run classifier
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    y_pred = [x if x < 6 else 0 for x in y_pred]
    y_test = [x if x < 6 else 0 for x in y_test]
    print accuracy(y_test,y_pred)
    acc_log.append(accuracy(y_test,y_pred))

plt.plot(depth,acc_log)
plt.show()

max_index, max_value = max(enumerate(acc_log), key=operator.itemgetter(1))
param = Cs[max_index]
classifier = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Run classifier
y_pred = classifier.fit(X_train, y_train).predict(X_test)
y_pred = [x if x < 6 else 0 for x in y_pred]
y_test = [x if x < 6 else 0 for x in y_test]
# Compute confusion matrix
class_names = [0,1,2,3,4,5]
cnf_matrix = confusion_matrix(y_test, y_pred, labels = class_names)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()