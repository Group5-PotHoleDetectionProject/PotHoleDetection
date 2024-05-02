import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Cropping2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import time, cv2, glob
global inputShape,size
def pothole_model():
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size,size,1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        # model.add(Dropout(.2))
        # model.add(Activation('relu'))
        # model.add(Dense(1024))
        # model.add(Dropout(.5))
        model.add(Dense(512))
        model.add(Dropout(.1))
        model.add(Activation('relu'))
        # model.add(Dense(256))
        # model.add(Dropout(.5))
        # model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

size=100

# Pothole_Data_Training 
Pothole_train_images = glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/train/Pothole/*.jpg")
Pothole_train_images.extend(glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/train/Pothole/*.jpeg"))
Pothole_train_images.extend(glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/train/Pothole/*.png"))

tr_1 = [cv2.imread(img,0) for img in Pothole_train_images]
for i in range(0,len(tr_1)):
    tr_1[i] = cv2.resize(tr_1[i],(size,size))
var1 = np.asarray(tr_1)


#Non-Pothole_Data_Training 
non_pothole_train_images = glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/train/Plain/*.jpg")
tr_2 = [cv2.imread(img,0) for img in non_pothole_train_images]
for i in range(0,len(tr_2)):
    tr_2[i] = cv2.resize(tr_2[i],(size,size))
var2 = np.asarray(tr_2)


#Pothole_Data_Testing 
Pothole_test_images = glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/test/Pothole/*.jpg")
te_1 = [cv2.imread(img,0) for img in Pothole_test_images]
for i in range(0,len(te_1)):
    te_1[i] = cv2.resize(te_1[i],(size,size))
var3 = np.asarray(te_1)

#Non-Pothole_Data_Testing
non_pothole_test_images = glob.glob("C:/Users/trinadh/Dropbox/PC/Desktop/capstoneproject/My Dataset/test/Plain/*.jpg")
te_2 = [cv2.imread(img,0) for img in non_pothole_test_images]
for i in range(0,len(te_2)):
    te_2[i] = cv2.resize(te_2[i],(size,size))
var4 = np.asarray(te_2)




A_training = []
A_training.extend(var1)
A_training.extend(var2)
A_training = np.asarray(A_training)

A_testing = []
A_testing.extend(var3)
A_testing.extend(var4)
A_testing = np.asarray(A_testing)

B_training1 = np.ones([var1.shape[0]],dtype = int)
B_training2 = np.zeros([var2.shape[0]],dtype = int)
B_testing1 = np.ones([var3.shape[0]],dtype = int)
B_testing2 = np.zeros([var4.shape[0]],dtype = int)

print(B_training1[0])
print(B_training2[0])
print(B_testing1[0])
print(B_testing2[0])

B_training = []
B_training.extend(B_training1)
B_training.extend(B_training2)
B_training = np.asarray(B_training)

B_testing = []
B_testing.extend(B_testing1)
B_testing.extend(B_testing2)
B_testing = np.asarray(B_testing)

A_training,B_training = shuffle(A_training,B_training)
A_testing,B_testing = shuffle(A_testing,B_testing)


A_training = A_training.reshape(A_training.shape[0], size, size, 1)
A_testing = A_testing.reshape(A_testing.shape[0], size, size, 1)

B_training = to_categorical(B_training)
B_testing = to_categorical(B_testing)

print("train shape X", A_training.shape)
print("train shape y", B_training.shape)

inputShape = (size, size, 1)
model = pothole_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(A_training, B_training, epochs=500,validation_split=0.1)
metrics = model.evaluate(A_testing, B_testing)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

model.save('model_sample.h5')

model_json = model.to_json()
with open("sample.json", "w") as json_file:
    json_file.write(model_json)
        
#confusion matrix
x_pred=model.predict(A_testing)
x_pred_class=np.argmax(x_pred,axis=1)
x_true=np.argmax(B_testing,axis=1)
matrix=confusion_matrix(x_true,x_pred_class)
plt.figure(figsize=(4,3))
sns.heatmap(matrix,annot=True,cmap='Blues',fmt='d', cbar=False, square=True)
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.show()

model.save_weights("sample.weights.h5")
print("Model Was Created And Saved To Disk")
