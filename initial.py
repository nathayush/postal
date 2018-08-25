# program works on single digits only
import necessary libraries
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import matplotlib.image as mpimg
import itertools
import imutils
import scipy
from keras.utils import plot_model

# since we are using a saved model, the initial code is commented out
model = load_model('model.h5')

#Loading the data
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Creating new arrays to store the augmented images after pre-processing it.
#Xfinal_train = []
#Yfinal_train = []


#print("Augmenting")
#for i in range(len(X_train)): #X_train[:10]
#For each 5 degrees in 10 degrees we create another training image.
#    for angle in range(0,10,5):
#        augmented = imutils.rotate_bound(X_train[i], angle)
#        augmented = np.array(augmented)
#        augmented = scipy.misc.imresize(augmented, (28,28))
#        Xfinal_train.append(augmented)
#        Yfinal_train.append(y_train[i])


#Xfinal_train = np.array(Xfinal_train)
#Yfinal_train = np.array(Yfinal_train)

# reshape to be [samples][pixels][width][height]
#Xfinal_train = Xfinal_train.reshape(Xfinal_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


#Normalizing
#Xfinal_train = Xfinal_train / 255.0
#X_test = X_test / 255.0


#Yfinal_train = np_utils.to_categorical(Yfinal_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]

# reshape image
img = mpimg.imread('test/noise.jpg')
img = img/ 255.0
img = img.reshape(1,28,28,1)


#def mymodel():
#            model = Sequential()

            # Layer 1
#            model.add(Conv2D(96, (5, 5), input_shape = (28,28,1), padding='valid',activation='relu'))
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#            model.add(Dropout(0.5))

            # Layer 2
#            model.add(Conv2D(256, (3, 3), input_shape = (28,28,1), padding='valid',activation='relu'))
#            model.add(MaxPooling2D(pool_size=(2, 2)))
#            model.add(Dropout(0.5))

            # Layer 3
#            model.add(ZeroPadding2D((1,1)))
#            model.add(Conv2D(384, (2, 2), padding='valid',activation='relu'))
#            model.add(Dropout(0.5))

            # Layer 4
#            model.add(ZeroPadding2D((1,1)))
#            model.add(Conv2D(384, (3, 3), padding='valid',activation='relu'))
#            model.add(Dropout(0.5))

            # Layer 5
#            model.add(ZeroPadding2D((1,1)))
#            model.add(Conv2D(256, (3, 3), padding='valid',activation='relu'))
#            model.add(MaxPooling2D(pool_size=(1, 1)))
#            model.add(Dropout(0.5))

            # Layer 6
#            model.add(Flatten())
#            model.add(Dense(512, activation='relu'))
#            model.add(Dropout(0.5))

            # Layer 7
#            model.add(Dense(256,activation='relu'))
#            model.add(activation('relu'))
#            model.add(Dropout(0.5))

            # Layer 8
#            model.add(Dense(10, activation='softmax'))


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(Xfinal_train, Yfinal_train, validation_data=(X_test,y_test), epochs=2, batch_size=32)

#model.save('yourmodel.h5')

#print(model.summary())

l = model.predict(img)[0]
m = max(l)
print("Predicted Value is: ", [i for i, j in enumerate(l) if j == m][0])
plot_model(model, to_file='model.png')
# #Plotting the loss function
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
