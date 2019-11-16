#1... Bulding the CNN

from keras.models import Sequential    #to initialize nn
from keras.layers import Convolution2D #1. add covolution layers
from keras.layers import MaxPooling2D  #2. the pooling step
from keras.layers import Flatten       #3. convert feater maps ino input for fully conected layer
from keras.layers import Dense         #4. add the fully connected layer
 
#initializing the cnn
classifier =Sequential()

#1.add convolution:
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3) , activation='relu'))

#2.pooling step:
classifier.add(MaxPooling2D(pool_size=(2,2)))


##ADDING NEW CONV LAYER TO IMPROVE PERFORMANCE

#classifier.add(Convolution2D(32,3,3 , activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))

#3.Flatten step:
classifier.add(Flatten())    

#4.fully connected layer:
classifier.add(Dense(output_dim=128,activation='relu'))  #hidden layer
classifier.add(Dense(output_dim=1,activation='sigmoid')) #output layer

#compiled cnn
classifier.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

#2...fitting the CNN t the images 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',    #directory of trsining set
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',       #directory of test set
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)