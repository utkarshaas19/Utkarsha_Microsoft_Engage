# Developer : Utkarsha Avirat Sutar
# Date : 22-May-2022
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
# Training the Model for Emotion Detection - UTKARSHA
utktrain_data_dir='data/train/'
utkvalidation_data_dir='data/test/'
utktrain_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,horizontal_flip=True,fill_mode='nearest')
utkvalidation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = utktrain_datagen.flow_from_directory(utktrain_data_dir,color_mode='grayscale',target_size=(48, 48),batch_size=32,class_mode='categorical',shuffle=True)
validation_generator = utkvalidation_datagen.flow_from_directory(utkvalidation_data_dir,color_mode='grayscale',target_size=(48, 48),batch_size=32,class_mode='categorical',shuffle=True)
# Mentioning all the Emotions that are to be Detected by the System - UTKARSHA
emotionslist=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
img, label = train_generator.__next__()
# Designing the Model - UTKARSHA
utkmodel = Sequential()
utkmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
utkmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
utkmodel.add(MaxPooling2D(pool_size=(2, 2)))
utkmodel.add(Dropout(0.1))
utkmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
utkmodel.add(MaxPooling2D(pool_size=(2, 2)))
utkmodel.add(Dropout(0.1))
utkmodel.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
utkmodel.add(MaxPooling2D(pool_size=(2, 2)))
utkmodel.add(Dropout(0.1))
utkmodel.add(Flatten())
utkmodel.add(Dense(512, activation='relu'))
utkmodel.add(Dropout(0.2))
utkmodel.add(Dense(7, activation='softmax'))
utkmodel.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Specifying Mudel Summary - UTKARSHA
print(utkmodel.summary())
# Specify the Path - UTKARSHA
train_path = "data/train/"
test_path = "data/test"
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)
print(num_train_imgs)
print(num_test_imgs)
epochs=30
history=utkmodel.fit(train_generator,steps_per_epoch=num_train_imgs//32,epochs=epochs,validation_data=validation_generator,validation_steps=num_test_imgs//32)
utkmodel.save('model_file.h5')