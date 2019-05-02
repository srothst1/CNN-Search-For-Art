#Sam Rothstein and Tymoteusz
#4/19/2019

#Imports
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import xlrd
from Painting_Convnet_Data import *

import random

#Process Image Data From Excel File

dim = 90

#Location of File
location = ('updated_data_set.xlsx') #new and updated file


# To open Workbook
wb = xlrd.open_workbook(location)
sheet = wb.sheet_by_index(0)
number_of_rows = sheet.nrows
number_of_cols = sheet.ncols

image_URL = [] #contains the url of the image
web_page_URL = [] #contains the url of the website that has the image
subset = [] #training, testing, validation
labels = [] #one hot label, representing a feature (ex. dog)
one_label = [] #only identifies a certain feature
mult_labels = [] #multiple hot lables, representing multiple features (ex. dog, chair)
standard_labels = [] #strings with the words that are represented in labels and mult_labels
rgb_images = [] #NxN arrays that contain rgb values
images = [] #image types that contain the acutal images


#enter = int(input("Please enter the number of images you want to process:  ")) #used for testing
enter = 5585 #length of the data set

print("processing data....")

#sort through excel file
for i in range(sheet.ncols):
    for j in range(enter + 1): #change to sheet.nrows
        #print(j) #shows status
        if i == 0:
            image_URL.append(sheet.cell_value(j,i))
            if j > 0:
                im_rgb = get_rgb(sheet.cell_value(j,i))
                rgb_images.append(im_rgb)
                im_pr = get_im(sheet.cell_value(j,i))
                images.append(im_pr)
        if i == 1:
            web_page_URL.append(sheet.cell_value(j,i))
        if i == 2:
            temp = sheet.cell_value(j,i)
            subset.append(temp[1:-1])
        if i == 3:
            temp = sheet.cell_value(j,i)
            lst = list(temp[2:-1].split(" "))
            standard_labels.append(lst)
            labels.append(hot_label(lst))
            mult_labels.append(mult_hot_label(lst))
            one_label.append(one_hot_label(lst))


image_URL = image_URL[1:]
web_page_URL = web_page_URL[1:]
subset = subset[1:]
labels = np.array(labels[1:])
one_label = np.array(one_label[1:])
mult_labels = np.array(labels[1:])
standard_labels = standard_labels[1:]


#testing data sets
print("- - -")
t = input("Please enter 0 if you would like to test a few data points: ")
if t == 0:
    x = ""
    while x != "exit":
        i = int(input("Please enter the index of the image you would like to see: "))
        images[i].show()
        print(rgb_images[i])
        print("one hot label: ")
        print(labels[i])
        print("multiple hot label: ")
        print(mult_labels[i])
        print("keyword tags: ")
        print(standard_labels[i])
        x = raw_input("Enter exit if you are done testing:  ")
print("- - -")
print("finished processing and testing data from Excel")


#Set up Testing Data, Training Data, and Validation Data
print("setting up training, testing, and validation data...")
train_X = []
train_Y = []
test_X = []
test_Y = []
validation_X = []
validation_Y = []


#standard sorting
for i in range(len(subset)):
    if subset[i] == "train":
        train_X.append(rgb_images[i])
        train_Y.append(labels[i])

    if subset[i] == "test":
        test_X.append(rgb_images[i])
        test_Y.append(labels[i])

    if subset[i] == "validation":
        train_X.append(rgb_images[i])
        train_Y.append(labels[i])


#convert to np.arrays
train_X = np.array(train_X)
test_X = np.array(test_X)
validation_X = np.array(validation_X)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)
validation_Y = np.array(validation_Y)


print("finished setting up training, testing, and validation data.")
print("creating convolutional neural netowrk... ")

#Create Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

#Construct the Model
#The First Layer Must Provide the Input Shape

neural_net = Sequential()
neural_net.add(Conv2D(64,(30,30),activation="relu",input_shape=(dim,dim,3)))
neural_net.add(Conv2D(64,(21,21),activation="relu"))
neural_net.add(Conv2D(64,(21,21),activation="relu"))
neural_net.add(MaxPooling2D(pool_size=2))
neural_net.add(Conv2D(32,(10,10),activation="relu")) #5,5
neural_net.add(Flatten())
neural_net.add(Dense(3000, activation='relu'))
neural_net.add(Dense(2500, activation='relu'))
neural_net.add(Dense(1000, activation='relu'))
neural_net.add(Dense(900, activation='relu'))
neural_net.add(Dense(98, activation='relu'))
neural_net.add(Dense(6, activation='softmax')) #was 10
neural_net.summary()


#Run Convolutional Neural Network
# Compile the model
neural_net.compile(optimizer="SGD", loss="categorical_crossentropy",
                   metrics=['accuracy'])
# Train the model
history = neural_net.fit(train_X, train_Y, verbose=1,
                         validation_data=(test_X, test_Y),
                         epochs=10)

loss, accuracy = neural_net.evaluate(test_X, test_Y, verbose=0)
print("accuracy: {}%".format(accuracy*100))

#Examine Which Test Data the Network is Failing to Predict
import matplotlib.pyplot as plt
from numpy import argmax
from numpy.random import randint

outputs = neural_net.predict(test_X)
answers = [argmax(output) for output in outputs]
targets = [argmax(target) for target in test_Y]

#uncomment for feedback
'''
for i in range(len(answers)):
    if answers[i] != targets[i]:
        print("Network predicted", answers[i], "Target is", targets[i])
'''
