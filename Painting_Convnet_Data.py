#Sam Rothstein and Tymoteusz
#4/19/2019
"""
This file contains the helper functions for our convnet program. We have made
it a separate file to make the code neater and more compact
"""

from PIL import Image
import requests
from io import BytesIO
import numpy as np
import xlrd
import random

dim = 90

def hot_label_all(lst_of_features):
    label = [0,0,0,0,0,0,0,0,0,0]
    for feature in lst_of_features:
        if feature == "aeroplane":
            label[0] = 1
            return label
        if feature == "bird":
            label[1] = 1
            return label
        if feature == "boat":
            label[2] = 1
            return label
        if feature == "chair":
            label[3] = 1
            return label
        if feature == "cow":
            label[4] = 1
            return label
        if feature == "diningtable":
            label[5] = 1
            return label
        if feature == "dog":
            label[6] = 1
            return label
        if feature == "horse":
            label[7] = 1
            return label
        if feature == "sheep":
            label[8] = 1
            return label
        if feature == "train":
            label[9] = 1
            return label
    return label


def hot_label(lst_of_features):
    label = [0,0,0,0,0,0]
    for feature in lst_of_features:
        if feature == "bird":
            label[0] = 1
            return label
        if feature == "boat":
            label[1] = 1
            return label
        if feature == "chair":
            label[2] = 1
            return label
        if feature == "diningtable":
            label[3] = 1
            return label
        if feature == "dog":
            label[4] = 1
            return label
        if feature == "horse":
            label[5] = 1
            return label
    return label


def one_hot_label(lst_of_features):
    label = [0,0,0,0,0,0,0,0,0,1]
    for feature in lst_of_features:
        if feature == "boat":
            return [0,0,1,0,0,0,0,0,0,0]
    return label

#return a multiple hot list
def mult_hot_label(lst_of_features):
    label = [0,0,0,0,0,0,0,0,0,0]
    for feature in lst_of_features:
        if feature == "aeroplane":
            label[0] = 1
        if feature == "bird":
            label[1] = 1
        if feature == "boat":
            label[2] = 1
        if feature == "chair":
            label[3] = 1
        if feature == "cow":
            label[4] = 1
        if feature == "diningtable":
            label[5] = 1
        if feature == "dog":
            label[6] = 1
        if feature == "horse":
            label[7] = 1
        if feature == "sheep":
            label[8] = 1
        if feature == "train":
            label[9] = 1
    return label

def mult_hot_label_all(lst_of_features):
    label = [0,0,0,0,0,0]
    for feature in lst_of_features:

        if feature == "bird":
            label[0] = 1
        if feature == "boat":
            label[1] = 1
        if feature == "chair":
            label[2] = 1

        if feature == "diningtable":
            label[3] = 1
        if feature == "dog":
            label[4] = 1
        if feature == "horse":
            label[5] = 1

    return label


def get_rgb(image_URL):
    all_rgb = []
    response = requests.get(image_URL)
    image  = Image.open(BytesIO(response.content))
    image = image.resize((dim,dim), Image.ANTIALIAS)
    #return image now for other funciton
    for x in range(dim):
        col = []
        for y in range(dim):
            r,g,b = image.getpixel((x,y))
            r = float(float(r)/255)
            g = float(float(g)/255)
            b = float(float(b)/255)
            col.append(tuple([r,g,b]))
        all_rgb.append(col)
    return all_rgb

def get_im(image_URL):
    response = requests.get(image_URL)
    image  = Image.open(BytesIO(response.content))
    image = image.resize((dim,dim), Image.ANTIALIAS)
    return image
