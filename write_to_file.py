from PIL import Image as PillowImage
import numpy as np
from PIL import ImageTk as imtk, Image as img
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox as msg
import imgutil as iutil

import globalvar
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2 as opencv
import numpy as np

CONST_WIDTH = 20
CONST_HEIGHT = 20

def openImage(filename):
    return opencv.imread(filename, opencv.IMREAD_UNCHANGED)

source_folder = "C:\Program Files (x86)\AKSARA\database\\Data_Training"
label_file = 'C:\Program Files (x86)\AKSARA\database\Label_Data_Training\\Data_Training_Label.anno'
output_file = 'C:\Program Files (x86)\AKSARA\database\\aksara.txt'

def rgb2grayscale(image):
    return opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)

def convertNPImage(image):
    np.set_printoptions(threshold=np.inf)
    return np.array(image)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def add_NParray(NParray, value):
    return np.append(NParray,value)

def resize_image(image):
    dim = (CONST_WIDTH, CONST_HEIGHT)
    resized = opencv.resize(image, dim, interpolation=opencv.INTER_AREA)
    return resized

def resize_show_image(image, const_width):
    try:
        height, width, channels = image.shape
    except:
        height, width = image.shape

    scale = float(const_width) / float(height)
    re_width = int(width * scale)
    re_height = const_width
    dim = (re_width, re_height)
    resized = opencv.resize(image, dim, interpolation=opencv.INTER_AREA)
    return resized

def getProjectionProfile(image):
    resized_image = resize_image(image)
    gray_image = rgb2grayscale(resized_image)
    np_image = convertNPImage(gray_image)
    horz_profile = getHorizontalProjectionProfile(np_image)
    vert_profile = getVerticalProjectionProfile(np_image)
    return horz_profile

def getHorizontalProjectionProfile(image):
    image[image == 0] = 1
    image[image == 255] = 0
    horizontal_projection = np.sum(image, axis=1)
    return horizontal_projection


def getVerticalProjectionProfile(image):
    image[image == 0] = 1
    image[image == 255] = 0
    vertical_projection = np.sum(image, axis=0)
    return vertical_projection


def horz_profile(args):
    pass


def vert_profile(args):
    pass


with open(label_file, "r") as file_input:
    with open(output_file, "a") as file_output:
        for file_label in file_input:

            file_name, attr1, attr2, attr3, attr4, label = file_label[:-1].split("\t")


            def getHorizontalProjectionProfile(image):
                image[image == 0] = 1
                image[image == 255] = 0
                horizontal_projection = np.sum(image, axis=1)
                return horizontal_projection
                horz_profile = iutil.getHorizontalProjectionProfile(np_image)
                max_horz_profile = max(horz_profile)
                horz_profile = horz_profile / max_horz_profile
                print(horz_profile)


            def getVerticalProjectionProfile(image):
                image[image == 0] = 1
                image[image == 255] = 0
                vertical_projection = np.sum(image, axis=0)
                return vertical_projection
                vert_profile = iutil.getVerticalProjectionProfile(np_image)
                max_vert_profile = max(vert_profile)
                vert_profile = vert_profile / max_vert_profile
                print(vert_profile)

            file_output.write("{filename}\t{label}\t {horz_profile} \t {vert_profile}\n".format(filename=file_name,label=label, horz_profile=horz_profile,vert_profile=vert_profile))

