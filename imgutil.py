import cv2 as opencv
import numpy as np

CONST_WIDTH = 20
CONST_HEIGHT = 20

def openImage(filename):
    return opencv.imread(filename, opencv.IMREAD_UNCHANGED)

def getProjectionProfile(image):
    resized_image = resize_image(image)
    gray_image = rgb2grayscale(resized_image)
    np_image = convertNPImage(gray_image)
    horz_profile = getHorizontalProjectionProfile(np_image)
    vert_profile = getVerticalProjectionProfile(np_image)
    return horz_profile

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
