import os
import cv2
import numpy as np
import math
import xlsxwriter
from collections import Counter
import math

data_testing = "C:\Program Files (x86)\AKSARA\database\\Data_Testing_Matriks.txt"
data_training = "C:\Program Files (x86)\AKSARA\database\\Data_Training_Matriks.txt"
output_file = "C:\Program Files (x86)\AKSARA\database\\aksara.txt"


def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    for index, example in enumerate(data):
        distance = distance_fn(example[:-1], query)
        neighbor_distances_and_indices.append((distance, index))

    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]
    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def string_to_array(data):
    row_string = data.split("|")
    temp_array = [row.split(" ") for row in row_string]
    return temp_array

with open(data_training, "r") as f:
    a = []
    for i in f:
        filenameTraining, patternTraining, string_array_Training = i[:-1].split("\t")
        nama = filenameTraining
        label = patternTraining
        training = np.array(string_to_array(string_array_Training), dtype=np.uint8)
        
        a.append((nama,label,training))

with open(data_testing, "r") as f:
    with open(output_file, "a") as output: 
        for i in f:
            filenameTesting, patternTesting, string_array_Testing = i[:-1].split("\t")
            testing= np.array(string_to_array(string_array_Testing), dtype=np.uint8)
            selectedPatternTesting = patternTesting
            print("Testing : ", selectedPatternTesting)

            max_value = -math.inf
            for nama,label,training in a:
                hasil = cv2.matchTemplate(testing, training,cv2.TM_CCOEFF_NORMED)[0][0]
                if hasil > max_value:
                    max_value = hasil
                    selectedPatternTraining = label
            
            output.write("{selectedPatternTesting}\t{max_value}\t{selectedPatternTraining}\n".format(
                selectedPatternTesting=selectedPatternTesting, max_value=max_value, selectedPatternTraining=selectedPatternTraining))

            print("Korelasi : ", max_value)
            print("Hasil Klasifikasi : ", selectedPatternTraining, "\n\n")
    output.close()
