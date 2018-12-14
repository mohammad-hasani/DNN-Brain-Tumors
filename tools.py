import vtk
from vtk.util import numpy_support
import os
import numpy as np
#import nrrd
from matplotlib import pyplot, cm
import cv2

"""
def read_nrrd_to_images():
    data_folder_name = './AutomatedSegmentation/case'
    store_folder_name = './images/'
    for i in range(1, 11):
        path = data_folder_name + str(i) + '/' + 'grayscale.nrrd'
        img = nrrd.read(path)
        data = img[0]
        tmp_path = store_folder_name + 'case' + str(i)
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        for j in range(data.shape[0]):
            tmp_path_2 = tmp_path + '/type0/'
            if not os.path.isdir(tmp_path_2):
                os.makedirs(tmp_path_2)
            cv2.imwrite(tmp_path_2 + str(i) + str(j) + '.jpg', data[j, :, :])
        for j in range(data.shape[1]):
            tmp_path_2 = tmp_path + '/type1/'
            if not os.path.isdir(tmp_path_2):
                os.makedirs(tmp_path_2)
            cv2.imwrite(tmp_path_2 + str(i) + str(j) + '.jpg', data[:, j, :])
        for j in range(data.shape[2]):
            tmp_path_2 = tmp_path + '/type2/'
            if not os.path.isdir(tmp_path_2):
                os.makedirs(tmp_path_2)
            cv2.imwrite(tmp_path_2 + str(i) + str(j) + '.jpg', data[:, :, j])
"""

def get_labels(case):
    data = np.zeros(256)

    if case == 'case1':
        data[169:209] = 1
    elif case == 'case2':
        data[96:131] = 1
    elif case == 'case3':
        data[141:174] = 1
    elif case == 'case4':
        data[130:170] = 1
    elif case == 'case5':
        data[160:182] = 1
    elif case == 'case6':
        data[141:181] = 1
    elif case == 'case7':
        data[160:200] = 1
    elif case == 'case8':
        data[87:109] = 1
    elif case == 'case9':
        data[106:160] = 1
    elif case == 'case10':
        data[96:129] = 1

    return data


def get_images(case):
    path = './images/case'
    data = []
    if case == 'case1':
        tmp_path = path + '1/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '1' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case2':
        tmp_path = path + '2/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '2' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case3':
        tmp_path = path + '3/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '3' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case4':
        tmp_path = path + '4/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '4' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case5':
        tmp_path = path + '5/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '5' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case6':
        tmp_path = path + '6/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '6' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case7':
        tmp_path = path + '7/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '7' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case8':
        tmp_path = path + '8/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '8' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case9':
        tmp_path = path + '9/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '9' + str(i) + '.jpg', 0)
            data.append(tmp)
    if case == 'case10':
        tmp_path = path + '10/type1/'
        for i in range(256):
            tmp = cv2.imread(tmp_path + '10' + str(i) + '.jpg', 0)
            data.append(tmp)
    return data


def get_data():
    cases = ['case' + str(i) for i in range(1, 11)]
    X = list()
    Y = list()
    for i, v in enumerate(cases):
        images = get_images(v)
        labels = get_labels(v)
        for j in range(len(images)):
            X.append(images[j])
            Y.append(labels[j])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def data_flatten(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    return np.array(new_data)
