#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import time
import pdb
pdb.set_trace()


class FACEINFO(ctypes.Structure):
    _fields_ = [('left', ctypes.c_int),('top', ctypes.c_int), 
                ('right', ctypes.c_int),('bottom', ctypes.c_int),
                ('FeatureSize', ctypes.c_int), ('feature', ctypes.c_float*256)]


def cal_sim(feat1, feat2):
    return np.dot(feat1,feat2)

def test(lib_path):
    lib = ctypes.cdll.LoadLibrary(lib_path)
    print(type(lib))
    
    ### initialization
    
    ## initial
    lib.initial.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.initial.restype = ctypes.c_int
    
    ## getFeat
    lib.getFeat.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.getFeat.restype = ctypes.POINTER(FACEINFO)
    
    ## getDetectFeat
    lib.getDetectFeat.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.getDetectFeat.restype = ctypes.POINTER(FACEINFO)

    ## getAllDetectFeat
    lib.getAllDetectFeat.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.getAllDetectFeat.restype = ctypes.POINTER(FACEINFO)

    ## releaseFaceInfo
    lib.releaseFaceInfo.argtypes = [ctypes.POINTER(FACEINFO)]
    lib.releaseFaceInfo.restype = None

    ## releaseArrayFaceInfo
    lib.releaseArrayFaceInfo.argtypes = [ctypes.POINTER(FACEINFO)]
    lib.releaseArrayFaceInfo.restype = None
   
    ## featCompare
    lib.featCompare.argtypes = [ctypes.POINTER(FACEINFO), ctypes.POINTER(FACEINFO)]
    lib.featCompare.restype = ctypes.c_float
    
    ## releaseModel
    lib.releaseModel.argtypes = None
    lib.releaseModel.restype = ctypes.c_int

    #                        #
    #     call function      #
    #                        #

    # demo one
    detector_path = bytes("./models/detect", "utf8")
    feature_path = bytes("./models/recognition", "utf8")
    init_ret = lib.initial(detector_path, feature_path)
    if init_ret == 0:
        print('model initialization success.')
    else:
        print('model initialization failed.')

    image_path = bytes("./image/test.jpg", "utf8")
    landmarks = [313.99, 133.16, 362.24, 139.61, 340.06, 162.63, 304.94, 180.23, 359.52, 187.45]
    landmarks_c = (ctypes.c_float * len(landmarks))(*landmarks)

    result = lib.getFeat(image_path, landmarks_c, 10)

    face_result = {}
    
    face_result['feature'] = np.zeros((result[0].FeatureSize), dtype=np.float32)
    for j in range(result[0].FeatureSize):
        face_result['feature'][j] = result[0].feature[j]

    ## demo two
    image_path3 = bytes("./image/multi-people.jpg", "utf8")
    t_start = time.time()
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)
    result3 = lib.getAllDetectFeat(image_path3, pnum)
    num = pnum[0]
    faceinfo_result3 = []
    if num>=1:
        for face_index in range(num):
            face_result = {}
            
            left = int(result3[face_index].left)
            top = int(result3[face_index].top)
            right = int(result3[face_index].right)
            bottom = int(result3[face_index].bottom)
            
            face_result['x'] = left
            face_result['y'] = top
            face_result['w'] = right - left + 1
            face_result['h'] = bottom - top + 1
            
            # one face feature
            face_result['feature'] = np.zeros((result3[face_index].FeatureSize), dtype=np.float32)
            for j in range(result3[face_index].FeatureSize):
                face_result['feature'][j] = result3[face_index].feature[j]
            faceinfo_result3.append(face_result)
    
    print("detect multi-people cost = {}".format(time.time() - t_start))
    print('faceinfo_result3 = {}'.format(faceinfo_result3))



    image_path1 = bytes("./image/000_0.bmp", "utf8")
    image_path2 = bytes("./image/000_1.bmp", "utf8")
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)
    result1 = lib.getDetectFeat(image_path1, pnum)
    num = pnum[0]
    faceinfo_result1 = []
    if num>=1:
        for face_index in range(num):
            face_result = {}
            
            left = int(result1[face_index].left)
            top = int(result1[face_index].top)
            right = int(result1[face_index].right)
            bottom = int(result1[face_index].bottom)
            
            face_result['x'] = left
            face_result['y'] = top
            face_result['w'] = right - left + 1
            face_result['h'] = bottom - top + 1
            
            # one face feature
            face_result['feature'] = np.zeros((result1[face_index].FeatureSize), dtype=np.float32)
            for j in range(result1[face_index].FeatureSize):
                face_result['feature'][j] = result1[face_index].feature[j]
            faceinfo_result1.append(face_result)
    
    
    
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)
    result2 = lib.getDetectFeat(image_path2, pnum)
    num = pnum[0]
    faceinfo_result2 = []
    if num >=1 :
        for face_index in range(num):
            face_result = dict()
            left = int(result2[face_index].left)
            top = int(result2[face_index].top)
            right = int(result2[face_index].right)
            bottom = int(result2[face_index].bottom)

            face_result['x'] = left
            face_result['y'] = top 
            face_result['w'] = right - left + 1
            face_result['h'] = bottom - top + 1
            
            # one face feature
            face_result['feature'] = np.zeros((result2[face_index].FeatureSize), dtype=np.float32)
            for j in range(result2[face_index].FeatureSize):
                face_result['feature'][j] = result2[face_index].feature[j]
            faceinfo_result2.append(face_result)
    feat1 = faceinfo_result1[0]['feature']/np.linalg.norm(faceinfo_result1[0]['feature'])
    feat2 = faceinfo_result2[0]['feature']/np.linalg.norm(faceinfo_result2[0]['feature'])
    sim_f = 1 - cdist(feat1.reshape(1,-1), feat2.reshape(1,-1), 'cosine')
    sim_c = cal_sim(feat1.reshape(1,-1),feat2.reshape(-1,1))
    ## feature_compare
    score = lib.featCompare(result1, result2)
    print('similarity = {}'.format(score))
    print('sim_f = {} , sim_c = {}'.format(sim_f, sim_c))
    
    



    ## releaseFaceInfo
    lib.releaseFaceInfo(result)
    lib.releaseFaceInfo(result1)
    lib.releaseFaceInfo(result2)
    lib.releaseArrayFaceInfo(result3)
    ## releaseModel
    model_ret = lib.releaseModel()
    if model_ret==0:
        print('model release success.')
    else:
        print('model release failed.')
    


if __name__ == "__main__" :
    lib_path = './lib/libglodonfacesdk.so'
    test(lib_path)
    


