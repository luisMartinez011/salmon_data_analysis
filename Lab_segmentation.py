import os
import cv2
import numpy as np

salmon_fan_score = {
    '20': [72.73, 23.5, 16.35],
    '21': [68.80, 27.80, 21.20],
    '22': [65.95, 30, 26.35],
    '23': [63.30, 32.15, 29.15],
    '24': [4, 5, 6],
    '25': [4, 5, 6],
    '26': [4, 5, 6],
    '27': [4, 5, 6],
    '28': [4, 5, 6],
    '29': [4, 5, 6],
    '30': [4, 5, 6],
    '31': [4, 5, 6],
    '32': [4, 5, 6],
    '33': [4, 5, 6],
    '34': [4, 5, 6],
}

salmon_fan = {
    '20': [1, 2, 3],
    '21': [4, 5, 6],
    '22': [7, 8, 9],
    '23': [4, 5, 6],
    '24': [4, 5, 6],
    '25': [4, 5, 6],
    '26': [4, 5, 6],
    '27': [4, 5, 6],
    '28': [4, 5, 6],
    '29': [4, 5, 6],
    '30': [4, 5, 6],
    '31': [4, 5, 6],
    '32': [4, 5, 6],
    '33': [4, 5, 6],
    '34': [4, 5, 6],
}

def Lab_Segmentation(image):
    lowerRange= np.array([0, 135, 135] , dtype="uint8")
    upperRange= np.array([255, 160, 195], dtype="uint8")
    mask = image[:].copy()

    imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imageRange = cv2.inRange(imageLab,lowerRange, upperRange)

    mask[:,:,0] = imageRange
    mask[:,:,1] = imageRange
    mask[:,:,2] = imageRange

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(image,mask)

    return faceLab
