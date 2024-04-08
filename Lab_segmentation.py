import os
import cv2
import numpy as np

salmon_fan_score = {
    '20': [72.73, 23.5, 16.35],
    '21': [68.80, 27.80, 21.20],
    '22': [65.95, 30, 26.35],
    '23': [63.30, 32.15, 29.15],
    '24': [60.8, 33.70, 32.45],
    '25': [58.60, 34.05, 33.60],
    '26': [57.70, 32.80, 32.10],
    '27': [56.50, 39.05, 38.10],
    '28': [54.20, 41.70, 1.10],
    '29': [52.35, 44.45, 41.50],
    '30': [50.35, 46, 41],
    '31': [48.35, 45.70, 40.40],
    '32': [47.05, 50.25, 42.10],
    '33': [45.10, 50, 38.80],
    '34': [44.20, 46.20, 33],
}

lab_ranges = {
    '20': [0.55, 0.90, 1.25],
    '21': [0.70, 0.90, 1.40],
    '22': [0.45, 0.50, 1.45],
    '23': [0.40, 0.45, 1.05],
    '24': [0.70, 1.10, 1.35],
    '25': [0.50, 1.05, 1.50],
    '26': [0.90, 2.30, 2.30],
    '27': [0.50, 0.95, 1.50],
    '28': [0.60, 1.10, 1.80],
    '29': [0.45, 0.75, 1.00],
    '30': [0.45, 1.10, 1.50],
    '31': [0.75, 4.00, 2.30],
    '32': [0.45, 1.15, 1.60],
    '33': [0.40, 1.00, 1.60],
    '34': [0.50, 1.40, 2.00],
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
