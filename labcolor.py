import os
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb

directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/test'

# Definir el rango de color salmón en HSV
lower_salmon = np.array([0, 100, 100])
upper_salmon = np.array([30, 255, 255])


# Función para recortar una región de interés (ROI) de la imagen original
def recortar_roi(image, x, y, w, h):
    return image[y:y + h, x:x + w]

def plotMinMax(Xsub_rgb,labels=["R","G","B"]):
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = np.min(Xsub_rgb[:,:,:,i])
        ma = np.max(Xsub_rgb[:,:,:,i])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))


for nombre_archivo in os.listdir(directorio_imagenes):
    directorio_imagenes_modificadas = directorio_imagenes_modificadas + "/" + nombre_archivo

    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)


        lab = cv2.cvtColor(imagen,cv2.COLOR_RGB2LAB)


        L,A,B=cv2.split(lab)
        print(L,A,B)
        # img = np.asarray(imagen, np.uint8)
        # Xsub_lab = rgb2lab(img)
        # plotMinMax(Xsub_lab,labels=["L","A","B"])

def salmon_color():
    colores_
