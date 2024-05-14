import os
import cv2
import numpy as np
from Lab_segmentation import LabSegmentation


#* Tomar este programa como referencia para cortar las imagenes
directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/imagenes-recortadas'

# Tamaño deseado de la imagen recortada
porcion_w = 299
porcion_h = 86

directorio_imagenes = './datasets/original-dataset/test/'

for file_name in os.listdir(directorio_imagenes):
    if file_name.endswith(".jpg"):

        ruta_imagen = os.path.join(directorio_imagenes, file_name)
        lab_segmentation = LabSegmentation(ruta_imagen)
        lab_segmentation.img_to_lab()
        lab_segmentation.get_ranges()
        (imagen_gris, mascara) = lab_segmentation.threshold_algorithm()

        imagen = cv2.imread(ruta_imagen)

        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        # Encontrar contornos en la imagen umbralizada
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Encontrar el contorno más grande
        contorno_salmon = max(contornos, key=cv2.contourArea)

        # Obtener un rectángulo delimitador alrededor del contorno del salmón
        x, y, w, h = cv2.boundingRect(contorno_salmon)

        # Recortar la región del salmón de la imagen original
        region_salmon = imagen[y:y+h, x:x+w]
        imagren2 = cv2.drawContours(imagen, contornos, -1, (0, 255, 0), 2)

        # Redimensionar la región del salmón al tamaño deseado (299x86 píxeles)
        region_salmon_redimensionada = cv2.resize(region_salmon, (299, 86))

        # Mostrar la imagen original y la región del salmón redimensionada
        cv2.imshow('Imagen Original', imagen)
        cv2.imshow('Imagen Origina22', imagren2)
        cv2.imshow('Imagen mascara', mascara)
        cv2.imshow('Región del Salmón', region_salmon_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
