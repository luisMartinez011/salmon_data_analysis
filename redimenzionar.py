import os
import cv2
import numpy as np

directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/test'

# Definir el rango de color salmón en HSV
lower_salmon = np.array([0, 100, 100])
upper_salmon = np.array([30, 255, 255])


# Función para recortar una región de interés (ROI) de la imagen original
def recortar_roi(image, x, y, w, h):
    return image[y:y + h, x:x + w]


for nombre_archivo in os.listdir(directorio_imagenes):
    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)
        cv2.imshow('Imagen Original', imagen)
        cv2.waitKey(0)

        # Convertir RGB a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # Umbralizar la imagen HSV para obtener solo los colores de acuerdo al umbral del color del salmon
        mask = cv2.inRange(hsv, lower_salmon, upper_salmon)

        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Recortar un pedazo del objeto color salmon si se detecta
        if contours:
            # Seleccionar el contorno con el área más grande
            largest_contour = max(contours, key=cv2.contourArea)

            # Encontrar la caja delimitadora ajustada al contorno
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Tamaño del paper
            porcion_w = 299
            porcion_h = 86

            # Recortar la porción de la imagen umbralizada
            imagen_recortada = recortar_roi(imagen, x, y, porcion_w, porcion_h)

            # Mostrar la imagen recortada
            cv2.imshow('Imagen Recortada', imagen_recortada)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(directorio_imagenes_modificadas, nombre_archivo), imagen_recortada)
print("Proceso completado.")
