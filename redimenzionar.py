import os
import cv2
import numpy as np

directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/test'

# Definir el rango de color salmon en HSV
lower_salmon = np.array([100, 100, 0])
upper_salmon = np.array([255, 255, 100])


# Función para recortar una región de interés (ROI) de la imagen original
def recortar_roi(image, x, y, w, h):
    return image[y:y + h, x:x + w]


for nombre_archivo in os.listdir(directorio_imagenes):
    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)

        # Convertir RGB a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)

        # Umbralizar la imagen HSV para obtener solo los colores azules
        mask = cv2.inRange(hsv, lower_salmon, upper_salmon)

        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Recortar un pedazo del objeto azul si se detecta
        if contours:
            # Seleccionar el contorno más grande (suponiendo que sea el objeto azul)
            largest_contour = max(contours, key=cv2.contourArea)

            # Obtener las coordenadas del rectángulo delimitador del contorno
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Recortar el área del objeto azul de la imagen original
            imagen_recortada = recortar_roi(imagen, x, y, w, h)
            cv2.imwrite(directorio_imagenes_modificadas, imagen_recortada)


print("Proceso completado.")
