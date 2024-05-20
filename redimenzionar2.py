import os
import cv2
import numpy as np

# Tomar este programa como referencia para cortar las imagenes

directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/imagenes-precortadas'
# Definir el rango de color salmón en HSV
lower_salmon = np.array([0, 100, 100])
upper_salmon = np.array([30, 255, 255])

# Tamaño deseado de la imagen recortada
porcion_w = 299
porcion_h = 86


# Función para recortar una región de interés (ROI) de la imagen original
def recortar_roi(image, x, y, w, h):
    return image[y:y + h, x:x + w]


for nombre_archivo in os.listdir(directorio_imagenes):
    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)
        # cv2.imshow('Imagen Original', imagen)
        # cv2.waitKey(0)

        # Convertir RGB a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # Umbralizar la imagen HSV para obtener solo los colores de acuerdo al umbral del color del salmón
        mask_salmon = cv2.inRange(hsv, lower_salmon, upper_salmon)

        # Encontrar los contornos en la máscara
        contours, _ = cv2.findContours(mask_salmon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar el contorno más grande (el contorno del salmón)
        if contours:
            contour_salmon = max(contours, key=cv2.contourArea)

            # Obtener las coordenadas del rectángulo delimitador del contorno del salmón
            x, y, w, h = cv2.boundingRect(contour_salmon)

            # Redimensionar el rectángulo delimitador para que abarque toda la imagen del salmón
            x = max(0, x - (porcion_w - w) // 2)
            y = max(0, y - (porcion_h - h) // 2)
            w = min(imagen.shape[1] - x, porcion_w)
            h = min(imagen.shape[0] - y, porcion_h)

            # Recortar la porción de la imagen original
            imagen_recortada = recortar_roi(imagen, x, y, w, h)

            # Mostrar la imagen recortada
            # cv2.imshow('Imagen Recortada', imagen_recortada)
            # cv2.waitKey(0)

            # Guardar la imagen recortada
            cv2.imwrite(os.path.join(directorio_imagenes_modificadas, nombre_archivo), imagen_recortada)

print("Proceso completado.")






