import os
import cv2
import numpy as np

#* Tomar este programa como referencia para cortar las imagenes
directorio_imagenes = './datasets/original-dataset/test'
directorio_imagenes_modificadas = './datasets/modified-dataset/imagenes-recortadas'
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
        cv2.imshow('Imagen Original', imagen)
        cv2.waitKey(0)

        # Convertir RGB a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # Umbralizar la imagen HSV para obtener solo los colores de acuerdo al umbral del color del salmón
        mask = cv2.inRange(hsv, lower_salmon, upper_salmon)

        # Aplicar operaciones morfológicas para reducir el ruido en la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontrar los límites de la región blanca en la máscara
        indices = np.where(mask == 255)
        y_min, y_max = np.min(indices[0]), np.max(indices[0])
        x_min, x_max = np.min(indices[1]), np.max(indices[1])

        # Calcular los desplazamientos necesarios para centrar la región recortada
        centro_x = (x_min + x_max) // 2
        centro_y = (y_min + y_max) // 2
        dx = max(0, porcion_w // 2 - centro_x)
        dy = max(0, porcion_h // 2 - centro_y)

        # Calcular las nuevas coordenadas para la región recortada
        nuevo_x_min = max(0, x_min - dx)
        nuevo_y_min = max(0, y_min - dy)
        nuevo_x_max = min(imagen.shape[1], x_max + (porcion_w - (x_max - x_min)) // 2)
        nuevo_y_max = min(imagen.shape[0], y_max + (porcion_h - (y_max - y_min)) // 2)

        # Recortar la porción de la imagen umbralizada
        imagen_recortada = recortar_roi(imagen, nuevo_x_min, nuevo_y_min, nuevo_x_max - nuevo_x_min,
                                        nuevo_y_max - nuevo_y_min)

        # Redimensionar la imagen recortada a las dimensiones deseadas
        imagen_recortada = cv2.resize(imagen_recortada, (porcion_w, porcion_h))

        # Mostrar la imagen recortada
        cv2.imshow('Imagen Recortada', imagen_recortada)
        cv2.waitKey(0)

        # Guardar la imagen recortada
        cv2.imwrite(os.path.join(directorio_imagenes_modificadas, nombre_archivo), imagen_recortada)

print("Proceso completado.")
