import os
import cv2

directorio_imagenes = './datasets/original-dataset/test'

roi_x, roi_y, roi_width, roi_height = 400, 400, 600, 600

for nombre_archivo in os.listdir(directorio_imagenes):
    if nombre_archivo.endswith(".jpg"):
        ruta_imagen = os.path.join(directorio_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen)
        roi = imagen[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        cv2.imwrite(ruta_imagen, roi)
print("Proceso completado.")
