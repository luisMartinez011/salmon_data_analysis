from Lab_segmentation import LabSegmentation
import os
#TODO: Hacer un for loop de las imagenes
directorio_imagenes = './datasets/modified-dataset/imagenes-precortadas/'

for file_name in os.listdir(directorio_imagenes):
    if file_name.endswith(".jpg"):

        ruta_imagen = os.path.join(directorio_imagenes, file_name)
        lab_segmentation = LabSegmentation(ruta_imagen)
        lab_segmentation.img_to_lab()
        lab_segmentation.get_ranges()
        (imagen_gris, mascara) = lab_segmentation.threshold_algorithm()
        lab_segmentation.save_statistics(imagen_gris, mascara)
