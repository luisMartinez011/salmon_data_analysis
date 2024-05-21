import os
import cv2
import numpy as np
from skimage import data
from skimage.color import rgb2lab, lch2lab, lab2lch,deltaE_ciede94
import sys

import matplotlib.pyplot as plt




np.set_printoptions(threshold=sys.maxsize)

#Links/ Bibliografia
# https://stackoverflow.com/questions/65509486/return-boolean-from-cv2-inrange-if-color-is-present-in-mask

class LabSegmentation():



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

    output_dir = {
        'histogramas': './datasets/modified-dataset/histogramas/',
        # 'imagenes': './datasets/modified-dataset/imagenes/',
        'matrices': './datasets/modified-dataset/matrices/',
        'imagenes_umbralizadas': './datasets/modified-dataset/imagenes_umbralizadas/',
    }

    #* Atributos
    # ?Imagen convertida a formato lab
    # lab_image

    #file_name

    # ?Es la imagen original sin editar
    #original_image

    # ?Matriz con los score de los pixeles de acuerdo al salmonfan
    # salmon_score

    def  __init__(self,imagen_path):
        self.original_image = self.get_image(imagen_path)
        self.file_name = os.path.splitext(os.path.basename(imagen_path))[0]


    #* Obtiene la imagen en rgb
    def get_image(self,imagen_path):
        img_original = cv2.imread(imagen_path)
        imagen_rgb = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        return imagen_rgb

    #* Obtiene los rangos del salmon fan card,
    #* Y los envia a una funcion donde se encarga de asignar
    #* cada pixel con su respectivo rango
    def get_ranges(self):
        lab_ranges = self.lab_ranges
        #Pixeles clasificados a que SalmonFan pertencen
        height, width, _ = self.lab_image.shape
        salmon_score = np.zeros((height, width), dtype=np.uint8)

        for score, colors in self.salmon_fan_score.items():
            l = colors[0]
            a = colors[1]
            b = colors[2]

            l_range = lab_ranges[score][0]
            a_range = lab_ranges[score][1]
            b_range = lab_ranges[score][2]

            l_lower_range = l - l_range
            a_lower_range = a - a_range
            b_lower_range = b - b_range

            l_upper_range = l + l_range
            a_upper_range = a + a_range
            b_upper_range = b + b_range

            lowerRange = {
                'L': l_lower_range,
                'a': a_lower_range,
                'b': b_lower_range
                }
            upperRange = {
                'L': l_upper_range,
                'a': a_upper_range,
                'b': b_upper_range
                }

            self.compare_salmonfan(lowerRange, upperRange, score,salmon_score)

        self.salmon_score = salmon_score
        # print(salmon_score)
        elementos_no_cero = np.count_nonzero(salmon_score)
        print('numero de pixeles clasificados: ', elementos_no_cero)


    #* Convierte la imagen a lab
    def img_to_lab(self):
        original_image = self.original_image

        # Make a copy of the original image
        modified_image = original_image.copy()

        # Get the height and width of the image
        height, width, _ = original_image.shape

        # Iterate over each pixel of the image
        for x in range(height):
            for y in range(width):
                # Get the BGR values of the pixel
                R,G,B = original_image[x, y]
                l,a,b = self.rgb_to_lab(R,G,B)

                modified_image[x,y] = [l,a,b]
        self.lab_image = modified_image

    #* Convierte un pixel a lab
    # L* is the luminance
    # or lightness component, it ranges from 0 to 100, while a*
    # (green to red) and b* (blue to yellow) are two chromatic
    # components, with values varying from −120 to +120
    def rgb_to_lab(self, r,g,b):
        def func(t):
            if (t > 0.008856):
                return np.power(t, 1/3.0);
            else:
                return 7.787 * t + 16 / 116.0;

        #Conversion Matrix
        matrix = [[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]]

        # RGB values lie between 0 to 1.0
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        rgb = [r, g, b] # RGB

        cie = np.dot(matrix, rgb);

        cie[0] = cie[0] /0.950456;
        cie[2] = cie[2] /1.088754;

        # Calculate the L
        L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];

        # Calculate the a
        a_original = 500*(func(cie[0]) - func(cie[1]));

        # Calculate the b
        b_original = 200*(func(cie[1]) - func(cie[2]));

        #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100
        # Lab = [L, a, b];

        # Definir los rangos original y nuevo
        rango_original = (-128, 127)
        rango_nuevo = (-120, 120)

        # Aplicar la transformación lineal utilizando np.interp
        a = np.interp(a_original, rango_original, rango_nuevo)
        b = np.interp(b_original, rango_original, rango_nuevo)

        Lab_OpenCV = [L, a, b];
        return Lab_OpenCV


    def compare_salmonfan(self, lowerRange, upperRange, current_score, salmon_score):

        lab_image = self.lab_image
        height, width, _ = lab_image.shape

        for x in range(height):
            for y in range(width):
                L,a,b = lab_image[x, y]

                check_L = lowerRange['L'] <= L <= upperRange['L']
                check_a = lowerRange['a'] <= a <= upperRange['a']
                check_b = lowerRange['b'] <= b <= upperRange['b']
                if check_L or( check_a or check_b):
                    salmon_score[x,y] = current_score
                else:
                    continue

    # TODO: Preguntar si este pedo esta bien
    #*Aplica el umbral binario donde
    # * los pixeles negros, son lo que corresponde al salmon
    # * Y los pixeles blancos son lo que NO corresponde
    def threshold_algorithm(self):
        img_original = self.original_image
        salmon_score = self.salmon_score
        umbral = 1  # Umbral de binarización
        _, mascara = cv2.threshold(salmon_score, umbral, 255, cv2.THRESH_BINARY)
        elementos_no_cero = np.count_nonzero(mascara)
        print('numero de threshold: ', elementos_no_cero)
        imagen_gris = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)

        # Aplicar la máscara a la imagen original para resaltar los píxeles no categorizados
        # imagen_resaltada = cv2.bitwise_and(imagen_gris, mascara, mask=mascara)
        # plt.imshow(imagen_resaltada, cmap='gray')
        # plt.title('Imagen Umbralizada')
        # plt.axis('off')

        # plt.show()
        return (imagen_gris, mascara)

    # TODO: Preguntar si asi van a querer la entrega del proyecto
    def save_statistics(self, imagen_gris, mascara):
        self.save_plots(imagen_gris, mascara)
        self.save_matrix()

    def save_matrix(self):
        lab_image = self.lab_image
        file_name = self.file_name
        folder_name = self.output_dir['matrices']
        path = os.path.join(folder_name, file_name)
        path = path + ".txt"

        row, col, _ = lab_image.shape
        matrix_2d = lab_image.reshape(row* col, 3)
        np.savetxt(path, matrix_2d, fmt="%d")


    def save_plots(self, imagen_gris, mascara):

        plt.subplot(2, 1, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.title('Imagen en Escala de Grises')
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(mascara, cmap='gray')
        plt.title('Imagen Umbralizada')
        plt.axis('off')
        # plt.show()

        #* Guarda las dos imagenes
        file_name = self.file_name
        folder_name = self.output_dir['imagenes_umbralizadas']
        path = os.path.join(folder_name, file_name)
        path = path + ".png"
        plt.savefig(path)
        plt.close()

        salmon_score = self.salmon_score
        elementos = salmon_score.ravel()
        elementos_sin_cero = elementos[elementos != 0]

        # Definir los bins para el histograma (del 20 al 34)
        bins = np.arange(20, 35)
        bins2 = np.arange(20, 36)
        # Calcular el histograma con bins predefinidos
        histograma, _ = np.histogram(elementos_sin_cero, bins=bins2)

        # Crear un array de valores del 20 al 34 para representar todos los posibles valores en el histograma
        valores_posibles = np.arange(20, 35)

        # Rellenar el histograma con ceros donde no hay valores en tus datos
        histograma_completo = np.zeros_like(valores_posibles)
        histograma_completo[np.isin(valores_posibles, bins)] = histograma

        histograma_completo = histograma_completo / 1e4

        # Crear el histograma
        plt.bar(valores_posibles, histograma_completo)
        plt.xlabel("SalmonFan Score")
        plt.ylabel("Pixels x 10^4")
        plt.title("Histograma de los salmon fan")
        # plt.show()

        #* Guarda el histograma
        file_name = self.file_name
        folder_name = self.output_dir['histogramas']
        path = os.path.join(folder_name, file_name)
        path = path + ".png"
        plt.savefig(path)
        plt.close()


