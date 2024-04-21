import os
import cv2
import numpy as np
from skimage import data
from skimage.color import rgb2lab, lch2lab, lab2lch,deltaE_ciede94




#Links/ Bibliografia
# https://stackoverflow.com/questions/65509486/return-boolean-from-cv2-inrange-if-color-is-present-in-mask

class LabSegmentation():

    directorio_imagenes = './datasets/original-dataset/train/farmstest-3_jpg.rf.a29f651a56b87147437c5abd12e68c66.jpg'

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
    def  __init__(self):
        self.original_image = self.get_image()

    def example(self):
        img = data.astronaut()
        img_lab = rgb2lab(img)
        delta_e = deltaE_ciede94(img_lab, [3,5,6])
        print(delta_e)
        img_lch = lab2lch(img_lab)
        img_lab2 = lch2lab(img_lch)

    def get_image(self):
        img_original = cv2.imread(self.directorio_imagenes)
        imagen_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        return imagen_rgb

    def get_ranges(self):
        lab_ranges = self.lab_ranges
        arreglo_vacio = np.zeros_like(self.imagen)

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
            # lowerRange = np.array([l_lower_range, a_lower_range, b_lower_range] , dtype="uint16")
            # upperRange = np.array([l_upper_range, a_upper_range, b_upper_range], dtype="uint16")


            self.compare_salmonfan(lowerRange, upperRange)
            # self.convert_to_lab(image)
            # self.example(image)



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

                # Update the pixel values in the modified image
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

    def compare_salmonfan(self, lowerRange, upperRange):
        #TODO: Convertir bien la imagen en formato lab de acuerdo a lo del paper
        #TODO:

         # Make a copy of the original image
        lab_image = self.lab_image

        # Get the height and width of the image
        height, width, _ = lab_image.shape

        # Iterate over each pixel of the image
        for x in range(height):
            for y in range(width):
                # Get the BGR values of the pixel
                L,a,b = lab_image[x, y]

                if lowerRange

                # Update the pixel values in the modified image
                modified_image[x,y] = [l,a,b]


        # mask = image[:].copy()
        # # image = (image/256).astype('uint8')
        # imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # imageRange = cv2.inRange(imageLab,lowerRange, upperRange)
        # # cv2.imshow('prueba', imageLab)

        # ones = cv2.countNonZero(imageRange)


        # mask[:,:,0] = imageRange
        # mask[:,:,1] = imageRange
        # mask[:,:,2] = imageRange

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # faceLab = cv2.bitwise_and(image,mask)

        # cv2.imshow("imagen salmon",faceLab )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return faceLab

lab_segmentation = LabSegmentation()
lab_segmentation.img_to_lab()
lab_segmentation.get_ranges()
