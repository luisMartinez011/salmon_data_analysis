## Descripción del Proyecto: Clasificación de Pixeles de Imágenes de Salmones

El objetivo de este proyecto es clasificar los pixeles de imágenes de salmones de acuerdo al color del salmónfan utilizando el espacio de color LAB. Para ello, se emplea un enfoque de segmentación y procesamiento de imágenes. El proyecto se divide en dos partes principales: el procesamiento y clasificación del salmón en el rango del salmónfan y el recorte de las imágenes al formato deseado.

### Descripción de Archivos:

#### Lab_segmentation.py

Este archivo se encarga del procesamiento y clasificación del salmón en el rango del salmónfan utilizando el espacio de color LAB. Incluye funciones para convertir las imágenes al espacio de color LAB, aplicar umbralización para detectar el salmón, encontrar contornos y clasificar los pixeles de acuerdo al salmónfan.

#### redimensionar2.py

Este archivo se encarga de recortar las imágenes en un formato de 299x86 píxeles. Contiene funciones para recortar y redimensionar las imágenes al tamaño deseado.

### Carpeta "datasets"

En esta carpeta se encuentran los conjuntos de datos utilizados en el proyecto. Contiene dos subcarpetas:

#### modified-dataset

Esta carpeta contiene las subcarpetas donde se guardan los distintos elementos del conjunto de datos, incluyendo:

- Histograma de las imágenes.
- Imágenes en escala de grises.
- Imágenes umbralizadas.
- Imágenes recortadas.
- Matrices de las imágenes en formato LAB.
-

#### modified-dataset

Esta carpeta se encarga de guardar las imagenes originales sin editar
