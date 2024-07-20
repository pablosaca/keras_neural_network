import os
import numpy as np
import pandas as pd
import glob
import cv2
from sklearn.preprocessing import MinMaxScaler


def load_house_images(df: pd.DataFrame, inputPath: str):
    """
    Cargar las imágenes de los inmuebles desde un archivo.
    
    Parameters
    ----------
    df: pd.DataFrame
        Tabla con los datos
    inputPath: str
        Directorio donde se encuentran las imagenes

    Returns
    -------
    images : np.ndarray
    """
    # initialize our images array (i.e., the house images themselves)
    images = []
    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))
        
        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        outputImage = np.zeros((128, 128, 3), dtype="uint8")
        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 64 64, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (64, 64))
            image = image[:, :, [2, 1, 0]]
            inputImages.append(image)
            
        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        outputImage[0:64, 0:64] = inputImages[0]
        outputImage[0:64, 64:128] = inputImages[1]
        outputImage[64:128, 64:128] = inputImages[2]
        outputImage[64:128, 0:64] = inputImages[3]
        # add the tiled image to our set of images the network will be
        # trained on
        images.append(outputImage)
    # return our set of images
    return np.array(images)


def load_images(
        df: pd.DataFrame,
        inputPath: str,
        alto_ancho: list,
        canal: int,
        tipo_foto=str
):
    """
    Cargar las imágenes de los inmuebles desde un archivo.
    
    Parameters
    ----------
    df: pd.DataFrame
        Tabla con los datos
    inputPath: str
        Directorio donde se encuentran las imagenes
    alto_ancho: list
        Altura y ancho de la imagen
    canal: int
        El canal sería 3 porque las imágenes estána a color
    tipo_foto: str
        Nombre que hace referencia a las cuatro fotos (baño, habitación, cocina y frontal)


    Returns
    -------
    images : np.ndarray
    """

    tipo_foto_list = ["frontal", "bedroom", "kitchen", "bathroom"]
    if tipo_foto not in tipo_foto_list:
        raise ValueError(f"incorrect string. Only available {tipo_foto_list}")

    n = len(df)
    alto = alto_ancho[0]
    ancho = alto_ancho[0]
    # initialize our images array (i.e., the house images themselves)
       
    images = []
    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, f"{i + 1}_{tipo_foto}*"])
        housePaths = sorted(list(glob.glob(basePath)))
        
        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        outputImage = np.zeros((alto, ancho, canal), dtype="uint8")
         # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (alto, ancho))
            image = image[:, :, [2, 1, 0]]
            inputImages.append(image)
  
        images.append(inputImages)
    # return our set of images
    return np.array(images).reshape((n, alto, ancho, canal))


def predict_inverse_transform(y_pred: np.ndarray, mix_max_norm: MinMaxScaler):
    """
    Función para realizar la transformación inversa de las predicciones dadas por el modelo
    Se tiene en cuenta primero la inversa de la normalización y después logartimos.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicciones del modelo (escala norma unidad)
    min_max_norm :  MinMaxScaler
        Método para realizar la normalización de los datos

    Returns
    -------
    y_pred : np.ndarray
        Predicciones del modelo (escala real)
    """    
    y_pred_inv = mix_max_norm.inverse_transform(y_pred)  # desnormalización (sobre norma de datos de entrenamiento)
    y_pred = np.exp(y_pred_inv)
    del y_pred_inv
    return y_pred
