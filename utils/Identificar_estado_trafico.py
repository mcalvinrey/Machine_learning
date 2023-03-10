import time, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import xml.etree.ElementTree as ET
from shapely.geometry import Point
import geopandas as gpd

def Identificar_estado_trafico():

    class_names = [0, 1]
    tt = 128   # Tamaño de las fotos (size = (tt*rat, tt))(ancho, alto)
    rat = 1.5  # Ratio ancho/alto para fotos
    batch_size = 32    # Batch_size 

    img_height = tt
    img_width = int(tt * rat)

    # Extraemos Nombre, Latitud y Longitud de las cámaras
    camaras = ET.parse("C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/raw_files/camaras.xml")
    raiz_camaras = camaras.getroot()
    Latitud = []
    Longitud = []
    Nombre = []
    for val in raiz_camaras.iter():
        if val.tag == "Posicion":
            Latitud.append(val[0].text)
            Longitud.append(val[1].text)
        elif val.tag == "Nombre":
            Nombre.append(val.text)
    ubi = pd.DataFrame({'Nombre': Nombre, 'Latitud': Latitud, 'Longitud': Longitud})

    #Cargamos el modelo
    loaded_model = tf.keras.models.load_model(
            "C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/modelos/M-30/modelAug")

    #Creamos el Df que tendrá las predicciones y lo rellenamos para las imágenes guardadas en C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/camaras_09_03_20h/
    predic = pd.DataFrame(columns=['Nombre', 'Estado_traf', "prob"])
    #print(len(os.listdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/camaras_09_03_20h/')), len(ubi))

    for ind, valor in enumerate(os.listdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/camaras_09_03_20h/')):
        try:
            Nombre = str(valor)[:-4]
            #print(Nombre, ubi['Nombre'][ind])
            
            img = tf.keras.utils.load_img('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/camaras_09_03_20h/'+valor, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch
            predictions = loaded_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predic.loc[ind,'Nombre']= Nombre
            predic.loc[ind,'Estado_traf']= class_names[np.argmax(score)]
            predic.loc[ind,'prob']= round(100 * np.max(score))
            
        except: pass

    print(ubi.Nombre, predic.Nombre)

    # Juntamos los DataFrames de ubi y predic en un único DataFrame
    ubicaciones = pd.merge(ubi, predic, on = ['Nombre'])

    # Creamos la columna 'Coords' como "Punto" a partir de Longitud y Latitud
    # Con geopandas creamos un geodataframe a partir de ubicaciones y lo guardamos como archivo .shp para poder utilizarlo en Qgis   
    ubicaciones['Coords'] = list(zip(ubicaciones.Longitud, ubicaciones.Latitud))
    print(type(ubicaciones.Coords))
    ubicaciones['Coords'] = ubicaciones.Coords.apply(lambda x : Point(x))
    gdf = gpd.GeoDataFrame(ubicaciones, geometry='Coords', crs = 'EPSG:4326')
    gdf.reset_index(drop = True, inplace = True)
    print(gdf.head())
    gdf.to_file(driver = 'ESRI Shapefile', filename='C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/Maps_files/ubicas.shp')

