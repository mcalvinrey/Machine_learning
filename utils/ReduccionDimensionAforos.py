
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'franklin gothic medium'
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.decomposition import PCA

def ReduccionDimensionesAforos():
    pd.set_option('display.max_columns', 500)
    os.chdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src')

    fq = 'M'   # 'D','7D','M','Q','Y' Frec. o nivel granularidad análisis
    nclus = 4  # nº de clusters, a partir de gráficos codo y silhouette 

    # Después de analizar y comparar los patrones semana-hora, y después
    # hacer clusters que hemos visto geográficamente además de con las
    # series mensuales, ahora queremos, para esas mismas series mensuales,
    # reducir la dimensión (en lugar del nº de meses=60), quedarnos con menos),
    # para poder graficarlo y comparar estos gráficos por direcciones, a
    # ver si tienen algún sentido comparados con los gráficos geográficos.
    # Para esto, lo haremos por direcciones, con lo que tomamos éstas del nombre
    # de cada serie.También escalamos los datos para mejor comparabilidad, puesto 
    # que lo que nos interesa es comparar evoluciones, no nivel:

    data = pd.read_csv('data/processed_files/Series.csv', sep = ';', header = 0)
    data['FecHora'] = pd.to_datetime(data.FecHora)
    data.sort_values(by = 'FecHora', inplace = True)
    data.set_index('FecHora', inplace = True)
    #print(data.tail())
    tsagg = data.resample(fq).agg('mean')
    seriesT = tsagg.T
    # Escalamos los datos para mejor comparabilidad
    #scaled2 = MinMaxScaler().fit_transform(tsagg)  
    scaled = pd.DataFrame(minmax_scale(tsagg))
    scaled.columns = tsagg.columns
    scaled.index = tsagg.index

    # Para graficarlo con nombres, buscamos en fichero ubicaciones:
    ubicas = pd.read_csv('data/raw_files/UbicacionEstacionesPermanentesSentidos' +
        '.csv', sep=';', header = 0)
    ubicas['Direc'] = ubicas.apply (lambda x: 'ES0' + str(x['Estación']) +
        '_' + str(x['Orient.']) if x['Estación'] < 10 else 'ES' + 
        str(x['Estación']) + '_' + str(x['Orient.']), axis = 1)
    ubicas = ubicas[['Direc','Nombre']]

    direcciones = ['E-O','N-S','O-E','S-N']

    for D in direcciones:
        series = scaled.filter([col for col in scaled.columns if 
                str(col)[5:8] == D], axis = 1).T
        series.columns = [str(col) for col in series.columns]

        print('\nDirección', D)
        print('Nº de series a reducir', series.shape[0])
        pca = PCA(n_components = 2).fit(series)

        print('Componentes de varianza explicada', pca.explained_variance_ratio_)
        print('Total varianza explicada', pca.explained_variance_ratio_.sum())
        
        proyeccion = pd.DataFrame(pca.transform(series))
        proyeccion['Direc'] = series.index
        proyeccion = pd.merge(proyeccion, ubicas, on = ['Direc'])
        proyeccion['Nome'] = proyeccion.Nombre.str.replace('del ', '',regex=True).\
        str.replace('de la ', '', regex=True).str.replace('de ', '', regex=True).\
        str.replace('Paseo ','', regex=True).str.replace('Avenida ', 'Avda.'
            , regex=True).str.replace('Calle ', '', regex=True).str.replace(
                "(M-30)", "", regex=True)
            
        plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim(min(proyeccion[0]), max(proyeccion[0]))
        ax.axvspan(min(proyeccion[0]), max(proyeccion[0]), facecolor='honeydew')
        ax.tick_params(colors='gray', axis= 'both', size = 2.5, 
                            labelsize = 5, rotation = 0)
        ax.scatter(proyeccion[0], proyeccion[1], c = 'r')
        ax.spines['bottom'].set_color('gray')   
        ax.spines['top'].set_color('gray') 
        ax.spines['right'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        #plt.title(D, fontsize = 12, color = 'gray')
        
        for i, txt in enumerate(proyeccion.Nome):
            ax.annotate(txt, (proyeccion.iloc[i, 0], proyeccion.iloc[i, 1]),
                    fontsize = 9, color = 'black')
        fig.savefig('data/Imagenes_aforos/' + D + '2dim.png', dpi = 320)