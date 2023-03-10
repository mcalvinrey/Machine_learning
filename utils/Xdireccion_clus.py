
import os
import pandas as pd
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
#import shapely.speedups

def Xdireccion_clus():
    os.chdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src')

    filin = 'KmeansDtw'    # Fichero con series y barycenter (rows)
    nclus = 4                    # nº de clusters
    fq = 'M'   # 'D','7D','M','Q','Y' Frec. o nivel granularidad análisis

    # Se separan las series por dirección (NS, SN, EO, OE) en ficheros
    # shp para gráficos en QGis con clusters obtenidos en paso anterior

    def cortoNom(x):
        nome = x.replace('de la', '').replace('Calle de', '')
        nome = nome.replace('Avenida', 'Avda.').replace('Calle', '')
        nome = nome.replace(' del ', ' ').replace('Paseo','')
        nome = nome.replace(' de ', ' ').replace('Fernández', 'Fdez.')
        nome = nome.replace('(M-30)','')
        return nome 


    ubicas = pd.read_csv('data/raw_files/UbicacionEstacionesPermanentesSentidos' +
        '.csv', sep=';', header = 0)
    ubicas['Direc'] = ubicas.apply (lambda x: 'ES0' + str(x['Estación']) +
        '_' + str(x['Orient.']) if x['Estación'] < 10 else 'ES' + 
        str(x['Estación']) + '_' + str(x['Orient.']), axis = 1)
    ubicas = ubicas[['Direc','Nombre','Orient.','Latitud', 'Longitud']]
    clusters =  pd.read_csv( 'C:/Users/mcalv/Desktop/Proyectos/Machine learning/src/data/processed_files/' + filin + 
            str(nclus) + str(fq) + '.csv', sep = ';',header = 0)
    clusters.rename( columns={'Unnamed: 0':'Direc'}, inplace=True )
    # Para borrar baricentros, cuando los hay:
    clusters = clusters[clusters['Direc'].apply(lambda x: len(x)) > 4]
    clusters = clusters[['Direc', 'clus']]
    final = pd.merge(ubicas, clusters, on = ['Direc'])
    final['Nome'] = final.Nombre.apply(cortoNom)
    tipos = sorted(final['Orient.'].unique())
        
    # Longitud y Latitud tienen ,; hay que poner .;
    final['Longitud'] = final.Longitud.replace({',':'.'}, regex = True). \
        astype(float)
    final['Latitud'] = final.Latitud.replace({',':'.'}, regex = True). \
        astype(float)
        
    for tipo in tipos:
        df = final[final['Orient.'] == tipo]
        df['Coords'] = list(zip(df.Longitud, df.Latitud))
        print(type(df.Coords))
        df['Coords'] = df.Coords.apply(lambda x : Point(x))
        gdf = gpd.GeoDataFrame(df, geometry='Coords', crs = 'EPSG:4326')
        gdf.reset_index(drop = True, inplace = True)
        print(gdf.head())
        gdf.to_file(driver = 'ESRI Shapefile', filename='data/Maps_files/' + tipo + filin +
                    str(nclus) + str(fq) +".shp")
