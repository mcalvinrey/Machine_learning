

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.metrics import silhouette_score
from tslearn import clustering, metrics

def clustersAforos():
    pd.set_option('display.max_columns', 500)

    os.chdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src')

    fq = 'M'   # 'D','7D','M','Q','Y' Frec. o nivel granularidad análisis
    nclus = 4  # nº de clusters, a partir de gráficos codo y silhouette después

    # Después de analizar y comparar los patrones semana-hora,
    # se pretende realizar algún tratamiento de aprendizaje automático
    # o minería de datos. Se propone hacer
    # clusters, lo q supone usar algún método no supervisado. 

    data = pd.read_csv('data/processed_files/Series.csv', sep = ';', header = 0)
    data['FecHora'] = pd.to_datetime(data.FecHora)
    data.sort_values(by = 'FecHora', inplace = True)
    data.set_index('FecHora', inplace = True)
    print(data.tail())

    # Queremos clusters q tengan en cuenta la evolución en el tiempo, no los
    # patrones semana-hora q hemos estudiado. Cambiamos nivel de granularidad
    # a mensual u otro, diario demasiado detallado y costoso computacionalmente.
    tsagg = data.resample(fq).agg('mean')
    seriesT = tsagg.T
    # Escalamos los datos para mejor comparabilidad
    #scaled2 = MinMaxScaler().fit_transform(tsagg)  
    scaled = pd.DataFrame(minmax_scale(tsagg))
    scaled.columns = tsagg.columns
    scaled.index = tsagg.index
    # Como ts_Kmeans los toma en rows...
    scaledT = scaled.T
    scaledT.columns = scaled.index.astype('str')
    # Una primera idea de similaridad vendrá dada por la correlación entre series
    cor = tsagg.corr()
    # Graficamente:
    fig, ax = plt.subplots()
    #fig.tight_layout()
    ax.set_xticks(np.arange(scaled.shape[1], step = 10), labels =
                            seriesT.index[::10], fontsize = 4)
    ax.set_yticks(np.arange(scaled.shape[1], step = 10), labels =
                            seriesT.index[::10], fontsize = 4)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",
            rotation_mode="anchor")
    im = ax.imshow(cor, cmap = 'YlOrRd_r')
    bar = plt.colorbar(im)
    bar.ax.tick_params(labelsize = 5)
    bar.set_label('Nivel correlación', fontsize = 7)
    plt.title('Correlación entre series de aforos ' + fq)
    plt.savefig('data/Imagenes_aforos/Corr' + fq +'.jpeg', dpi = 600)
    #plt.show()
    
    # La idea para hacer los clusters es utilizar alguna medida de distancia o 
    # disimilaridad, (a veces se usa 1- abs(cor)) pero no bueno pq pueden haber
    # muchas correlaciones cruzadas importantes con distinto nº de lags q no están
    # contempladas usando solo la correlación contemporánea (lags = 0). Se podría
    # usar el KMeans de sklearn q utiliza la distancia euclídea punto a punto en el
    # mismo momento del tiempo. Especialmente para time series,
    # el paquete tslearn hace clusters q utilizan distintas distancias, por ej.,
    # la euclídea (se supone resultados iguales a los de sklearn.KMeans). También
    # tiene la disimilaridad tdw (le falta la propiedad triangular para ser medida 
    # de distancia) q lo q hace es flexibilizar la medida euclídea considerando en 
    # cada punto la distancia euclídea al más cercano de la otra serie contando
    # también los puntos q están en momentos del tiempo próximos. Se ve claro en la 
    # función que se define a continuación q admite la distancia entre un momento
    # antes y un momento después al actual...
    '''
    # Es lento pero válido. Lo dejo aquí pq así se entiende mejor lo q
    # hace la distancia dtw...
    def calcula_dtw(a1, a2):
        #Computa el dynamic time warping entre dos sequences
        dtw = {}
        for i in range(len(a1)):
            dtw[(i, -1)] = float('inf')
        for i in range(len(a2)):
            dtw[(-1, i)] = float('inf')
        dtw[(-1, -1)] = 0

        for i in range(len(a1)):
            for j in range(len(a2)):
                dist = (a1[i]-a2[j])**2
                dtw[(i, j)] = dist + min(dtw[(i-1, j)], dtw[(i, j-1)], 
                        dtw[(i-1, j-1)])
        return np.sqrt(dtw[len(a1)-1, len(a2)-1])

    DTW = np.zeros((scaled.shape[1], scaled.shape[1]))
    for i in tqdm(range(scaled.shape[1])):
        for j in tqdm(range(scaled.shape[1])):
            DTW[i, j] = calcula_dtw(scaled.iloc[:,i], scaled.iloc[:,j])
    '''
    # Hay un algoritmo más rápido en tslearn para calcularlo: 
    DTW = np.zeros((scaled.shape[1], scaled.shape[1]))
    for i in tqdm(range(scaled.shape[1])):  # tqdm muestra la barra de progreso :)
        for j in tqdm(range(scaled.shape[1])):
            DTW[i, j] = metrics.dtw(scaled.iloc[:,i], scaled.iloc[:,j])
    # Graficamente:
    fig, ax = plt.subplots()
    #fig.tight_layout()
    ax.set_xticks(np.arange(scaled.shape[1], step = 10), labels =
                            seriesT.index[::10], fontsize = 4)
    ax.set_yticks(np.arange(scaled.shape[1], step = 10), labels =
                            seriesT.index[::10], fontsize = 4)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
            rotation_mode = "anchor")
    im = ax.imshow(DTW, cmap = 'YlOrRd')
    bar = plt.colorbar(im)
    bar.ax.tick_params(labelsize = 5)
    bar.set_label('Nivel DTW', fontsize = 7)
    plt.title('DTW entre series de aforos ' + fq)
    plt.savefig('data/Imagenes_aforos//Dtw' + fq +'.jpeg', dpi = 600)
    #plt.show()
    
    # Se ve q, similarmente a lo q ocurre con la correlación, algunas series
    # como ES06_SN (Arturo Soria), ES10_OE (Avda Oporto), ES14_EO (Ortega y Gasset)
    # están bastante distantes de todas las demás

    # Hacemos función q ejecuta KMeans y plot gráficos para nº clusters: 
    def ts_kmeans(data, max_clusters = 10, metric = 'dtw', seed = 823,
                score = 'codo'):
        #Runs KMeans according to max_cluster range
        distortions = []
        rango = range(2, max_clusters + 1)
        for K in tqdm(rango):
            model = clustering.TimeSeriesKMeans(n_clusters = K, 
                metric= metric, n_jobs = -1, max_iter = 10, random_state = seed)
            model.fit(data)
            if score == 'codo':
                distortions.append(model.inertia_)
            else:
                distortions.append(silhouette_score(data, model.labels_))
        plt.figure(figsize=(10,4))
        plt.plot(rango, distortions, 'bx-')
        plt.xlabel('Nº clusters')
        if score == 'codo':
            plt.ylabel('Inercia')
            plt.title('Método del codo')
        else:
            plt.ylabel('Puntuación S.')
            plt.title('Método Silhouette')
        plt.show()
        return

    # Para ts_kmeans las series deben estar en filas => scaled.T
    # Vemos gráficos para hasta 10 clusters con distancias:
    ts_kmeans(data = scaledT, max_clusters = 10, metric='dtw', seed = 123)
    ts_kmeans(data = scaledT, max_clusters = 10, metric='dtw', seed = 123, 
            score = 'silhouette')
    ts_kmeans(data = scaledT, max_clusters = 10, metric='euclidean', seed = 123, 
            score = 'silhouette')
    # El método silhouette parece que elige 2 clusters, mientras codo quizás 4
    # Los otros dan información parecida:
    #ts_kmeans(data = scaled.T, max_clusters = 10, metric='softdtw', seed = 24)
    #ts_kmeans(data = scaled.T, max_clusters = 10, metric='euclidean', seed = 24)

    # Ahora, función para computar inercia. Primero defino la
    # distancia euclidea para vectores (time series puede considerarse así)
    def euclidean_distance(a, b):  # a y b son vectores en cualquier dim
        return sum((y-x)**2 for x, y in zip(a, b)) ** 0.5

    def inerciaB(mm, eliminar = 0):
        # Suma de Cuadrados de Distancias a Centro De Gravedad
        # La entrada es dataframe q incluye series (traspuestas) con last column
        # el nº del cluster al q pertenece serie. 
        # eliminar se utiliza para no considerar en la suma un cluster concreto,
        # por ej. DBSCAN q tiene grupo de outliers y lo ponemos como cluster...
        mm.columns = [*mm.columns[:-1], 'clus']
        #mm.columns = list(mm.columns[:-1]) + ['clus']
        cdgs = mm.groupby('clus').mean()
        scdg = sum([euclidean_distance(mm.iloc[i,:-1], cdgs.loc[mm.iloc[i,-1]])
                    **2 if mm.iloc[i, -1] != eliminar else 0 
                    for i in range(len(mm))])
        return scdg
    
    # Ahora hacemos los modelos con != métricas para nº clusters propuesto arriba:    
    model =  clustering.TimeSeriesKMeans(n_clusters = nclus, metric = 'dtw',
        random_state= 51, n_init = 10).fit(scaled.T)
    seriesT['clus'] = model.fit_predict(scaledT) + 1
    scaledT['clus'] = seriesT['clus'] 
    # Para computar el barycenter, se plantea como problema de optimización (min)
    # Así, cuanto más próxima está serie a barycentro, mayor peso tiene esa serie
    # en el cómputo:
    barycenter = model.cluster_centers_
    # Como queremos guardar los baricentros junto con las series:
    barycenter = np.reshape(barycenter, (barycenter.shape[0], 
                    barycenter.shape[1])) 
    barycenters = pd.DataFrame(barycenter)
    barycenters.columns = seriesT.columns[:-1]
    barycenters['clus'] = list(range(1, nclus + 1))
    barycenters.index = ['c'+str(j) for j in range(1, nclus + 1)]
    clusters = pd.concat((seriesT,barycenters), axis = 0)
    # Lo guardamos para graficar después:
    clusters.to_csv('data/processed_files/KmeansDtw' + str(nclus) + str(fq) + '.csv', index = True, 
                    mode = 'w', sep =';', header = True)

    print('\nINERCIA de KMEANS con distancia DTW', model.inertia_)
    print('INERCIA B de KMEANS con distancia DTW',  inerciaB(scaledT))
    print('INERCIA C de KMEANS con distancia DTW',  
        sum(model.transform(scaled.T).min(axis = 1)**2))

    # Ahora, ¿cuantas series hay en cada cluster?
    print(seriesT.clus.value_counts(normalize = False))

    # Repetimos todo ahora usando las otras 2 métricas:
    # (Con frequencia daily nunca termina, demasiado largo)...
    model2 =  clustering.TimeSeriesKMeans(n_clusters = nclus, n_init = 10,
            metric="softdtw", random_state= 87).fit(scaled.T)
    seriesT['clus'] = model2.fit_predict(scaled.T) + 1
    scaledT['clus'] = seriesT['clus'] 
    barycenter = model2.cluster_centers_
    barycenter = np.reshape(barycenter, (barycenter.shape[0], 
                    barycenter.shape[1])) 
    barycenters = pd.DataFrame(barycenter)
    barycenters.columns = seriesT.columns[:-1]
    barycenters['clus'] = list(range(1, nclus + 1))
    barycenters.index = ['c'+str(j) for j in range(1, nclus + 1)]
    clusters = pd.concat((seriesT,barycenters), axis = 0)
    # Lo guardamos para graficar después:
    clusters.to_csv('data/processed_files/KmeansSoftdtw'+ str(nclus)+ str(fq) +'.csv',
                index = True, mode = 'w', sep =';', header = True)
    print('\nINERCIA de KMEANS con distancia softDTW', model2.inertia_)
    print('INERCIA B de KMEANS con distancia softDTW',  inerciaB(scaledT))
    print('INERCIA C de KMEANS con distancia softDTW',  
        sum(model2.transform(scaled.T).min(axis = 1)**2))
    print(seriesT.clus.value_counts(normalize = False))

    model3 =  clustering.TimeSeriesKMeans(n_clusters = nclus, n_init = 10, 
                metric = 'euclidean', random_state= 95).fit(scaled.T)
    clusters = seriesT
    seriesT['clus'] = model3.fit_predict(scaled.T) + 1
    scaledT['clus'] = seriesT['clus'] 
    barycenter = model3.cluster_centers_
    barycenter = np.reshape(barycenter, (barycenter.shape[0], 
                    barycenter.shape[1])) 
    barycenters = pd.DataFrame(barycenter)
    barycenters.columns = seriesT.columns[:-1]
    barycenters['clus'] = list(range(1, nclus + 1))
    barycenters.index = ['c'+str(j) for j in range(1, nclus + 1)]
    clusters = pd.concat((seriesT,barycenters), axis = 0)
    # Lo guardamos para graficar después:
    clusters.to_csv('data/processed_files/KmeansEuclidean' + str(nclus) +str(fq)+ '.csv',
                    index = True, mode = 'w', sep =';', header = True)
    print('\nINERCIA de KMEANS con distancia euclídea', model3.inertia_)
    print('INERCIA B de KMEANS con distancia euclídea',  inerciaB(scaledT))
    print('INERCIA C de KMEANS con distancia euclídea',  
        sum(model3.transform(scaled.T).min(axis = 1)**2))

    print(seriesT.clus.value_counts(normalize = False))

    # A pesar de haber ajustado mucho los parámetros para Kmeans DTW, cambian
    # mucho los clusters que se obtienen, es decir, resulta muy inestable. Vamos
    # a probar ahora con DBSCAN. No es específico para time series y default 
    # como métrica la euclídea, veamos cuantos clusters toma y como son. Probando
    # se ve que con min_samples = 5 quedan máximo 2 clusters, y también cambiamos
    # epsilon, que para hacer mas de un cluster tiene que ser > 0.5 y menor q ?:
    from sklearn.cluster import DBSCAN
    model4 = DBSCAN(eps = 0.5, min_samples = 4)
    # DBSCAN llama outliers con clase -1, en nuestro caso será 1(+2):

    seriesT['clus'] = model4.fit_predict(scaled.T) + 2
    scaledT['clus'] = seriesT['clus'] 
    ncl = len(np.unique(model4.labels_))
    print('\nDBSCAN elige clusters =', np.unique(model4.labels_))
    # CUIDADO!!! El nº de clusters es +2

    # Podemos calcular el centro de los clusters y llamamos barycenters. Por 
    #semejanza de los otros métodos, lo escalamos:
    barycenter = seriesT.groupby('clus').mean()
    barycenters = pd.DataFrame(minmax_scale(barycenter.T)).T
    barycenters.columns = seriesT.columns[:-1]
    barycenters['clus'] = barycenters.index 
    barycenter['clus'] = barycenter.index
    barycenters.index = ['c'+str(j) for j in range(1, ncl + 1)]
    clusters = pd.concat((seriesT,barycenters), axis = 0)
    # Lo guardamos para graficar después:
    clusters.to_csv('data/processed_files/DBSCANEuclidean' + str(ncl)+ str(fq) + '.csv', 
                    index = True, mode = 'w', sep =';', header = True)

    print('\nINERCIA B de DBSCAN con distancia euclídea', inerciaB(scaledT))
    # Aquí hemos sumado también las distancias al cluster 1 que en realidad no es
    # un cluster sino el grupo de los anómalos, por lo q la comparación no es
    # del todo correcta.... quizás habría que eliminar estas
    print('\nINERCIA de DBSCAN con distancia euclídea eliminando outliers', 
        inerciaB(scaledT, 1))
    # Pero comparación buena claro si hay muchos outliers...

    print(seriesT.clus.value_counts(normalize = False))

    # También se puede ejecutar DBSCAN con una distancia expresada mediante matriz
    # y podemos usar DTW:
    model5 = DBSCAN(eps = 0.5, min_samples = 4, metric = 'precomputed')
    # DBSCAN llama outliers con clase -1, en nuestro caso será 1(+2):
    seriesT['clus'] = model5.fit(DTW).labels_ + 2
    scaledT['clus'] = seriesT['clus'] 
    ncl = len(np.unique(model5.labels_))
    print('\nDBSCAN elige clusters =', np.unique(model5.labels_))
    # CUIDADO!!! El nº de clusters es + 2
    # Podemos calcular el centro de los clusters y llamamos barycenters. Por 
    #semejanza de los otros métodos, lo escalamos:
    barycenter = seriesT.groupby('clus').mean()
    barycenters = pd.DataFrame(minmax_scale(barycenter.T)).T
    barycenters.columns = seriesT.columns[:-1]
    barycenters['clus'] = barycenters.index 
    barycenters.index = ['c'+str(j) for j in range(1, ncl + 1)]
    clusters = pd.concat((seriesT,barycenters), axis = 0)
    # Lo guardamos para graficar después:
    clusters.to_csv('data/processed_files/DBSCANdtw' + str(ncl) + str(fq) + '.csv', index = True, 
                    mode = 'w', sep =';', header = True)

    print('\nINERCIA  B de DBSCAN con distancia DTW', inerciaB(scaledT))
    print('\nINERCIA B de DBSCAN con distancia DTW eliminando outliers', 
        inerciaB(scaledT, 1))


    print(seriesT.clus.value_counts(normalize = False))

