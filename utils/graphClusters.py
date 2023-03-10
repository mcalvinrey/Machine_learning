
import os
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

def graphClusters():
    os.chdir('C:/Users/mcalv/Desktop/Proyectos/Machine learning/src')

    filin = 'KmeansDTW'    # Fichero con series y barycenter (rows)
    nclus = 4    # nº de clusters
    fq = 'M'   # 'D','7D','M','Q','Y' Frec. o nivel granularidad análisis
    ngf = 2      # nº de gráficos por fila

    # Gráfico de las series agrupadas por cluster incluído su barycenter
    # Como los clusters se han construído para las series escaladas, en el gráfico
    # aparecen las series con su escala normal y el baricenter con su escala en 
    # el eje de la derecha, ya que tiene rango escalado...


    data =  pd.read_csv('data/processed_files/'+ filin + str(nclus) + str(fq) + '.csv',
                        sep = ';', header = 0)
    #sers = pd.pivot_table(data, columns = data.index)
    #sers.columns = data['Unnamed: 0']
    #sers = sers.iloc[:-1, :]
    datanobar = data.iloc[:-nclus, :]
    clusmeans = datanobar.groupby('clus').agg('mean')
    databar = data.iloc[-nclus:, :]
    datanobar.set_index('Unnamed: 0', inplace = True, drop = True)
    databar.set_index('Unnamed: 0', inplace = True, drop = True)
    fechas = pd. to_datetime(data.columns[1: -1])
        
    fig, ax1 = plt.subplots(ceil(nclus/ngf), ngf, sharex = True,
                                figsize=(15, 11))
    for ng in range(1, nclus + 1):
        ax1 = plt.subplot(ceil(nclus/ngf), ngf, ng)

        ax2 = ax1.twinx()
        ax1.set_xlim(min(fechas), max(fechas))

        ax1.spines['bottom'].set_color('white')   
        ax1.spines['top'].set_color('white') 
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.xaxis.label.set_color('navy')
        ax1.yaxis.label.set_color('navy')
        ax1.axvspan(min(fechas), max(fechas), facecolor='whitesmoke')
        ax1.xaxis.grid(linestyle = '--')
        ax1.tick_params(colors='navy', axis= 'x', size = 0.5, 
                        labelsize = 7, rotation = 30)
        ax1.tick_params(colors='navy', axis= 'y', size = 0.5, 
                        labelsize = 7, rotation = 0)
        ax2.tick_params(colors='navy', axis= 'y', size = 0.5, 
                        labelsize = 7, rotation = 0) 
        cuales = datanobar[datanobar.clus == ng]
        ax1.set_title('Cluster ' + str(ng) + ' (' + str(len(cuales)) + 
                    ' series)', fontsize = 11)
        for r in range(len(cuales)):
            ax1.plot(fechas, cuales.iloc[ r, :-1], color = 'b', lw = 0.1)
        ax1.plot(fechas, clusmeans.loc[ng], color = 'b', lw = 1)
        ax2.set_ylim(0, 1)
        ax2.plot(fechas, databar.loc['c' + str(ng)][:-1], color = 'red',
                lw = 1)
            
    for ng in range(nclus + 1, (ceil(nclus/ngf) * ngf) + 1):
        ax1 = plt.subplot(ceil(nclus/ngf), ngf, ng)
        ax2 = ax1.twinx()
        ax1.set_xlim(min(fechas), max(fechas))

        ax1.spines['bottom'].set_color('white')   
        ax1.spines['top'].set_color('white') 
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.xaxis.label.set_color('navy')
        ax1.yaxis.label.set_color('navy')
        ax1.axvspan(min(fechas), max(fechas), facecolor='whitesmoke')
        ax1.xaxis.grid(linestyle = '--')
        ax1.tick_params(colors='navy', axis= 'x', size = 0.5, 
                        labelsize = 7, rotation = 30)
        ax1.tick_params(colors='navy', axis= 'y', size = 0.5, 
                        labelsize = 7, rotation = 0)
        ax2.tick_params(colors='navy', axis= 'y', size = 0.5, 
                        labelsize = 7, rotation = 0)  
        ax2.set_ylim(0, 1)
        ax2.plot(fechas, databar.loc['c1'][:-1], 
                color = 'whitesmoke', lw = 1)
    fig.tight_layout()
    plt.savefig('data/Imagenes_aforos/'+filin + str(nclus) + str(fq) +'.jpeg', dpi = 600)
        
    #plt.show()
        