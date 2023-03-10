
import time
import clustersAforos
import graphClusters
import ReduccionDimensionAforos
import Xdireccion_clus

import Identificar_estado_trafico
import importar_fotos_M30
import ClassifyC_M30
 
t = time.time() 
clustersAforos.clustersAforos()
graphClusters.graphClusters()
ReduccionDimensionAforos.ReduccionDimensionesAforos()
Xdireccion_clus.Xdireccion_clus()

ClassifyC_M30.ClassifyC_M30()
Identificar_estado_trafico.Identificar_estado_trafico()
importar_fotos_M30.importar_fotos_M30()

print(time.time()-t)