import pandas as pd
import selection
import numpy as np
import statistics
import lectura
import matplotlib.pyplot as plt



#file = "tabla_Completa.csv"
file = "tabla_calibration_validation.csv"

#file = "tabla_completa_sigma1km.csv"
#file = "tabla_completa_sigma5km.csv"


#file = "tabla_completa_sigma1km_GPM.csv"
#file = "tabla_completa_sigma5km_GPM.csv"
#file = "tabla_completa_sigma5km_GPM_Et.csv"
#file = "tabla_completa_sigma1km_GPM_Et.csv"


#file = "tabla_completa_sigma5km_GPM_Et_HR.csv"
## nueva tabla que posee el 70 % de los datos con GPM_Et_HR



####---------------------------------------------------------------------------
#file = "tabla_completa_sigma5km_GPM_HR_Et.csv"

#file = "tabla_training_set.csv"
data = lectura.lecturaCompleta_etapa1(file)
#data = lectura.lecturaCompleta_etapa2(file)

#print data


# se mezclan las observaciones de las tablas
# semilla para mezclar los datos en forma aleatoria
np.random.seed(0)
dataNew = selection.shuffle(data)
dataNew = data.reset_index(drop=True)
#print dataNew
#dataNew = data

statistics.stats(dataNew,'SM_CONAE')
#statistics.stats(dataNew,'SM_SMAP')


#statistics.stats(dataNew,'SMAP')

matrix = np.array(dataNew)
print "Orden de las variables"
print list(dataNew)
print statistics.calc_vif(matrix)


