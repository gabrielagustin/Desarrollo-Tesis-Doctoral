import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as aca
import selection
import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura
import application
import copy
import MLPregression
import SMmaps


#-----------------------------------------------------------------------------
# Etapa 2
# Se calibra el modelo con HS-SMAP a 36 Km
# se valida con HS-CONAE
#-----------------------------------------------------------------------------

#file = "tabla_calibration_validation_conSMSMAP.csv"

#file = "tabla_completa_sigma1km.csv"
#file = "tabla_completa_sigma5km.csv"

#file = "tabla_completa_sigma1km_GPM.csv"
#file = "tabla_completa_sigma5km_GPM.csv"

#file = "tcsv"
#file = "tabla_completa_sigma1km_GPM_Et.csv"

####-----------------------------------------------------------------
file = 'tabla_Completa.csv'

fileCal = 'tabla_completa_Calibracion.csv'
fileVal = 'tabla_completa_Validacion.csv'

data = lectura.lecturaCompleta_etapa2(file)


dataCal = lectura.lecturaCompleta_etapa2(fileCal)
dataVal = lectura.lecturaCompleta_etapa2(fileVal)

# se mezclan las observaciones de las tablas
# semilla para mezclar los datos en forma aleatoria


np.random.seed(0)
dataNew = selection.shuffle(dataCal)
dataTraining = dataNew.reset_index(drop=True)

np.random.seed(0)
dataNew = selection.shuffle(dataVal)
dataTest = dataNew.reset_index(drop=True)

#dataTest = dataVal

#frames = [dataTraining, dataTest]

#result = pd.concat(frames)


## formula

#print result

#formula = "SM_SMAP ~ 1+T_s+PP+Sigma0" # caso 1
#formula = "SM_SMAP ~ 1+HR+T_s+PP+Sigma0" # caso 2
formula = "SM_SMAP ~ 1+T_s+Et+PP+Sigma0" # caso3

print "Modelo planteado: " + str(formula)
model2 = smf.ols(formula, data).fit()
print "R^2 del modelo: " + str(model2.rsquared)


####### obtencion automatica de los porcentajes de datos
###### para entrenar y probar

#print "Obtencion automatica de los porcentajes de datos"
#type = "RMSE"
#type = "R^2"
#var = "SMAP"
#minimo, porc = selection.calc_best_porcentaje(dataNew, formula, type, var)
#print "Porcentaje de prueba calculado auto: " + str(porc)
#print "Error para ese porcentaje: " + str(minimo)
#porc = 0.70
#print "Porcentaje de datos de calculo: " + str(porc)


#### division de los datos para entrenamiento y prueba
#nRow = len(dataNew.index)
#numTraining=int(round(nRow)*porc)
#print "Cantidad de elementos para el calculo de coeff: " + str(numTraining)
#numTest=int((nRow)-numTraining)
#print "Cantidad de elementos para prueba: " +str(numTest)


#dataTraining =  dataNew.ix[:numTraining, :]
#dataTraining = selection.shuffle(dataTraining)
#dataTraining = dataTraining.reset_index(drop=True)
##print dataTraining

#dataTest = dataNew.ix[numTraining + 1:, :]
##print dataTest
##lectura.graph(dataTest)

#dataTraining





#### calibracion
print "------------------------------------------------------------------------"
print "Calibracion: "
MLRmodel = smf.ols(formula, dataTraining).fit()
print MLRmodel.summary()



#### error de calibracion
xxx = copy.copy(dataTraining)
del xxx['SM_SMAP']
yTraining_SMAP = np.array(dataTraining['SM_SMAP'])
yTraining_SMAP = 10**yTraining_SMAP
#print dataTraining
yCalMLR = MLRmodel.predict(xxx)
#yTraining = 10**(yTraining)
yCalMLR = 10**(yCalMLR)
#print yCal

print "calibracion con SMAP Vs SMAP"
rmse = statistics.RMSE(np.array(yTraining_SMAP),np.array(yCalMLR))
print "RMSE:" + str(rmse)
bias = statistics.bias(yTraining_SMAP,yCalMLR)
print "Bias:" + str(bias)
RR = MLRmodel.rsquared
print "R^2: " + str(RR)

#print "calibracion con SMAP Vs CONAE"
#rmse = statistics.RMSE(np.array(yTraining_CONAE),np.array(yCal))
#print "RMSE:" + str(rmse)
#bias = statistics.bias(yTraining_CONAE,yCal)
#print "Bias:" + str(bias)
#RR = sklearn.metrics.r2_score(yTraining_CONAE, yCal)
#print "R^2:" + str(RR)



#### se guardan los coeficientes del modelo entrenado
print "Los coeficientes del modelo son: "
coeff =  MLRmodel.params
#print coeff[1]

print "Calculo de los VIF: "
print "Orden de las variables"
print list(dataNew)
matrix = np.array(dataTraining)
vifs = statistics.calc_vif(matrix)
print vifs
#vifs = aca.variance_inflation_factor(matrix,2)
#print vifs
#### prueba del modelo





### -------------------------------------------------------------------------
print "------------------------------------------------------------------------"
print "Validacion: "

yTest_SMAP = np.array(dataTest["SM_SMAP"])
yTest_SMAP = 10**yTest_SMAP
del dataTest['SM_SMAP']


#y = np.array(dataTest["SMAP"])

pred = MLRmodel.predict(dataTest)
yAproxMLR = np.array(pred)

yAproxMLR = 10**(yAproxMLR)


#print "Rango real: "+ str(np.max(y))+"..." + str(np.min(y))
#print "Rango aproximado: "+ str(np.max(yAprox))+"..." + str(np.min(yAprox))

## se obtiene el error
rmse = 0
rmse = statistics.RMSE(yAproxMLR,yTest_SMAP)
print "RMSE SMAP MLR:" + str(rmse)

print "pearson"
print np.corrcoef(yAproxMLR,yTest_SMAP)[1,0]

bias = statistics.bias(yAproxMLR,yTest_SMAP)
print "Bias prueba SMAP MLR:" + str(bias)

RR = sklearn.metrics.r2_score(yTest_SMAP, yAproxMLR)
print "R^2 SMAP MLR:" + str(RR)

data2 = pd.DataFrame({'yTest_SMAP' :yTest_SMAP,'yAproxMLR' :yAproxMLR})
RR = smf.ols('yTest_SMAP ~ 1+ yAproxMLR', data2).fit().rsquared
print "R^2 222: "+str(RR)





print "------------------------------------------------------------------------"
MLPmodel, yCalMLP, yAproxMLP = MLPregression.mlp(fileCal,fileVal, "etapa2")
print yCalMLP.shape
print yAproxMLP.shape
print yTraining_SMAP.shape


#v1 = []
#v2 = []
#v3 = []
#v4 = []
#v5 = []
#for i in range(len(yTraining_SMAP)):
    #v1.append(float(yTraining_SMAP[i]))
    #v2.append(float(yCalMLR[i]))
    #v3.append(float(yCalMLP[i]))

#v1 = np.array(v1)
#v2 = np.array(v2)
#v3 = np.array(v3)
#application.taylorGraph(v1, v2, v3)




fig = plt.figure(1,facecolor="white")
fig1 = fig.add_subplot(111,aspect='equal')
#fig1.set_title('Calibration')
fig1.scatter(yTraining_SMAP,yCalMLR, s=10, color='blue',linewidth=3, label='MLR')
fig1.scatter(yTraining_SMAP,yCalMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')

print  yTraining_SMAP.shape
print yCalMLP.shape

v1 = []
v2 = []
v3 = []
for i in range(len(yTraining_SMAP)):
    v1.append(float(yTraining_SMAP[i]))
    v2.append(float(yCalMLR[i]))
    v3.append(float(yCalMLP[i]))


z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'blue')
z = np.polyfit(v1,v3, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'green')


fig1.set_xlabel("SMAP SM [% Vol.]",fontsize=12)
fig1.set_ylabel("estimated SM [% Vol.]",fontsize=12)
fig1.legend(loc=4, fontsize = 'medium')
fig1.axis([9,55, 9,55])
x = np.linspace(*fig1.get_xlim())
fig1.plot(x, x, linestyle="--", color='black')

plt.grid(True)




#fig = plt.figure(2,facecolor="white")
#fig1 = fig.add_subplot(111,aspect='equal')
#fig1.set_title('Cal con SMAP Vs CONAE')
#fig1.scatter(yTraining_CONAE,yCal, s=10, color='black',linewidth=3, label='MLR')
#fig1.scatter(yTraining_CONAE,yCalMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')
#v1 = []
#v2 = []
#v3 = []
#for i in range(len(yTraining_CONAE)):
    #v1.append(float(yTraining_CONAE[i]))
    #v2.append(float(yCal[i]))
    #v3.append(float(yCalMLP[i]))


#z = np.polyfit(v1,v2, 1)
#g = np.poly1d(z)
#fig1.plot(v1,g(v1),'black')
#z = np.polyfit(v1,v3, 1)
#g = np.poly1d(z)
#fig1.plot(v1,g(v1),'green')


#fig1.plot(yTraining_CONAE, yTraining_CONAE)

fig1.set_xlabel("observed value [% GSM]",fontsize=12)
fig1.set_ylabel("estimated value [% GSM]",fontsize=12)
fig1.legend(loc=4, fontsize = 'medium')
fig1.axis([10,55, 10,55])
plt.grid(True)


fig = plt.figure(3,facecolor="white")
fig1 = fig.add_subplot(111,aspect='equal')
#fig1.set_title('Validation')
fig1.scatter(yTest_SMAP,yAproxMLR, s=10, color='blue',linewidth=3, label='MLR')
fig1.scatter(yTest_SMAP,yAproxMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')


v1 = []
v2 = []
v3 = []
for i in range(len(yTest_SMAP)):
    v1.append(float(yTest_SMAP[i]))
    v2.append(float(yAproxMLR[i]))
    v3.append(float(yAproxMLP[i]))


z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'blue')
z = np.polyfit(v1,v3, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'green')

fig1.set_xlabel("SMAP SM [% Vol.]",fontsize=12)
fig1.set_ylabel("estimated SM [% Vol.]",fontsize=12)
#fig1.set_xlabel("valor observado [% GSM]",fontsize=12)
#fig1.set_ylabel("valor estimado [% GSM]",fontsize=12)
#fig1.plot([5, 45], [5, 45], ls="--", c=".3")
fig1.legend(loc=4, fontsize = 'medium')
fig1.axis([9,55, 9,55])
x = np.linspace(*fig1.get_xlim())
fig1.plot(x, x, linestyle="--", color='black')
plt.grid(True)


####

fig = plt.figure(4,facecolor="white")
fig1 = fig.add_subplot(111)
#fig1.set_title('Validation')

#print dataVal
fig1.scatter(yAproxMLR,dataVal.PP, s=10, color='blue',linewidth=3, label='MLR')
fig1.scatter(yAproxMLP,dataVal.PP, s=10, marker="^", color="green", linewidth=3, label='MLP')
fig1.set_xlabel("valor estimado [% Vol.]",fontsize=12)
fig1.set_ylabel("NDVI",fontsize=12)



#### grafico de taylor
v1 = []
v2 = []
v3 = []
for i in range(len(yTest_SMAP)):
    v1.append(float(yTest_SMAP[i]))
    v2.append(float(yAproxMLR[i]))
    v3.append(float(yAproxMLP[i]))


v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)
application.taylorGraph(v1, v2, v3)

SMmaps.calculateMaps(MLRmodel, MLPmodel, "etapa2")

plt.show()

#