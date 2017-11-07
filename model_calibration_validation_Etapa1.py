import pandas as pd
import statsmodels.formula.api as smf
import selection
import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura
import application
import MLPregression
import copy
import SMmaps


file = "tabla_calibration_validation.csv"

data = lectura.lecturaCompleta_etapa1(file)
#print data

## se mezclan las observaciones de las tablas
## semilla para mezclar los datos en forma aleatoria


np.random.seed(0)
dataNew = selection.shuffle(data)
dataNew = dataNew.reset_index(drop=True)

## formula

formula = "SM_CONAE ~ 1+T_aire+HR+PP"

print "Modelo planteado: " + str(formula)
model2 = smf.ols(formula, dataNew).fit()
print "R^2 del modelo: " + str(model2.rsquared)


####### obtencion automatica de los porcentajes de datos
###### para entrenar y probar

print "Obtencion automatica de los porcentajes de datos"
type = "RMSE"
#type = "R^2"
var = "RSOILMOIST"
porc = 0.75
print "Porcentaje de datos de calculo: " + str(porc)


### division de los datos para entrenamiento y prueba
nRow = len(dataNew.index)
numTraining=int(round(nRow)*porc)
print "Cantidad de elementos para el calculo de coeff: " + str(numTraining)
numTest=int((nRow)-numTraining)
print "Cantidad de elementos para prueba: " +str(numTest)


dataTraining =  dataNew.ix[:numTraining, :]
dataTraining = selection.shuffle(dataTraining)
dataTraining = dataTraining.reset_index(drop=True)
#print dataTraining

dataTest = dataNew.ix[numTraining + 1:, :]
#print dataTest
#lectura.graph(dataTest)

#### Calibracion
print "Calibracion: "
MLRmodel = smf.ols(formula, dataTraining).fit()
print MLRmodel.summary()
print "R^2 del modelo: " + str(MLRmodel.rsquared)


#### error de calibracion
xxx = copy.copy(dataTraining)
del xxx['SM_CONAE']
yTraining = dataTraining['SM_CONAE']
yCal = MLRmodel.predict(xxx)
yTraining = 10**(yTraining)
yCal = 10**(yCal)

rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
print "RMSE:" + str(rmse)
bias = statistics.bias(yTraining,yCal)
print "Bias:" + str(bias)


#print "RMSE del modelo: " + str(np.sqrt(model.mse_resid))

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

print "Validacion: "

y = np.array(dataTest["SM_CONAE"])

pred = MLRmodel.predict(dataTest)
yAprox = np.array(pred)
#### aca!!!
#y = np.exp(y)
#yAprox = np.exp(yAprox)

y = 10**(y)
yAprox = 10**(yAprox)

bias = statistics.bias(yAprox,y)
print "Bias Validacion:" + str(bias)

print "Rango real: "+ str(np.max(y))+"..." + str(np.min(y))
print "Rango aproximado: "+ str(np.max(yAprox))+"..." + str(np.min(yAprox))


## se obtiene el error
rmse = 0
rmse = statistics.RMSE(y,yAprox)
print "RMSE:" + str(rmse)
RR = sklearn.metrics.r2_score(y, yAprox)
print "R2:" + str(RR)
error = np.zeros((len(y),1))
#for i in range(0,len(error)):
error = np.abs(y-yAprox)


RR = smf.ols('y ~ 1+ yAprox', dataTest).fit().rsquared
print "R^2 222: "+str(RR)


xx = np.linspace(0,len(y),len(y))
MLPmodel, yCalMLP, yAproxMLP = MLPregression.mlp(porc,file, "etapa1")


v1 = []
v2 = []
v3 = []
for i in range(len(y)):
    v1.append(float(yTraining[i]))
    v2.append(float(yCal[i]))
    v3.append(float(yCalMLP[i]))

fig = plt.figure(1,facecolor="white")
fig1 = fig.add_subplot(111,aspect='equal')
#fig1, ax = plt.subplots()

#fig1.set_title('Humedad Vs Humedad Aprox')
fig1.scatter(yTraining ,yCal, s=10, color='black',linewidth=3, label='MLR')
fig1.scatter(yTraining,yCalMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')
yt =np.array(yTraining)
z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'black')
z = np.polyfit(v1,v3, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'green')
fig1.set_xlabel("observed value [% GSM]",fontsize=12)
fig1.set_ylabel("estimated value [% GSM]",fontsize=12)



fig1.plot(yt, yt)
fig1.legend(loc=4, fontsize = 'medium')
fig1.axis([5,45, 5,45])
plt.grid(True)




fig = plt.figure(2,facecolor="white")
fig1 = fig.add_subplot(111,aspect='equal')
#fig1, ax = plt.subplots()

#fig1.set_title('Humedad Vs Humedad Aprox')
fig1.scatter(y,yAprox, s=10, color='black',linewidth=3, label='MLR')

fig1.scatter(y,yAproxMLP, s=10, marker="^", color="green", linewidth=3, label='MLP')


v1 = []
v2 = []
v3 = []
for i in range(len(y)):
    v1.append(float(y[i]))
    v2.append(float(yAprox[i]))
    v3.append(float(yAproxMLP[i]))


z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'black')
z = np.polyfit(v1,v3, 1)
g = np.poly1d(z)
fig1.plot(v1,g(v1),'green')

#cor = np.corrcoef(v1,v2)[0,1]
#if (cor >0 ):
    #cor=(cor)*(cor)
#else:
    #cor=(cor*(-1))*(cor*(-1))
fig1.plot(y, y)
#fig1.text(np.min(v1), np.max(v2), 'r^2=%5.3f' % cor, fontsize=15)

#fig1.text(16, 35, 'R^2=%5.3f' % RR, fontsize=12)
#fig1.text(16, 32, 'r^2=%5.3f' % cor, fontsize=12)
fig1.set_xlabel("observed value [% GSM]",fontsize=12)
fig1.set_ylabel("estimated value [% GSM]",fontsize=12)
fig1.legend(loc=4, fontsize = 'medium')
fig1.axis([5,45, 5,45])
plt.grid(True)


#print "Correlacion de Pearson"+":" + str(cor)
#print "R^2 de la prueba"+":" + str(RR)

plt.show()

#print "Aplication: "
#file = "tabla_aplication.csv"
#file = "tabla_calibration_validation.csv"
#application.application(file, model, MLPmodel, "etapa1")

# Se obtienen los mapas de HS con los modelos calibrados
#SMmaps.calculateMaps(MLRmodel, MLPmodel, "etapa1")
