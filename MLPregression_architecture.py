# -*- coding: utf-8 -*-
import lectura
import numpy as np
import selection
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import statistics
import sklearn
import sys
from sklearn.preprocessing import StandardScaler



### etapa 1
#etapa = 1
#file = "tabla_calibration_validation.csv"
#data = lectura.lecturaCompletaMLP_etapa1(file)




### etapa 2

#file = "tabla_calibration_validation_conSMSMAP.csv"

#file = "tabla_completa_sigma1km.csv"
#file = "tabla_completa_sigma5km.csv"

#file = "tabla_completa_sigma1km_GPM.csv"
#file = "tabla_completa_sigma5km_GPM_Et.csv"
#file = "tabla_completa_sigma1km_GPM_Et.csv"

etapa = 2
#file = "tabla_completa_sigma5km_GPM_HR_Et.csv"

#data = lectura.lecturaCompletaMLP_etapa2(file)

fileCal = 'tabla_completa_Calibracion.csv'
fileVal = 'tabla_completa_Validacion.csv'

dataCal = lectura.lecturaCompletaMLP_etapa2(fileCal)
#print data
dataVal = lectura.lecturaCompletaMLP_etapa2(fileVal)



### etapa 3
#file = "tabla_calibration_validation_conSMSMAP_GPM.csv"
#data = lectura.lecturaCompletaMLP_etapa3(file)
#print data
print "Modelo MLP"


#np.random.seed(0)
#dataNew = selection.shuffle(data)
#dataNew = dataNew.reset_index(drop=True)

#porc = 0.75
#print "Porcentaje de datos de calculo: " + str(porc)


### division de los datos para entrenamiento y prueba
#nRow = len(dataNew.index)
#numTraining=int(round(nRow)*porc)
#print "Cantidad de elementos para calibrar: " + str(numTraining)
#numTest=int((nRow)-numTraining)
#print "Cantidad de elementos para validar: " +str(numTest)

#dataTraining =  dataNew.ix[:numTraining, :]
#dataTraining = selection.shuffle(dataTraining)
#dataTraining = dataTraining.reset_index(drop=True)

#### test
#dataTest = dataNew.ix[numTraining + 1:, :]
#dataTest = dataTest.reset_index(drop=True)


np.random.seed(0)
dataNew = selection.shuffle(dataCal)
dataTraining = dataNew.reset_index(drop=True)

np.random.seed(0)
dataNew = selection.shuffle(dataVal)
dataTest = dataNew.reset_index(drop=True)


if (etapa == 1):
    OldRange = (np.max(dataTraining.T_aire)  - np.min(dataTraining.T_aire))
    NewRange = (1 + 1)
    dataTraining.T_aire = (((dataTraining.T_aire - np.min(dataTraining.T_aire)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
    NewRange = (1 + 1)
    dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
    NewRange = (1 + 1)
    dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
    NewRange = (1 + 1)
    dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange) -1

    #print dataTraining.describe()



    OldRange = (np.max(dataTest.T_aire)  - np.min(dataTest.T_aire))
    NewRange = (1 + 1)
    dataTest.T_aire = (((dataTest.T_aire - np.min(dataTest.T_aire)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
    NewRange = (1 + 1)
    dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
    NewRange = (1 + 1)
    dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
    NewRange = (1 + 1)
    dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange) -1

    yTraining = dataTraining['SM_CONAE']
    del dataTraining['SM_CONAE']
    xTraining = dataTraining
    print dataTraining

    yTest = dataTest['SM_CONAE']
    del dataTest['SM_CONAE']
    test_x = dataTest



if (etapa == 2):

    OldRange = (np.max(dataTraining.T_s)  - np.min(dataTraining.T_s))
    NewRange = (1 + 1)
    dataTraining.T_s = (((dataTraining.T_s - np.min(dataTraining.T_s)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.Et)  - np.min(dataTraining.Et))
    NewRange = (1 + 1)
    dataTraining.Et = (((dataTraining.Et - np.min(dataTraining.Et)) * NewRange) / OldRange) -1


    #OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
    #NewRange = (1 + 1)
    #dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
    NewRange = (1 + 1)
    dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
    NewRange = (1 + 1)
    dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange) -1



    OldRange = (np.max(dataTest.T_s)  - np.min(dataTest.T_s))
    NewRange = (1 + 1)
    dataTest.T_s = (((dataTest.T_s - np.min(dataTest.T_s)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTest.Et)  - np.min(dataTest.Et))
    NewRange = (1 + 1)
    dataTest.Et = (((dataTest.Et - np.min(dataTest.Et)) * NewRange) / OldRange) -1

    #OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
    #NewRange = (1 + 1)
    #dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange) -1


    OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
    NewRange = (1 + 1)
    dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange) -1

    OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
    NewRange = (1 + 1)
    dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange) -1


    yTraining = dataTraining['SM_SMAP']
    del dataTraining['SM_SMAP']
    xTraining = dataTraining
    print dataTraining

    yTest = dataTest['SM_SMAP']
    del dataTest['SM_SMAP']
    test_x = dataTest




##yTraining = dataTraining['RSOILMOIST']
##del dataTraining['RSOILMOIST']
##xTraining = dataTraining

##yTest = dataTest['RSOILMOIST']
##del dataTest['RSOILMOIST']
##test_x = dataTest



#print dataTest



#var = "RR"
#var = "rmse"
###----------------------------------------------------------------------------

##acti = 'tanh'
#acti = 'logistic'
#acti = 'relu'
##acti = 'identity'

#sol = 'adam'
#sol = 'sgd'
##sol = 'lbfgs'

#iter = 5000

##65
#random_s = 9
#lr = 'constant'
#lr = 'adaptive'
##lr = 'invscaling'

#l_rate = 0.0001
#alpa = 0.0015
#momen = 0.75

#### HR
#acti = 'relu'
#sol = 'adam'
#iter = 5000
#l_rate = 0.0055
#alpa = 0.0205
#momen = 0.99
#random_s = 5
#lr = 'adaptive'


## ET
acti = 'relu'
sol = 'adam'
iter = 10000
l_rate = 0.061
alpa = 0.0001
momen = 0.1
random_s = 1
lr = 'adaptive'




lrate = np.arange(0.001, 0.5,0.05)
print lrate

alpal = np.arange(0.0001, 0.01,0.05)
print alpal

momenl = np.arange(0.1, 0.99,0.05)
#print momenl

menorRMSE = 999
errorBest= 0
indexC1 = 1
indexC2 = 0
#for kk in range(0, len(lrate)):
    #l_rate= lrate[kk]
    #print "lrate: " +str(l_rate)

#for kk in range(0, len(alpal)):
    #alpa= alpal[kk]
    #print "alpa: " +str(alpa)


#for kk in range(0, len(momenl)):
    #momen= momenl[kk]
    #print "momen: " +str(momen)


bandera = False
for j in range(1,7):
    for k in range(0,7):
        ## MultiLayer Perceptron
        if (k == 0):
            reg = MLPRegressor(hidden_layer_sizes=(j), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
                       learning_rate=lr, learning_rate_init=l_rate, power_t=0.5, max_iter=iter, shuffle=True,
                       random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.3, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
        else:
            reg = MLPRegressor(hidden_layer_sizes=(j,k), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
                       learning_rate=lr, learning_rate_init=l_rate, power_t=0.5, max_iter=iter, shuffle=True,
                       random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.3, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
        print k
        print j
        reg = reg.fit(xTraining, yTraining)

        yAprox = reg.predict(test_x)

        #print yTest
        rmse = statistics.RMSE(np.array(yTest),np.array(yAprox))
        #print "RMSE:" + str(rmse)
        RR = sklearn.metrics.r2_score(yTest, yAprox)
        #print "Coeficiente de Determinacion:" + str(RR)
        if (rmse < menorRMSE):
            print rmse
            indexC1 = j
            indexC2 = k
            menorRMSE = rmse
            mayorRR = RR
            errorBest = rmse
            ll = l_rate
            ap = alpa
            mm = momen


print "capa oculta 1 nro de neuronas: " +str(indexC1)
print "capa oculta 2 nro de neuronas: " +str(indexC2)
print "mayor coeficiente de determinacion: "+ str(mayorRR)
print "menor RMSE: " + str(errorBest)
print "l_rate: " + str(ll)
print "alpha: " + str(ap)
print "momento: " + str(mm)

print "---------------------------------------------------------------------"

l_rate = ll
alpa = ap
momen = mm


#print "indice capa 1: " + str(indexC1)
#print "indice capa 2: " + str(indexC2)
if (indexC2 == 0):
    reg = MLPRegressor(hidden_layer_sizes=(indexC1), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
               learning_rate=lr, learning_rate_init=l_rate, power_t=0.5, max_iter=iter, shuffle=True,
               random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.3, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)
else:
    reg = MLPRegressor(hidden_layer_sizes=(indexC1,indexC2), activation= acti, solver= sol, alpha=alpa,batch_size='auto',
           learning_rate=lr, learning_rate_init=l_rate, power_t=0.5, max_iter=iter, shuffle=True,
           random_state=random_s, tol=0.0001, verbose=False, warm_start=False, momentum=momen,
           nesterovs_momentum=True, early_stopping=False, validation_fraction=0.3, beta_1=0.9, beta_2=0.999,
           epsilon=1e-08)

reg = reg.fit(xTraining, yTraining)


yCal = reg.predict(xTraining)


print "MLP Calibracion: "
rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
print "RMSE:" + str(rmse)
RR = sklearn.metrics.r2_score(yTraining, yCal)
print "Coeficiente de Determinacion:" + str(RR)
bias = statistics.bias(yTraining, yCal)
print "Bias:" + str(bias)


yAprox = reg.predict(test_x)

print "MLP Validacion: "
rmse = statistics.RMSE(np.array(yTest),np.array(yAprox))
print "RMSE:" + str(rmse)
RR = sklearn.metrics.r2_score(yTest, yAprox)
print "Coeficiente de Determinacion:" + str(RR)
bias = statistics.bias(yTest, yAprox)
print "Bias:" + str(bias)




v1 = yTest
v2 = yAprox

z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
cor = np.corrcoef(v1,v2)[0,1]
if (cor >0 ):
    cor=(cor)*(cor)
else:
    cor=(cor*(-1))*(cor*(-1))

fig = plt.figure(1,facecolor="white")
ax1 = fig.add_subplot(111,aspect='equal')
ax1.plot(v1,g(v1),'black')
ax1.text(10, 30, 'R^2=%5.3f' % RR, fontsize=12)
ax1.text(10, 28, 'r^2=%5.3f' % cor, fontsize=12)
ax1.set_xlabel("observed value [% GSM]",fontsize=12)
ax1.set_ylabel("estimated value [% Vol.]",fontsize=12)
#ax1.set_xlabel("valor observado [% GSM]",fontsize=12)
#ax1.set_ylabel("valor estimado [% GSM]",fontsize=12)
#ax1.scatter(x, y, s=10, c='b', marker="s", label='real')
ax1.scatter(yTest,yAprox, s=10,color='black',linewidth=3)# c='r', marker="o", label='NN Prediction')
ax1.axis([5,45, 5,45])
plt.grid(True)

xx = np.linspace(0,len(yTest),len(yTest))


fig = plt.figure(2,facecolor="white")
fig0 = fig.add_subplot(111)
fig0.text(0, 15, 'RMSE=%5.3f' % rmse, fontsize=12)


fig0.scatter(xx, yTest, color='blue',linewidth=3,label='SM')
fig0.scatter(xx, yAprox,s=65, color='black',marker = "*",label='SM_Aprox')
fig0.legend(loc=1, fontsize = 'medium')
#fig0.set_xlabel("Samples",fontsize=12)
#fig0.set_ylabel("Soil moisture [% GSM]",fontsize=12)
fig0.set_xlabel("muestras",fontsize=12)
fig0.set_ylabel("humedad de suelo [% Vol.]",fontsize=12)
#
plt.show()

#