# -*- coding: utf-8 -*-
import lectura
import numpy as np
import selection
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import statistics
import sklearn
import pandas as pd
import statsmodels.formula.api as smf

def mlp(fileCal, fileVal, type):
#def mlp(porc, file, type):
    print "Modelo MLP"
    if (type == "etapa1"):
        data = lectura.lecturaCompletaMLP_etapa1(file)
        print data
        varCal = 'SM_CONAE'
        varVal = 'SM_CONAE'
        indexC1 = 1
        indexC2 = 4
        #acti = 'relu'
        #sol = 'adam'
        #iter = 50000
        #l_rate = 0.01
        #alpa = 0.0001
        #momen = 0.9
        #random_s = 9
        #lr = 'adaptive'


        #acti = 'tanh'
        #acti = 'logistic'
        acti = 'relu'
        #acti = 'identity'

        #sol = 'adam'
        sol = 'sgd'
        #sol = 'lbfgs'

        iter = 5000

        #65
        random_s = 9
        #lr = 'constant'
        lr = 'adaptive'
        #lr = 'invscaling'

        l_rate = 0.0001
        alpa = 0.015
        momen = 0.75


    if (type == "etapa2"):

        dataCal = lectura.lecturaCompletaMLP_etapa2(fileCal)
        #print data
        dataVal = lectura.lecturaCompletaMLP_etapa2(fileVal)

        np.random.seed(0)
        dataNew = selection.shuffle(dataCal)
        dataTraining = dataNew.reset_index(drop=True)

        np.random.seed(0)
        dataNew = selection.shuffle(dataVal)
        dataTest = dataNew.reset_index(drop=True)

        varCal = 'SM_SMAP'
        #indexC1 = 8 ### lo obtengo con sigma0 a 5km con GPM y Et con bilinear, ndvi <0.8 pero con acti = 'logistic'
        #indexC2 = 0
        #### -----------------------------------------------------------------------
        #indexC1 = 5 ### lo obtengo con sigma0 a 5km con GPM con bilinear, ndvi <0.8
        #indexC2 = 5

        ### para HR
        #indexC1 = 5 ### lo obtengo con sigma0 a 5km con GPM y HR con bilinear, ndvi <0.8
        #indexC2 = 4
        #acti = 'relu'
        #sol = 'adam'
        #iter = 10000
        #l_rate = 0.0055
        #alpa = 0.0205
        #momen = 0.01
        #random_s = 5
        #lr = 'adaptive'

        #### para ET
        indexC1 = 3
        indexC2 = 6
        acti = 'relu'
        sol = 'adam'
        iter = 10000
        l_rate = 0.061
        alpa = 0.0001
        momen = 0.1
        random_s = 1
        lr = 'adaptive'



    if (type == "etapa3"):
        data = lectura.lecturaCompletaMLP_etapa3(file)
        varCal = 'SMAP'
        varVal = 'SM_CONAE'
        indexC1 = 4
        indexC2 =3



    if (type == "etapa1"):

        np.random.seed(0)
        dataNew = selection.shuffle(data)
        dataNew = dataNew.reset_index(drop=True)
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

        yTraining = dataTraining[varCal]
        del dataTraining[varCal]
        xTraining = dataTraining

        yTest_SMAP = dataTest['SM_CONAE']
        del dataTest[varCal]
        test_x = dataTest

    if (type == "etapa2"):

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

        yTraining = dataTraining[varCal]
        del dataTraining[varCal]
        xTraining = dataTraining
        #print "datos de calibracion"
        #print xTraining

        yTest_SMAP = dataTest['SM_SMAP']
        del dataTest[varCal]
        test_x = dataTest
        #print "datos de validacion"
        #print test_x

    #print "---------------------------------------------------------------------"

    print "indice capa 1: " + str(indexC1)
    print "indice capa 2: " + str(indexC2)
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
    print "AQUIIIII: " +str(yCal.shape)

    print "------------------------------------------------------------------------"
    print "MLP Calibracion: "
    print "calibracion SMAP vs SMAP"
    rmse = statistics.RMSE(np.array(yTraining),np.array(yCal))
    print "RMSE:" + str(rmse)
    RR = sklearn.metrics.r2_score(yCal, yTraining)
    print "Coeficiente de Determinacion:" + str(RR)
    data2 = pd.DataFrame({'yCal' :yCal,'yTraining' :yTraining})
    RR = smf.ols('yCal ~ 1+ yTraining', data2).fit().rsquared
    print "R^2 222: "+str(RR)



    bias = statistics.bias(yTraining, yCal)
    print "Bias:" + str(bias)
    #print "calibracion SMAP vs CONAE"
    #rmse = statistics.RMSE(np.array(yTraining_CONAE),np.array(yCal))
    #print "RMSE:" + str(rmse)
    #RR = sklearn.metrics.r2_score(yTraining_CONAE, yCal)
    #print "Coeficiente de Determinacion:" + str(RR)
    #bias = statistics.bias(yTraining_CONAE, yCal)
    #print "Bias:" + str(bias)

    yAprox = reg.predict(test_x)

    print "------------------------------------------------------------------------"
    print "MLP Validacion: "
    print "calibracion con SMAP/ validacion con SMAP"
    rmse = statistics.RMSE(np.array(yTest_SMAP),np.array(yAprox))
    print "RMSE:" + str(rmse)
    RR = sklearn.metrics.r2_score(yTest_SMAP, yAprox)
    #RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data).fit().rsquared
    print "Coeficiente de Determinacion:" + str(RR)
    data2 = pd.DataFrame({'yTest_SMAP' :yTest_SMAP,'yAprox' :yAprox})
    RR = smf.ols('yTest_SMAP ~ 1+ yAprox', data2).fit().rsquared
    print "R^2 222: "+str(RR)
    bias = statistics.bias(yTest_SMAP, yAprox)
    print "Bias:" + str(bias)


    print "pearson"
    print np.corrcoef(yAprox,yTest_SMAP)[1,0]


    #v1 = yTest
    #v2 = yAprox

    #z = np.polyfit(v1,v2, 1)
    #g = np.poly1d(z)
    #cor = np.corrcoef(v1,v2)[0,1]
    #if (cor >0 ):
        #cor=(cor)*(cor)
    #else:
        #cor=(cor*(-1))*(cor*(-1))

    #fig = plt.figure(1,facecolor="white")
    #ax1 = fig.add_subplot(111,aspect='equal')
    #ax1.plot(v1,g(v1),'black')
    #ax1.text(10, 30, 'R^2=%5.3f' % RR, fontsize=12)
    #ax1.text(10, 28, 'r^2=%5.3f' % cor, fontsize=12)
    #ax1.set_xlabel("observed value [% GSM]",fontsize=12)
    #ax1.set_ylabel("estimated value [% GSM]",fontsize=12)
    ##ax1.set_xlabel("valor observado [% GSM]",fontsize=12)
    ##ax1.set_ylabel("valor estimado [% GSM]",fontsize=12)
    ##ax1.scatter(x, y, s=10, c='b', marker="s", label='real')
    #ax1.scatter(yTest,yAprox, s=10,color='black',linewidth=3)# c='r', marker="o", label='NN Prediction')
    #ax1.axis([5,45, 5,45])
    #plt.grid(True)

    #xx = np.linspace(0,len(yTest),len(yTest))


    #fig = plt.figure(2,facecolor="white")
    #fig0 = fig.add_subplot(111)
    #fig0.text(0, 15, 'RMSE=%5.3f' % rmse, fontsize=12)


    #fig0.scatter(xx, yTest, color='blue',linewidth=3,label='SM')
    #fig0.scatter(xx, yAprox,s=65, color='black',marker = "*",label='SM_Aprox')
    #fig0.legend(loc=1, fontsize = 'medium')
    ##fig0.set_xlabel("Samples",fontsize=12)
    ##fig0.set_ylabel("Soil moisture [% GSM]",fontsize=12)
    #fig0.set_xlabel("muestras",fontsize=12)
    #fig0.set_ylabel("humedad de suelo [% GSM]",fontsize=12)
    #plt.show()

    return reg, yCal, yAprox