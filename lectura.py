import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import selection

from sklearn.preprocessing import normalize
from sklearn import preprocessing

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


import functools
def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


def func(x, a, b,):
    return a*np.exp(-b*x)




def lecturaCompleta_etapa1(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_CONAE = data.SM_CONAE *100
    del data['NDVI_30m_B']
    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.SM_CONAE > perc10SM) & (data.SM_CONAE <= perc90SM)]
    data = dataNew


    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))





    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    #print "Filtro por estaciones malas"
    #print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.50)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    #statistics(data)
    data.SM_CONAE = np.log10(data.SM_CONAE)
    data.RSOILTEMPC = np.log10(data.RSOILTEMPC)
    data.Tension_va = np.log10(data.Tension_va)
    data.T_aire = np.log10(data.T_aire)
    data.HR = np.log10(data.HR)

    del data['RSOILTEMPC']
    del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    #del data['PP']


    del data['FECHA_HORA']
    del data['ID_DISPOSI']
    #del data['SM10Km_PCA']
    #del data['SMAP']
    del data['NDVI']

    #statistics(data)
    #graph2(data)
    #data.to_csv('/home/gag/Desktop/salida.csv')

    print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data

####----------------------------------------------------------------------------
    #print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    #perc5Ta = math.ceil(np.percentile(data.T_a,0))
    #print "percentile Ta 5: " + str(perc5Ta)
    #perc90Ta = math.ceil(np.percentile(data.T_a, 95))
    #print "percentile Ta 95: " + str(perc90Ta)
    #dataNew = data[(data.T_a > perc5Ta) & (data.T_a < perc90Ta)]
    #data = dataNew

    #print "Numero de muestras: " + str(len(data))


    #### se filtra el rango de valores de Tension_va
    #perc5Ea = math.ceil(np.percentile(data.e_a,0))
    #print "percentile Ea 5: " + str(perc5Ea)
    #perc90Ea = math.ceil(np.percentile(data.e_a, 95))
    #print "percentile Ea 95: " + str(perc90Ea)
    #dataNew = data[(data.e_a > perc5Ea) & (data.e_a < perc90Ea)]
    #data = dataNew
    ## se filtra el rango de valores Humedad de suelo de conae
    #perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
    #print "percentile humedad 5: " + str(perc10SM)
    #perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
    #print "percentile humedad 90: " + str(perc90SM)
    #print "Filtro por humedad"
    #dataNew = data[(data.SM_CONAE > perc10SM) & (data.SM_CONAE <= perc90SM)]
    #data = dataNew



def lecturaCompleta_etapa2(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    #data.SM_CONAE = data.SM_CONAE *100
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273


    statistics(data)

    ### se filtra el rango de valores Humedad de suelo de SMAP
    dataNew = data[(data.SM_SMAP > 5) & (data.SM_SMAP <= 50)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 99))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew


    #print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.T_s,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    ###se filtra el rango de valores de evapotransporacion
    data.Et = data.Et *0.1/8.0
    perc10Et = math.ceil(np.percentile(data.Et, 5))
    print "percentile Et 5: " + str(perc10Et)
    perc90Et = math.ceil(np.percentile(data.Et, 95))
    print "percentile Et 90: " + str(perc90Et)
    print "Filtro por humedad"
    dataNew = data[(data.Et > perc10Et) & (data.Et <= perc90Et)]
    data = dataNew

    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))


    #statistics(data)

    print "--------------------------------------------------------------------"
    print "Estadisticas antes de aplicar logaritmo y borrar las variables no usadas"
    statistics(data)
    print "--------------------------------------------------------------------"
    data.SM_SMAP = np.log10(data.SM_SMAP)
    data.T_s = np.log10(data.T_s)
    data.Et = np.log10(data.Et)
    #data.T_a = np.log10(data.T_a)
    data.HR = np.log10(data.HR)

    del data['HR']
    #del data['Et']

    ## se terminan de eliminar las columnas que no son necesarias
    #del data['FECHA_HORA']
    del data['NDVI']
    #del data['ID_DISPOSI']
    #graph(data)

    #print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data

####---------------------------------------------------------------------------
def lecturaCompleta_etapa2_NDVI(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    #data.SM_CONAE = data.SM_CONAE *100
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273


    statistics(data)

    ### se filtra el rango de valores Humedad de suelo de SMAP
    dataNew = data[(data.SM_SMAP > 5) & (data.SM_SMAP <= 50)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 99))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew


    #print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.T_s,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    ###se filtra el rango de valores de evapotransporacion
    data.Et = data.Et *0.1/8.0
    perc10Et = math.ceil(np.percentile(data.Et, 5))
    print "percentile Et 5: " + str(perc10Et)
    perc90Et = math.ceil(np.percentile(data.Et, 95))
    print "percentile Et 90: " + str(perc90Et)
    print "Filtro por humedad"
    dataNew = data[(data.Et > perc10Et) & (data.Et <= perc90Et)]
    data = dataNew

    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))


    #statistics(data)

    data.SM_SMAP = np.log10(data.SM_SMAP)
    data.T_s = np.log10(data.T_s)
    data.Et = np.log10(data.Et)
    #data.T_a = np.log10(data.T_a)
    data.HR = np.log10(data.HR)

    del data['HR']
    #del data['Et']

    ## se terminan de eliminar las columnas que no son necesarias
    #del data['FECHA_HORA']
    #del data['NDVI']
    #del data['ID_DISPOSI']
    #statistics(data)
    #graph(data)

    #print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data

####---------------------------------------------------------------------------


def lecturaCompleta_etapa2_2(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    #data.SM_CONAE = data.SM_CONAE *100
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273


    statistics(data)

    ## se filtra el rango de valores Humedad de suelo de SMAP
    dataNew = data[(data.SM_SMAP > 10) & (data.SM_SMAP <= 45)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    dataNew = data[(data.Sigma0 > -20) & (data.Sigma0 < -6)]
    data = dataNew

    #### se filtra el rango de valores de HR

    dataNew = data[(data.HR > 58) & (data.HR < 90)]
    data = dataNew



    ### se filtra el rango de valores de RSOILTEMPC
    dataNew = data[(data.T_s > 7) & (data.T_s < 27)]
    data = dataNew


    dataNew = data[ (data.Et > 1.0) & (data.NDVI < 41)]
    data = dataNew
    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Numero de muestras Final: " + str(len(data))


    #statistics(data)

    data.SM_SMAP = np.log10(data.SM_SMAP)
    data.T_s = np.log10(data.T_s)
    data.Et = np.log10(data.Et)
    #data.T_a = np.log10(data.T_a)
    data.HR = np.log10(data.HR)

    #del data['HR']
    #del data['Et']

    ## se terminan de eliminar las columnas que no son necesarias
    #del data['FECHA_HORA']
    del data['NDVI']
    #del data['ID_DISPOSI']
    #statistics(data)
    #graph(data)

    #print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data

####---------------------------------------------------------------------------

def lecturaSimple_etapa1(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_CONAE = data.SM_CONAE *100
    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.SM_CONAE > perc10SM) & (data.SM_CONAE <= perc90SM)]
    data = dataNew


    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))





    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    #print "Filtro por estaciones malas"
    #print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.50)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    #del data['RSOILMOIST']
    del data['NDVI_30m_B']
    #del data['HR']
    #del data['FECHA_HORA']
    del data['NDVI']
    del data['ID_DISPOSI']
    return data



def lecturaSimple_etapa2(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    #data.SM_CONAE = data.SM_CONAE *100
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273


    statistics(data)

    ## se filtra el rango de valores Humedad de suelo de SMAP
    dataNew = data[(data.SM_SMAP > 10) & (data.SM_SMAP <= 45)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 99))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew


    #print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.T_s,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    ###se filtra el rango de valores de evapotransporacion
    data.Et = data.Et *0.1/8.0
    perc10Et = math.ceil(np.percentile(data.Et, 15))
    print "percentile Et 5: " + str(perc10Et)
    perc90Et = math.ceil(np.percentile(data.Et, 95))
    print "percentile Et 90: " + str(perc90Et)
    print "Filtro por humedad"
    dataNew = data[(data.Et > perc10Et) & (data.Et <= perc90Et)]
    data = dataNew

    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    #del data['RSOILMOIST']
    #del data['NDVI_30m_B']
    #del data['HR']
    #del data['FECHA_HORA']
    del data['NDVI']
    #del data['ID_DISPOSI']
    return data

####----------------------------------------------------------------------------


def lecturaCompleta_etapa3(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.RSOILMOIST = data.RSOILMOIST *100
    data.SMAP = data.SMAP *100
    #del data['SMAP']
    del data['PP']

    del data['NDVI_30m_B']
    del data['SM10Km_PCA']

    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew


    dataNew = data[(data.SMAP > 15) & (data.SMAP <= 45)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))


    #statistics(data)

    data.SMAP = np.log10(data.SMAP)
    data.RSOILTEMPC = np.log10(data.RSOILTEMPC)
    data.Tension_va = np.log10(data.Tension_va)
    data.T_aire = np.log10(data.T_aire)
    data.HR = np.log10(data.HR)
    data.GPM = data.GPM*0.1

    del data['RSOILTEMPC']
    del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    #del data['PP']

    ## se terminan de eliminar las columnas que no son necesarias
    del data['FECHA_HORA']
    del data['NDVI']
    del data['ID_DISPOSI']
    #statistics(data)
    #graph(data)
    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    return data




####----------------------------------------------------------------------------

def lecturaCompletaMARS(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_CONAE = data.SM_CONAE *100
    #del data['NDVI']
    del data['NDVI_30m_B']
    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.SM_CONAE > perc10SM) & (data.SM_CONAE <= perc90SM)]
    data = dataNew


    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))





    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    print "Filtro por estaciones malas"
    print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.55)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    ##statistics(data)


    del data['RSOILTEMPC']
    del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    #del data['PP']

    ## se terminan de eliminar las columnas que no son necesarias
    del data['FECHA_HORA']
    del data['ID_DISPOSI']
    del data['NDVI']

    #graph2(data)
    #data.to_csv('/home/gag/Desktop/salida.csv')

    print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data



####----------------------------------------------------------------------------


def lecturaCompletaMLP_etapa1(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_CONAE = data.SM_CONAE *100
    #del data['NDVI']
    del data['NDVI_30m_B']
    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.SM_CONAE, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.SM_CONAE, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.SM_CONAE > perc10SM) & (data.SM_CONAE <= perc90SM)]
    data = dataNew


    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))





    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    print "Filtro por estaciones malas"
    print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.55)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    ##statistics(data)


    del data['RSOILTEMPC']
    del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    #del data['PP']

    ## se terminan de eliminar las columnas que no son necesarias
    del data['FECHA_HORA']
    del data['ID_DISPOSI']
    del data['NDVI']

    #graph2(data)
    #data.to_csv('/home/gag/Desktop/salida.csv')

    print "maximo humedad de suelo: " + str(np.max(data.SM_CONAE))
    return data

####----------------------------------------------------------------------------


def lecturaCompletaMLP_etapa2(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273
    #graph(data)

    dataNew = data[(data.SM_SMAP > 5) & (data.SM_SMAP <= 50)]
    data = dataNew

    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0 > perc5Back) & (data.Sigma0 < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 99))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.T_s,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    ## se filtra el rango de valores de evapotransporacion
    data.Et = data.Et *0.1/8.0
    perc10Et = math.ceil(np.percentile(data.Et, 5))
    print "percentile Et 5: " + str(perc10Et)
    perc90Et = math.ceil(np.percentile(data.Et, 95))
    print "percentile Et 90: " + str(perc90Et)
    print "Filtro por humedad"
    dataNew = data[(data.Et > perc10Et) & (data.Et <= perc90Et)]
    data = dataNew


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.55)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    statistics(data)
    #graph(data)
    ### se terminan de eliminar las columnas que no son necesarias
    del data['HR']
    #del data['Et']
    del data['NDVI']
    return data

####----------------------------------------------------------------------------


def lecturaCompletaMLP_etapa2_2(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    data.T_s = data.T_s -273
    #graph(data)

    dataNew = data[(data.SM_SMAP > 10) & (data.SM_SMAP <= 45)]
    data = dataNew


    ### se filtra el rango de valores de backscattering
    dataNew = data[(data.Sigma0 > -20) & (data.Sigma0 < -6)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de HR

    dataNew = data[(data.HR > 58) & (data.HR < 90)]
    data = dataNew



    ### se filtra el rango de valores de RSOILTEMPC
    dataNew = data[(data.T_s > 7) & (data.T_s < 27)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    dataNew = data[ (data.Et > 1.0) & (data.NDVI < 41)]
    data = dataNew
    #statistics(data)


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.8)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))



    statistics(data)
    #graph(data)
    ### se terminan de eliminar las columnas que no son necesarias
    #del data['HR']
    del data['Et']
    del data['NDVI']
    return data

####----------------------------------------------------------------------------





def lecturaCompletaMLP_etapa3(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.RSOILMOIST = data.RSOILMOIST *100
    data.SMAP = data.SMAP *100
    del data['PP']
    del data['NDVI_30m_B']
    del data['SM10Km_PCA']
    #statistics(data)
    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    print "percentile humedad 5: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 95))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    dataNew = data[(data.SMAP > 15) & (data.SMAP <= 45)]
    data = dataNew


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,0))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 95))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 95))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 95))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 96))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))





    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    #print "Filtro por estaciones malas"
    #print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.0) & (data.NDVI_30m_B < 0.55)]
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))


    ##statistics(data)

    #data.RSOILMOIST = (data.RSOILMOIST - np.mean(data.RSOILMOIST))/(np.max(data.RSOILMOIST) - np.min(data.RSOILMOIST))

    data.RSOILTEMPC = (data.RSOILTEMPC - np.mean(data.RSOILTEMPC))/(np.max(data.RSOILTEMPC) - np.min(data.RSOILTEMPC))
    data.Tension_va = (data.Tension_va - np.mean(data.Tension_va))/(np.max(data.Tension_va) - np.min(data.Tension_va))
    data.T_aire = (data.T_aire - np.mean(data.T_aire))/(np.max(data.T_aire) - np.min(data.T_aire))
    data.HR = (data.HR - np.mean(data.HR))/(np.max(data.HR) - np.min(data.HR))
    data.Sigma0_VV_ = (data.Sigma0_VV_ - np.mean(data.Sigma0_VV_))/(np.max(data.Sigma0_VV_) - np.min(data.Sigma0_VV_))
    data.GPM = data.GPM*0.1
    data.GPM = (data.GPM - np.mean(data.GPM))/(np.max(data.GPM) - np.min(data.GPM))


    ## se terminan de eliminar las columnas que no son necesarias
    del data['FECHA_HORA']
    del data['NDVI']
    del data['ID_DISPOSI']


    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    return data







###-------------------------------------------------------------------------
def statistics(data):
    print "-------------------------------------------------------------------"
    print data.describe()
    print "-------------------------------------------------------------------"
    return


def lecturaNew(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    print "Numero inicial de muestras: " + str(len(data))
    data.RSOILMOIST = data.RSOILMOIST *100


    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    print "percentile humedad 20: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 98))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew
    print "Media Humedad:" + str(np.mean(data.RSOILMOIST))
    print "desvio Humedad:" + str(np.std(data.RSOILMOIST))
    print "varianza Humedad:" + str(np.var(data.RSOILMOIST))


    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,5))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))
    ### se filtra el rango de valores de HR
    perc5HR = math.ceil(np.percentile(data.HR,0))
    print "percentile HR 5: " + str(perc5HR)
    perc90HR = math.ceil(np.percentile(data.HR, 98))
    print "percentile HR 95: " + str(perc90HR)
    dataNew = data[(data.HR > perc5HR) & (data.HR < perc90HR)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))

    ### se filtra el rango de valores de T_aire
    perc5Ta = math.ceil(np.percentile(data.T_aire,0))
    print "percentile Ta 5: " + str(perc5Ta)
    perc90Ta = math.ceil(np.percentile(data.T_aire, 98))
    print "percentile Ta 95: " + str(perc90Ta)
    dataNew = data[(data.T_aire > perc5Ta) & (data.T_aire < perc90Ta)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de RSOILTEMPC
    perc5Ts = math.ceil(np.percentile(data.RSOILTEMPC,0))
    print "percentile Ts 5: " + str(perc5Ts)
    perc90Ts = math.ceil(np.percentile(data.RSOILTEMPC, 98))
    print "percentile Ts 95: " + str(perc90Ts)
    dataNew = data[(data.RSOILTEMPC > perc5Ts) & (data.RSOILTEMPC < perc90Ts)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de Tension_va
    perc5Tv = math.ceil(np.percentile(data.Tension_va,0))
    print "percentile Tv 5: " + str(perc5Tv)
    perc90Tv = math.ceil(np.percentile(data.Tension_va, 98))
    print "percentile Tv 95: " + str(perc90Tv)
    dataNew = data[(data.Tension_va > perc5Tv) & (data.Tension_va < perc90Tv)]
    data = dataNew

    print "Numero de muestras: " + str(len(data))



    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    #dataNew = data[(data.ID_DISPOSI != 122)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 124)]
    #data = dataNew
    #dataNew = data[(data.ID_DISPOSI != 128)]
    #data = dataNew
    print "Filtro por estaciones malas"
    print "Numero de muestras: " + str(len(data))


    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.55)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    ## se terminan de eliminar las columnas que no son necesarias
    del data['FECHA_HORA']
    del data['ID_DISPOSI']
    del data['NDVI']

    #del data['RSOILTEMPC']
    #del data['Tension_va']
    #del data['T_aire']
    #del data['HR']

    graph(data)
    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    return data






#####-------------------------------------------------------------------------

def lecturaAplicacion(file):
    #file ="tablaCompleta.csv"
    data = pd.read_csv(file, sep=',', decimal=",")
    #print data
    print "Numero inicial de muestras: " + str(len(data))
    ### se convierten los valores de SM a porcentaje
    data.RSOILMOIST = data.RSOILMOIST *100

    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos

    ## se filtra el rango de valores Humedad de suelo de conae
    #perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    perc10SM = 6.25
    print "minimo SM: " + str(perc10SM)
    #perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 95))
    #perc90SM = 43.36
    perc90SM = 39
    print "maximo SM: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    #### se filtra el rango de valores de backscattering
    #perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,1))
    perc5Back = -17.82
    print "minimo back: " + str(perc5Back)
    #perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    #perc90Back =-4.53
    perc90Back =-6
    print "maximo back: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew
    print "Filtro por back"
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de T_aire
    min = 9.25
    print "minimo T_aire : " + str(min)
    #perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    max = 24.7
    print "maximo T_aire : " + str(max)
    dataNew = data[(data.T_aire  > min) & (data.T_aire  < max)]
    data = dataNew
    print "Filtro por T_aire"
    print "Numero de muestras: " + str(len(data))


    #### se filtra el rango de valores de HR
    min = 58
    print "minimo RH : " + str(min)
    max = 83.63
    print "maximo RH : " + str(max)
    dataNew = data[(data.HR  > min) & (data.HR  < max)]
    data = dataNew
    print "Filtro por HR"
    print "Numero de muestras: " + str(len(data))


    # se filtra el rango de valores de NDVI
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))



    ## se terminan de eliminar las columnas que no son necesarias
    #del data['ID_DISPOSI']
    del data['NDVI']
    del data['RSOILTEMPC']
    del data['Tension_va']
    #del data['T_aire']
    #del data['HR']

    data.T_aire = np.log10(data.T_aire)
    data.HR = np.log10(data.HR)


    del data['FECHA_HORA']
    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    graph3(data)
    return data




    #file ="tablaCompleta.csv"
    data = pd.read_csv(file, sep=',', decimal=",")
    #print data
    print "Numero inicial de muestras: " + str(len(data))
    ### se convierten los valores de SM a porcentaje
    data.RSOILMOIST = data.RSOILMOIST *100

    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos

    ## se filtra el rango de valores Humedad de suelo de conae
    #perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    perc10SM = 6.25
    print "minimo SM: " + str(perc10SM)
    #perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 95))
    #perc90SM = 43.36
    perc90SM = 39
    print "maximo SM: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew
    print "Numero de muestras: " + str(len(data))


    #### se filtra el rango de valores de backscattering
    #perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,1))
    perc5Back = -17.82
    print "minimo back: " + str(perc5Back)
    #perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    #perc90Back =-4.53
    perc90Back =-6
    print "maximo back: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew
    print "Filtro por back"
    print "Numero de muestras: " + str(len(data))

    #### se filtra el rango de valores de T_aire
    min = 9.25
    print "minimo T_aire : " + str(min)
    #perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    max = 24.7
    print "maximo T_aire : " + str(max)
    dataNew = data[(data.T_aire  > min) & (data.T_aire  < max)]
    data = dataNew
    print "Filtro por T_aire"
    print "Numero de muestras: " + str(len(data))


    #### se filtra el rango de valores de HR
    min = 58
    print "minimo RH : " + str(min)
    max = 83.63
    print "maximo RH : " + str(max)
    dataNew = data[(data.HR  > min) & (data.HR  < max)]
    data = dataNew
    print "Filtro por HR"
    print "Numero de muestras: " + str(len(data))


    # se filtra el rango de valores de NDVI
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.51)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))



    ## se terminan de eliminar las columnas que no son necesarias
    del data['ID_DISPOSI']
    del data['NDVI']
    #del data['RSOILTEMPC']
    #del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    data.RSOILTEMPC = (data.RSOILTEMPC - np.mean(data.RSOILTEMPC))/(np.max(data.RSOILTEMPC) - np.min(data.RSOILTEMPC))
    data.Tension_va = (data.Tension_va - np.mean(data.Tension_va))/(np.max(data.Tension_va) - np.min(data.Tension_va))
    data.T_aire = (data.T_aire - np.mean(data.T_aire))/(np.max(data.T_aire) - np.min(data.T_aire))
    data.HR = (data.HR - np.mean(data.HR))/(np.max(data.HR) - np.min(data.HR))
    data.PP = (data.PP - np.mean(data.PP))/(np.max(data.PP) - np.min(data.PP))
    data.Sigma0_VV_ = (data.Sigma0_VV_ - np.mean(data.Sigma0_VV_))/(np.max(data.Sigma0_VV_) - np.min(data.Sigma0_VV_))


    del data['FECHA_HORA']
    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    graph3(data)
    return data

#####-------------------------------------------------------------------------
def lecturaAplicacionMLP(file):
    #file ="tablaCompleta.csv"
    data = pd.read_csv(file, sep=',', decimal=",")
    #print data
    print "Numero inicial de muestras: " + str(len(data))

    ## se terminan de eliminar las columnas que no son necesarias
    #del data['NDVI']
    #del data['RSOILTEMPC']
    #del data['Tension_va']
    #del data['T_aire']
    #del data['HR']
    return data








def graph(data):

    fig = plt.figure(1,facecolor="white")
    #fig1 = fig.add_subplot(111)
    ##y = data.Sigma0_VV_
    y = data.SM_SMAP
    #xx = np.linspace(0,len(y),len(y))
    ##yy = fft(y) / len(y)
    #fig1.scatter(xx,y , color='blue',linewidth=3,label='HS')
    #fig1.plot(xx,y , color='blue',linewidth=3,label='HS')
    #fig1.set_title(u'SM_CONAE')
    #y=np.array(data.SM_CONAE)

    #fechas = np.array(data.FECHA_HORA)
    #vSalidaNew = []
    #vFechasNew = []
    #for i in range(0, len(y)):
        #if ((i%10)== 0):
            #vSalidaNew.append(y[i])
            #vFechasNew.append(fechas)
        #else:
            #vSalidaNew.append('')
            #vFechasNew.append('')
    #plt.xticks(np.arange(len(vSalidaNew)), vFechasNew,  size = 'small', rotation = 45)

    fig3 = plt.subplots()
    ax = plt.subplot(2,1,1)
    xx = np.linspace(0,len(y),len(y))
    pp = data.PP
    y2_ticks = np.linspace(0, np.max(pp), 10)
    y2_ticks = np.round(y2_ticks,2)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax.set_yticks(-1 * y2_ticks)
    ax.set_yticklabels(y2_ticklabels)
    ax.bar(xx, -pp, 0.1, color='r')
    ax.set_title(u'PP7d - Soil Temp')
    ax.set_ylabel('PP [mm]', color='r')
    plt.xlim(-5, len(pp))


    ax1 = plt.subplot(2,1,2)
    #y = data.RSOILTEMPC
    ##ax1.scatter(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    ax1.set_ylabel('temp [C]', color='b')
    yy = data.T_s
    ax1.plot(xx,yy , color='red',linewidth=3,label='air_temp')
    plt.xlim(-5, len(yy))

    #ax2 = ax1.twinx()
    ##y = data.interHR
    ##y = data.interHRpro
    #y = data.RSOILMOIST
    ##ax2.scatter(xx,y , color='green',linewidth=3,label='SM')
    #ax2.plot(xx,y , color='green',linewidth=3,label='SM')
    #ax2.set_ylabel('SM', color='g')
    #ax1.scatter(0, 0, color='green',linewidth=3, label='SM')
    #ax1.legend(loc=4)
    #plt.xlim(-5, len(y))



    #####-------------------------------------------------------------------------

    ##fig4 = plt.subplots()
    ##ax = plt.subplot(2,1,1)
    ##xx = np.linspace(0,len(y),len(y))
    ###pp = data.interPP7d
    ##pp = data.PP
    ##y2_ticks = np.linspace(0, np.max(pp), 10)
    ##y2_ticks = np.round(y2_ticks,2)
    ##y2_ticklabels = [str(i) for i in y2_ticks]
    ##ax.set_yticks(-1 * y2_ticks)
    ##ax.set_yticklabels(y2_ticklabels)
    ##ax.bar(xx, -pp, 0.1, color='r')
    ##ax.set_title(u'PP7d - HR')
    ##ax.set_ylabel('PP [mm]', color='r')
    ##plt.xlim(-5, len(pp))

    #fig3 = plt.subplots()
    #ax1 = plt.subplot(2,1,1)
    ##y = data.RSOILTEMPC
    ###ax1.scatter(xx,y , color='blue',linewidth=3,label='Soil_temp')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    ##ax1.set_ylabel('temp', color='b')
    #yy =  data.RSOILTEMPC
    #ax1.plot(xx,yy , color='blue',linewidth=3,label='soil_temp')


    #ax2 = ax1.twinx()
    ##y = data.interHR
    ##y = data.interHRpro
    #y = data.RSOILMOIST
    ##ax2.scatter(xx,y , color='green',linewidth=3,label='SM')
    #ax2.plot(xx,y , color='green',linewidth=3,label='SM')
    #ax2.set_ylabel('SM', color='g')
    #ax1.scatter(0, 0, color='green',linewidth=3, label='SM')
    #ax1.legend(loc=4)
    #plt.xlim(-5, len(y))



    ##ax1 = plt.subplot(2,1,2)
    ##y = data.HR
    ##ax1.scatter(xx,y , color='blue',linewidth=3,label='HR')
    ###ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    ##ax1.set_ylabel('HR', color='b')
    ##plt.xlim(-5, len(y))




    ##fig = plt.figure(4,facecolor="white")
    ##fig4 = fig.add_subplot(111)
    ###y = data.interHRpro
    ##y = data.PP7d_MJ
    ##xx = np.linspace(0,len(y),len(y))
    ##fig4.plot(xx,y , color='blue',linewidth=3,label='PP_MJ')
    ##y = data.PP7d_MB
    ##fig4.plot(xx,y , color='green',linewidth=3,label='PP_MB')
    ###fig4.set_title(u'interHRpro')
    ##fig4.set_title(u'interHR PP7d')
    ##plt.legend()

    #fig = plt.figure(5,facecolor="white")
    #fig5 = fig.add_subplot(111)
    #y = data.Sigma0_VV_
    #xx = np.linspace(0,len(y),len(y))
    #fig5.scatter(xx,y , color='green',linewidth=3,label='Sigma0_VV_')
    #fig5.set_title(u'Sigma0_VV_')


    #fig = plt.figure(6,facecolor="white")
    #fig6 = fig.add_subplot(111)
    #y = data.NDVI
    #xx = np.linspace(0,len(y),len(y))
    #fig6.scatter(xx,y , color='blue',linewidth=3,label='HS')
    #fig6.set_title(u'NDVI')
    #y=np.array(data.NDVI)
    #fechas = np.array(data.FECHA_HORA)
    ##print fechas
    #vSalidaNew = []
    #vFechasNew = []
    #for i in range(0, len(y)):
        #if ((i%7)== 0):
            #vSalidaNew.append(y[i])
            #vFechasNew.append(fechas)
        #else:
            #vSalidaNew.append('')
            #vFechasNew.append('')
    #plt.xticks(np.arange(len(vSalidaNew)), vFechasNew,  size = 'small', rotation = 45)

    ##fig = plt.figure(7,facecolor="white")
    ##fig7 = fig.add_subplot(111)
    ###y = data.Sigma0_VV_
    ##y1 = data.interPP7d
    ##y2 = data.PP7d_MB
    ##xx = np.linspace(0,len(y),len(y))
    ##fig7.plot(xx,y1 , color='blue',linewidth=3,label='interpolacion_PP7d')
    ##fig7.plot(xx,y2 , color='green',linewidth=3,label='PP7d_MonteBuey')
    ##plt.xlim(-5, len(y1))
    ##plt.legend()



    plt.show()
    return



####---------------------------------------------------------------------------
#def graph2(data):

    #y = data.RSOILMOIST
    ##xx = np.linspace(0,len(y),len(y))
    #x = data.PP
    #fig3 = plt.subplots(3, facecolor="white")

    #ax1 = plt.subplot(1,1,1)
    ##y = data.NDVI
    #ax1.scatter(x,y, linewidth=0.7,label='NDVI', marker = "*")

    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('NDVI', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    #ax1 = plt.subplot(3,1,2)
    #y = data.RSOILMOIST
    #ax1.scatter(xx,y, linewidth=0.5,label='SM')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('SM [%GSM]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    #ax = plt.subplot(3,1,3)
    #xx = np.linspace(0,len(y),len(y))
    #pp = data.PP
    #y2_ticks = np.linspace(0, np.max(pp), 9)
    #print y2_ticks
    #y2_ticks = np.round(y2_ticks,0)
    #y2_ticklabels = [str(int(i)) for i in y2_ticks]
    ##ax.set_yticks(-1 * y2_ticks)
    #ax.set_yticks(y2_ticks)
    #ax.set_yticklabels(y2_ticklabels)
    #ax.bar(xx, pp, 0.1, color='r')
    ##ax.set_title(u'PP7d - SM - Soil Temp')
    #ax.set_ylabel('PP [mm]', fontsize=14)
    #ax.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)




    #fig3 = plt.subplots(2, facecolor="white")

    #ax1 = plt.subplot(3,1,1)
    #y = data.Sigma0_VV_
    #ax1.scatter(xx,y, linewidth=0.5,label='Sig')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel(r'$\sigma^0$' + '[dB]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)




    #ax1 = plt.subplot(3,1,2)
    #y = data.HR
    #ax1.scatter(xx,y,linewidth =0.5,label='RH')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('HR [%]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)


    #ax1 = plt.subplot(3,1,3)
    #y = data.T_aire
    #ax1.scatter(xx,y, linewidth = 0.5,label='Ta')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Ta [$^\circ$C]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    #fechas = np.array(data.FECHA_HORA)
    ##print fechas
    #vSalidaNew = []
    #vFechasNew = []
    #for i in range(0, len(fechas)):
        #vSalidaNew.append(y[i])
        ##if (np.mod(i,2) ==0):
        #aaa = str(fechas[i])
        ##aaa = aaa[:-5]
        #dd = aaa[:-12]
        ##print "dia"+str(dd)
        #mm = aaa[-11:-9]
        ##print "mes"+str(mm)
        #yy = aaa[-8:-6]
        ##print "anio"+str(yy)
        #ff =str(mm +'/' + dd+'/'+ yy)
        #vFechasNew.append(ff)
        ##else:
            ##vFechasNew.append("")
        #plt.xticks(xx, vFechasNew,  fontsize=12, rotation =30 )

    ##ax1.set_xlabel('Day(mm/dd/yy)', fontsize=14)
    ###ax2 = ax1.twinx()
    ###y = data.RSOILMOIST
    ####ax2.scatter(xx,y , color='green',linewidth=0.5,label='Ts')
    ###ax2.set_ylabel('Ts [C]')
    ####ax2.legend(loc=4)
    ###ax2.set_xticklabels([])
    ###plt.xlim(-5, len(y))
    #fig3 = plt.subplots(3, facecolor="white")


    #ax1 = plt.subplot(3,1,1)
    #y = data.RSOILTEMPC
    #ax1.scatter(xx,y, linewidth = 0.5,label='Ts')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Ts [$^\circ$C]',fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)


    #ax1 = plt.subplot(3,1,2)
    #y = data.Tension_va
    #ax1.scatter(xx,y, linewidth = 0.5,label='Vt')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Vt [hPa]', fontsize=14)
    #ax1.set_xlabel('Day(mm/dd/yy)', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    #, rotation =45 )





    plt.show()
    return

def graph2(data):

    dataNew = data[ (data.PP <20)]
    x = dataNew.RSOILMOIST
    y = dataNew.PP

    fig3 = plt.subplots(3, facecolor="white")
    ax1 = plt.subplot(1,1,1)
    ax1.scatter(x,y, linewidth=0.9,label='NDVI', marker = "*")

    popt, pcov = curve_fit(func, x, y, p0=(1, 1e-6))
    print "AQUI----------------------------------------------------------"
    print popt

    xx = np.linspace(10, 45.0, len(x))
    #print xx

    #popt = np.array([0.11, -0.105])
    yy = func(xx, *popt)

    ax1.plot(xx,yy , color='blue',linewidth=2,label='Soil_temp')
    ax1.set_xlabel("observed value [% GSM]",fontsize=12)
    ax1.set_ylabel("PP [mm]",fontsize=12)
    a = np.round(popt[0],3)
    b = -np.round(popt[1],3)
    tit = "y="+str(a)+"exp("+str(b)+"x)"
    ax1.text(10, 7, tit, fontsize=12)

    #fig3 = plt.subplots(3, facecolor="white")
    #ax1 = plt.subplot(1,1,1)
    #ax1.scatter(y,yy, linewidth=0.9,label='NDVI', marker = "*")

    #ax1 = plt.subplot(3,1,2)
    #y = data.RSOILMOIST
    #ax1.scatter(xx,y, linewidth=0.5,label='SM')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('SM [%GSM]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    #ax = plt.subplot(3,1,3)
    #xx = np.linspace(0,len(y),len(y))
    #pp = data.PP
    #y2_ticks = np.linspace(0, np.max(pp), 9)
    #print y2_ticks
    #y2_ticks = np.round(y2_ticks,0)
    #y2_ticklabels = [str(int(i)) for i in y2_ticks]
    ##ax.set_yticks(-1 * y2_ticks)
    #ax.set_yticks(y2_ticks)
    #ax.set_yticklabels(y2_ticklabels)
    #ax.bar(xx, pp, 0.1, color='r')
    ##ax.set_title(u'PP7d - SM - Soil Temp')
    #ax.set_ylabel('PP [mm]', fontsize=14)
    #ax.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)




    #fig3 = plt.subplots(2, facecolor="white")

    #ax1 = plt.subplot(3,1,1)
    #y = data.Sigma0_VV_
    #ax1.scatter(xx,y, linewidth=0.5,label='Sig')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel(r'$\sigma^0$' + '[dB]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)




    #ax1 = plt.subplot(3,1,2)
    #y = data.HR
    #ax1.scatter(xx,y,linewidth =0.5,label='RH')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('HR [%]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)


    #ax1 = plt.subplot(3,1,3)
    #y = data.T_aire
    #ax1.scatter(xx,y, linewidth = 0.5,label='Ta')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Ta [$^\circ$C]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    #fechas = np.array(data.FECHA_HORA)
    ##print fechas
    #vSalidaNew = []
    #vFechasNew = []
    #for i in range(0, len(fechas)):
        #vSalidaNew.append(y[i])
        ##if (np.mod(i,2) ==0):
        #aaa = str(fechas[i])
        ##aaa = aaa[:-5]
        #dd = aaa[:-12]
        ##print "dia"+str(dd)
        #mm = aaa[-11:-9]
        ##print "mes"+str(mm)
        #yy = aaa[-8:-6]
        ##print "anio"+str(yy)
        #ff =str(mm +'/' + dd+'/'+ yy)
        #vFechasNew.append(ff)
        ##else:
            ##vFechasNew.append("")
        #plt.xticks(xx, vFechasNew,  fontsize=12, rotation =30 )

    ##ax1.set_xlabel('Day(mm/dd/yy)', fontsize=14)
    ###ax2 = ax1.twinx()
    ###y = data.RSOILMOIST
    ####ax2.scatter(xx,y , color='green',linewidth=0.5,label='Ts')
    ###ax2.set_ylabel('Ts [C]')
    ####ax2.legend(loc=4)
    ###ax2.set_xticklabels([])
    ###plt.xlim(-5, len(y))
    #fig3 = plt.subplots(3, facecolor="white")


    #ax1 = plt.subplot(3,1,1)
    #y = data.RSOILTEMPC
    #ax1.scatter(xx,y, linewidth = 0.5,label='Ts')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Ts [$^\circ$C]',fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)


    #ax1 = plt.subplot(3,1,2)
    #y = data.Tension_va
    #ax1.scatter(xx,y, linewidth = 0.5,label='Vt')
    ##ax1.plot(xx,y , color='blue',linewidth=3,label='Soil_temp')
    #ax1.set_ylabel('Vt [hPa]', fontsize=14)
    #ax1.set_xlabel('Day(mm/dd/yy)', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    #, rotation =45 )


    plt.show()
    return


def graph3(data): ### para la aplicacion

    y = data.RSOILMOIST
    xx = np.linspace(0,len(y),len(y))
    fig3 = plt.subplots(3, facecolor="white")

    ax1 = plt.subplot(5,1,1)
    ax1.scatter(xx,y, linewidth=0.5,label='SM')
    ax1.set_ylabel('SM [%GSM]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    ax1 = plt.subplot(5,1,2)
    y = data.Sigma0_VV_
    ax1.scatter(xx,y, linewidth=0.5,label='Sig')
    ax1.set_ylabel(r'$\sigma^0$' + '[dB]', fontsize=14)
    ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    ax = plt.subplot(5,1,3)
    xx = np.linspace(0,len(y),len(y))
    pp = data.PP
    y2_ticks = np.linspace(0, np.max(pp), 9)
    print y2_ticks
    y2_ticks = np.round(y2_ticks,0)
    y2_ticklabels = [str(int(i)) for i in y2_ticks]
    #ax.set_yticks(-1 * y2_ticks)
    ax.set_yticks(y2_ticks)
    ax.set_yticklabels(y2_ticklabels)
    ax.bar(xx, pp, 0.1, color='r')
    #ax.set_title(u'PP7d - SM - Soil Temp')
    ax.set_ylabel('PP [mm]', fontsize=14)
    #ax.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)



    ax1 = plt.subplot(5,1,4)
    y = data.HR
    ax1.scatter(xx,y,linewidth =0.5,label='RH')
    ax1.set_ylabel('HR [%]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    ax1 = plt.subplot(5,1,5)
    y = data.T_aire
    ax1.scatter(xx,y, linewidth = 0.5,label='Ta')
    ax1.set_ylabel('Ta [$^\circ$C]', fontsize=14)
    #ax1.set_xticklabels([])
    #plt.xlim(-0.5, len(y)+0.5)

    plt.show()
    return



def lectura2(file):
    #file ="tablaCompleta.csv"
    data = pd.read_csv(file, sep=',', decimal=",")
    #print data
    print "Numero inicial de muestras: " + str(len(data))
    #graph(data)
    #del data['FECHA_HORA']
    ### se convierten los valores de SM a porcentaje
    data.RSOILMOIST = data.RSOILMOIST *100

    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    dataNew = data[ (data.NDVI > 0.0) & (data.NDVI < 0.50)]
    data = dataNew
    print "Filtro por NDVI"
    print "Numero de muestras: " + str(len(data))

    ## se filtra el rango de valores Humedad de suelo de conae
    perc10SM = math.ceil(np.percentile(data.RSOILMOIST, 0))
    print "percentile humedad 20: " + str(perc10SM)
    perc90SM = math.ceil(np.percentile(data.RSOILMOIST, 90))
    print "percentile humedad 90: " + str(perc90SM)
    print "Filtro por humedad"
    dataNew = data[(data.RSOILMOIST > perc10SM) & (data.RSOILMOIST <= perc90SM)]
    data = dataNew
    print "Media Humedad:" + str(np.mean(data.RSOILMOIST))
    print "desvio Humedad:" + str(np.std(data.RSOILMOIST))
    print "varianza Humedad:" + str(np.var(data.RSOILMOIST))


    print "Numero de muestras: " + str(len(data))


    ### se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0_VV_,1))
    print "percentile back 5: " + str(perc5Back)
    perc90Back = math.ceil(np.percentile(data.Sigma0_VV_, 95))
    print "percentile back 95: " + str(perc90Back)
    dataNew = data[(data.Sigma0_VV_ > perc5Back) & (data.Sigma0_VV_ < perc90Back)]
    data = dataNew


    ## se filtran los dispositivos considerados que no se encuentran operativos
    ### por la gente de conae 122 124 128
    dataNew = data[(data.ID_DISPOSI != 122)]
    data = dataNew
    dataNew = data[(data.ID_DISPOSI != 124)]
    data = dataNew
    dataNew = data[(data.ID_DISPOSI != 128)]
    data = dataNew
    print "Filtro por estaciones malas"
    print "Numero de muestras: " + str(len(data))

    ## se terminan de eliminar las columnas que no son necesarias
    del data['ID_DISPOSI']
    #del data['NDVI']
    #del data['RSOILTEMPC']
    #del data['Tension_va']
    #del data['T_aire']
    #del data['HR']

    graph2(data)
    #print data
    del data['FECHA_HORA']
    print "maximo humedad de suelo: " + str(np.max(data.RSOILMOIST))
    return data


if __name__ == "__main__":
    #file = "filteredTable_dateAverageComplete2.csv"
    file = "tabla_calibration_validation.csv"
    data = lectura(file)
    graph2(data)