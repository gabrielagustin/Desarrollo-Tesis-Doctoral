#!/usr/bin/python
import gdal
from gdalconst import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from osgeo import gdal, ogr
import sys
from scipy.stats import threshold
import scipy.signal as sgn
from sklearn.preprocessing import normalize
from numpy import linalg as LA
import pandas as pd
import functions


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal
from numpy import linspace
from numpy import meshgrid
import selection


fechaSMAP = []
fechaSentinel = []
fechaNDVI = []
fechaMYD = []


fechaSMAP.append("2015-04-11")
fechaSentinel.append("2015-04-08")
fechaNDVI.append("2015-04-07")
fechaMYD.append("2015-04-06")

#fechaSMAP.append("2015-04-19")
#fechaSentinel.append("2015-04-18")
#fechaNDVI.append("2015-04-23")
#fechaMYD.append("2015-04-18")


fechaSMAP.append("2015-05-02")
fechaSentinel.append("2015-05-02")
fechaNDVI.append("2015-05-09")
fechaMYD.append("2015-05-02")


fechaSMAP.append("2015-05-10")
fechaSentinel.append("2015-05-12")
fechaNDVI.append("2015-05-09")
fechaMYD.append("2015-05-12")


fechaSMAP.append("2015-05-26")
fechaSentinel.append("2015-05-26")
fechaNDVI.append("2015-05-25")
fechaMYD.append("2015-05-25")

fechaSMAP.append("2015-06-03")
fechaSentinel.append("2015-06-05")
fechaNDVI.append("2015-06-10")
fechaMYD.append("2015-06-05")

fechaSMAP.append("2015-06-19")
fechaSentinel.append("2015-06-19")
fechaNDVI.append("2015-06-26")
fechaMYD.append("2015-06-18")


#fechaSMAP.append("2015-06-30")
#fechaSentinel.append("2015-06-29")
#fechaNDVI.append("2015-06-26")
#fechaMYD.append("2015-07-03")


fechaSMAP.append("2015-07-24")
fechaSentinel.append("2015-07-23")
fechaNDVI.append("2015-07-28")
fechaMYD.append("2015-07-20")

fechaSMAP.append("2015-08-17")
fechaSentinel.append("2015-08-16")
fechaNDVI.append("2015-08-13")
fechaMYD.append("2015-08-13")

fechaSMAP.append("2015-08-30")
fechaSentinel.append("2015-08-30")
fechaNDVI.append("2015-08-29")
fechaMYD.append("2015-08-29")

fechaSMAP.append("2015-09-10")
fechaSentinel.append("2015-09-09")
fechaNDVI.append("2015-09-14")
fechaMYD.append("2015-09-06")

fechaSMAP.append("2015-09-23")
fechaSentinel.append("2015-09-23")
fechaNDVI.append("2015-09-30")
fechaMYD.append("2015-09-22")


#fechaSMAP.append("2015-10-04")
#fechaSentinel.append("2015-10-03")
#fechaNDVI.append("2015-06-26")
#fechaMYD.append("2015-10-07")


fechaSMAP.append("2015-10-28")
fechaSentinel.append("2015-10-27")
fechaNDVI.append("2015-10-16")
fechaMYD.append("2015-10-24")

fechaSMAP.append("2015-11-13")
fechaSentinel.append("2015-11-10")
fechaNDVI.append("2015-11-17")
fechaMYD.append("2015-11-09")

fechaSMAP.append("2015-11-21")
fechaSentinel.append("2015-11-20")
fechaNDVI.append("2015-11-17")
fechaMYD.append("2015-11-17")

fechaSMAP.append("2015-12-18")
fechaSentinel.append("2015-12-14")
fechaNDVI.append("2015-12-19")
fechaMYD.append("2015-12-11")

fechaSMAP.append("2015-12-28")
fechaSentinel.append("2015-12-28")
fechaNDVI.append("2016-01-01")
fechaMYD.append("2015-12-27")

fechaSMAP.append("2016-01-08")
fechaSentinel.append("2016-01-07")
fechaNDVI.append("2016-01-01")
fechaMYD.append("2016-01-09")

fechaSMAP.append("2016-01-19")
fechaSentinel.append("2016-01-21")
fechaNDVI.append("2016-01-17")
fechaMYD.append("2016-01-17")

fechaSMAP.append("2016-01-27")
fechaSentinel.append("2016-01-31")
fechaNDVI.append("2016-02-02")
fechaMYD.append("2016-01-25")

fechaSMAP.append("2016-02-14")
fechaSentinel.append("2016-02-14")
fechaNDVI.append("2016-02-18")
fechaMYD.append("2016-02-10")

fechaSMAP.append("2016-02-25")
fechaSentinel.append("2016-02-24")
fechaNDVI.append("2016-02-18")
fechaMYD.append("2016-02-26")

fechaSMAP.append("2016-03-12")
fechaSentinel.append("2016-03-09")
fechaNDVI.append("2016-03-21")
fechaMYD.append("2016-03-13")

fechaSMAP.append("2016-03-20")
fechaSentinel.append("2016-03-19")
fechaNDVI.append("2016-03-21")
fechaMYD.append("2016-03-21")

fechaSMAP.append("2016-04-02")
fechaSentinel.append("2016-04-02")
fechaNDVI.append("2016-04-06")
fechaMYD.append("2016-04-06")

fechaSMAP.append("2016-04-13")
fechaSentinel.append("2016-04-12")
fechaNDVI.append("2016-04-22")
fechaMYD.append("2016-04-22")

fechaSMAP.append("2016-04-24")
fechaSentinel.append("2016-04-26")
fechaNDVI.append("2016-04-22")
fechaMYD.append("2016-04-22")

fechaSMAP.append("2016-05-20")
fechaSentinel.append("2016-05-20")
fechaNDVI.append("2016-05-08")
fechaMYD.append("2016-05-20")







Smap = []
Ts = []
Ta = []
HR = []
PP = []
Ea = []
sigma0 = []
NDVI = []
Et = []


arr = np.arange(len(fechaSMAP))
print arr

np.random.seed(0)
np.random.shuffle(arr)

print arr


porc=0.7
cant = len(arr)
print cant
numCal=int(round(cant)*porc)
print "Cantidad de fechas para calibracion: " + str(numCal)
numVal=int((cant)-numCal)
print "Cantidad de fechas para validacion: " +str(numVal)



indexCal =  arr[:numCal]
print indexCal
indexVal = arr[numCal:]
print indexVal

#dir = "ggarcia"
dir = "gag"


#for i in range(0,len(fechaSMAP)):
###-----------------------------------------------------------------
#for k in range(0,len(indexCal)):
    #i = indexCal[k]
    #print i
###-----------------------------------------------------------------
for k in range(0,len(indexVal)):
    i = indexVal[k]
    print i
###-----------------------------------------------------------------
    print fechaSMAP[i]
    fileSmap = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[i]+"/recorte/SM.img"
    src_ds_Smap, bandSmap, GeoTSmap, ProjectSmap = functions.openFileHDF(fileSmap, 1)

    fileEt = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MYD16/"+fechaMYD[i]+"/MYD16A_reprojected.data/ET_500m.img"
    src_ds_Et, bandEt, GeoTEt, ProjectEt = functions.openFileHDF(fileEt, 1)

    #fileTa = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/T_aire.asc"
    #src_ds_Ta, bandTa, GeoTTa, ProjectTa = functions.openFileHDF(fileTa, 1)

    #fileTs = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/Ts.asc"
    #src_ds_Ts, bandTs, GeoTTs, ProjectTs = functions.openFileHDF(fileTs, 1)

    ### temperatura de superficie de SMAP
    fileTs = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[i]+"/subset_reprojected.data/surface_temperature.img"
    src_ds_Ts, bandTs, GeoTTs, ProjectTs = functions.openFileHDF(fileTs, 1)

    ### PP de INTA
    ##filePP = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/PP.asc"
    ###PP de GPM
    filePP = "/media/"+dir+"/TOURO Mobile/GPM/"+fechaSentinel[i]+"/recorte.img"
    src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)


    fileHR = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/HR.asc"
    src_ds_HR, bandHR, GeoTHR, ProjectHR = functions.openFileHDF(fileHR, 1)

    #fileEa = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/Tension_vapor.asc"
    #src_ds_Ea, bandEa, GeoTEa, ProjectEa = functions.openFileHDF(fileEa, 1)


    fileNDVI = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/"+fechaNDVI[i]+"/NDVI_reproyectado_recortado"
    src_ds_NDVI, bandNDVI, GeoTNDVI, ProjectNDVI = functions.openFileHDF(fileNDVI, 1)


    ##fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/Sigma0_VV_db.img"

    #fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/1kx1k"
    fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/5kx5k"
    src_ds_Sar, bandSar, GeoTSar, ProjectSar = functions.openFileHDF(fileSar, 1)
    #print bandSar.shape

    ### se cambian las resoluciones de todas las imagenes a la de la sar
    #type = "Nearest"
    type = "Bilinear"
    nRow, nCol = bandSar.shape


    data_src = src_ds_Smap
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchSmap = match.ReadAsArray()
    #type = "Bilinear"
    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchSmap, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])


    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchSmap, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    #data_src = src_ds_Ta
    #data_match = src_ds_Sar
    #match = functions.matchData(data_src, data_match, type, nRow, nCol)
    #band_matchTa = match.ReadAsArray()

    data_src = src_ds_Ts
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchTs = match.ReadAsArray()

    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchTs, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    type = "Bilinear"
    data_src = src_ds_Et
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchEt = match.ReadAsArray()

    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchEt, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    type = "Bilinear"
    data_src = src_ds_PP
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchPP = match.ReadAsArray()

    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchPP, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])


    data_src = src_ds_HR
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchHR = match.ReadAsArray()


    #data_src = src_ds_Ea
    #data_match = src_ds_Sar
    #match = functions.matchData(data_src, data_match, type, nRow, nCol)
    #band_matchEa = match.ReadAsArray()

    data_src = src_ds_NDVI
    data_match = src_ds_Sar
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_matchNDVI = match.ReadAsArray()

    #fig, ax = plt.subplots()
    #im3 = ax.imshow(band_matchNDVI, interpolation='None',cmap='gray')
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    #plt.show()

    Smap.append(band_matchSmap.flatten())
    #Ta.append(band_matchTa.flatten())
    Ts.append(band_matchTs.flatten())
    PP.append(band_matchPP.flatten())
    HR.append(band_matchHR.flatten())
    #Ea.append(band_matchEa.flatten())
    sigma0.append(bandSar.flatten())
    NDVI.append(band_matchNDVI.flatten())
    Et.append(band_matchEt.flatten())

a = np.array(Smap).flatten()

#b = np.array(Ta).flatten()

c = np.array(Ts).flatten()

d = np.array(PP).flatten()

e = np.array(HR).flatten()

#f = np.array(Ea).flatten()

g = np.array(sigma0).flatten()

h = np.array(NDVI).flatten()

p = np.array(Et).flatten()


#df = pd.DataFrame({'SM_SMAP' :a,'T_s' :c,'PP' :d, 'Sigma0':g, 'NDVI':h, 'Et': p, 'HR':e})
#df.to_csv("/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Modelo/tabla_Completa.csv", decimal = ",")
#print "Archivo Validacion creado con exito!"


## se van a generar dos archivos una tabla completa para calibrar (70 porciento de las fechas)
## y otra tabla para validar (el 30 porciento restante)


#df = pd.DataFrame({'SM_SMAP' :a,'T_s' :c,'PP' :d, 'Sigma0':g, 'NDVI':h, 'Et': p, 'HR':e})
#df.to_csv("/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Modelo/tabla_completa_Calibracion.csv", decimal = ",")
#print "Archivo Calibracion creado con exito!"

#df = pd.DataFrame({'SM_SMAP' :a,'T_s' :c,'PP' :d, 'Sigma0':g, 'NDVI':h, 'Et': p, 'HR':e})
#df.to_csv("/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Modelo/tabla_completa_Validacion.csv", decimal = ",")
#print "Archivo Validacion creado con exito!"
#