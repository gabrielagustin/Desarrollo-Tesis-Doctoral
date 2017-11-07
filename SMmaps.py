#!/usr/bin/python
import gdal
from gdalconst import *
import matplotlib.pyplot as plt
import numpy.ma as np
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma


def geospatial_coor(nameFile):
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    #Tvis-animated.gif
    try:
        src_ds = gdal.Open(nameFile)
    except RuntimeError, e:
        print 'Unable to open INPUT.tif'
        print e
        sys.exit(1)
    gt1 = src_ds.GetGeoTransform()
    ##### r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    #if "MODIS" in nameFile:
        #print "ACA MODIS"
        #print src_ds.RasterXSize
        #print src_ds.RasterYSize
        #print gt1
        #r1 = [gt1[0], gt1[3], gt1[3] + (gt1[5] * src_ds.RasterYSize), gt1[0] + (gt1[1] * src_ds.RasterXSize)]
    #else:
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * src_ds.RasterXSize), gt1[3] + (gt1[5] * src_ds.RasterYSize)]
    return r1

def openImage(nameFile,intersection):
        # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    band_num = 1
    #Tvis-animated.gif
    try:
        src_ds = gdal.Open(nameFile)
    except RuntimeError, e:
        print 'Unable to open INPUT.tif'
        print e
        sys.exit(1)
    try:
        srcband = src_ds.GetRasterBand(band_num)
    except RuntimeError, e:
        # for example, try GetRasterBand(10)
        print 'Band ( %i ) not found' % band_num
        print e
        sys.exit(1)
    gt1 = src_ds.GetGeoTransform()
    Project = src_ds.GetProjection()
    print nameFile
    print gt1
    xOrigin = gt1[0]
    yOrigin = gt1[3]
    pixelWidth = float(gt1[1])
    pixelHeight = float(gt1[5])
    xmin = intersection[0]
    ymax = intersection[1]
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((np.abs(intersection[0])-np.abs(intersection[2]))/pixelWidth)
    ycount =  int((np.abs(intersection[1])-np.abs(intersection[3]))/pixelHeight)
    #print nameFile
    if (xoff == 0 and yoff == 0):
        lc = srcband.ReadAsArray()
    else:
        #print nameFile
        #print xoff
        #print yoff
        #print xcount
        #print ycount
        #if "MODIS" in nameFile:
            #lc = srcband.ReadAsArray(xoff, yoff, int(xcount), int(ycount))
        #else:
        lc = srcband.ReadAsArray(xoff, yoff, int(xcount), int(ycount))
    return src_ds, lc, gt1, Project

def applyNDVIfilter(sar,L8, etapa):
    result = sar
    rSar, cSar = sar.shape
    mask = np.ones((rSar, cSar))
    #print cFactor
    if (etapa == "etapa1"):
        print "YES"
        for i in range(0, rSar):
            for j in range(0, cSar):
                if (L8[i, j] > 0.45 ): mask[i,j] = 0
                if (L8[i, j] < 0.1 ): mask[i,j] = 0
    if (etapa == "etapa2"):
        for i in range(0, rSar):
            for j in range(0, cSar):
                if (L8[i, j] > 0.8 ): mask[i,j] = 0
                if (L8[i, j] < 0.1 ): mask[i,j] = 0
    result = sar*mask
    return result,mask


def applyWaterfilter(sar,modis):
    #print sar.shape
    rSar, cSar = sar.shape
    mask = np.zeros((rSar, cSar))
    #print modis.shape
    rModis, cModis = modis.shape
    for i in range(0, rSar):
        for j in range(0, cSar):
            if (modis[i, j] < 0.1): mask[i,j] = 1
    #mask = np.ones((r,c))
    result = sar*mask*-30
    return result, mask



def applyBackfilter(sar):
    result = threshold(sar, threshmin=-18, threshmax=-6, newval=0)
    print result
    #result = sgn.medfilt(result, 9)
    #result[result < 0] = 1
    mask = result
    mask[mask < 0] = 1
    result = result*mask

    r,c = sar.shape
    mask = np.ones((r,c))
    mask = mask - result
    return result, mask



def applyCityfilter(sar, L8_maskCity):
    r, c = sar.shape
    result = sar
    mask = np.ones((r,c))
    for i in range(0, r):
        for j in range(0, c):
            if (L8_maskCity[i, j] != 0 ): mask[i,j] = 0
    result = result*mask
    return result, mask



def meteoMap (sar, meteo):
    rSar, cSar = sar.shape
    sarMeteo = np.zeros((rSar, cSar))
    rMeteo, cMeteo = meteo.shape
    rFactor = rSar/float(rMeteo)
    #print rFactor
    cFactor = cSar/float(cMeteo)
    #print cFactor
    for i in range(0, rSar):
        for j in range(0, cSar):
            indexi = int(i/rFactor)
            indexj = int(j/cFactor)
            sarMeteo[i,j] = meteo[indexi, indexj]
    return sarMeteo



def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output



def calculateMaps(MLRmodel, MLPmodel, etapa):

    #dir = "ggarcia"
    dir = "gag"

    path = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Modelo/mapasCreados/Etapa2/"

    fechaSentinel= []
    fechaNDVI= []
    fechaLandsat8=[]
    fechaSMAP=[]
    fechaMYD=[]

    if (etapa == "etapa1"):
        fechaSentinel.append("2015-06-29")
        fechaLandsat8.append("2015-06-18")
        fechaSentinel.append("2015-10-03")
        fechaLandsat8.append("2015-10-08")
        fechaSentinel.append("2015-12-28")
        fechaLandsat8.append("2015-12-27")
        fechaSentinel.append("2016-03-19")
        fechaLandsat8.append("2016-03-16")

        Ta = []
        HR = []
        PP = []
        sigma0 = []


        for i in range(0,len(fechaSentinel)):
            print fechaSentinel[i]

            fileTa = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/T_aire.asc"
            src_ds_Ta, bandTa, GeoTTa, ProjectTa = functions.openFileHDF(fileTa, 1)

            filePP = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/PP.asc"
            src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)

            fileHR = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/HR.asc"
            src_ds_HR, bandHR, GeoTHR, ProjectHR = functions.openFileHDF(fileHR, 1)

            fileNDVI = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Landsat8/"+fechaLandsat8[i]+"/NDVI_recortado"
            src_ds_NDVI, bandNDVI, GeoTNDVI, ProjectNDVI = functions.openFileHDF(fileNDVI, 1)

            fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/recorte_30mx30m.img"

            nameFileMLR = "mapa_MLR_30m_"+str(fechaSentinel[i])
            nameFileMLP = "mapa_MLP_30m_"+str(fechaSentinel[i])


            src_ds_Sar, bandSar, GeoTSar, ProjectSar = functions.openFileHDF(fileSar, 1)
            print ProjectSar


            fileMascara = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Landsat8/2015-06-18/mascaraciudadyalgomas_reprojected/subset_1_of_Band_Math__b1_5.data/Band_Math__b1_5.img"
            src_ds_Mas, bandMas, GeoTMas, ProjectMas = functions.openFileHDF(fileMascara, 1)

            ### se cambian las resoluciones de todas las imagenes a la de la sar
            #type = "Nearest"
            type = "Bilinear"
            nRow, nCol = bandSar.shape

            data_src = src_ds_Mas
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchCity = match.ReadAsArray()

            data_src = src_ds_Ta
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchTa = match.ReadAsArray()

            data_src = src_ds_PP
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchPP = match.ReadAsArray()
            #fig, ax = plt.subplots()
            #ax.imshow(bandSar, interpolation='None',cmap=cm.gray)
            #plt.show()

            data_src = src_ds_HR
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchHR = match.ReadAsArray()

            data_src = src_ds_NDVI
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchNDVI = match.ReadAsArray()

            ### se filtra la imagen SAR
            #print "Se filtran las zonas con NDVI mayores a 0.45 y con NDVI menores a 0"
            print etapa
            sarEnmask, maskNDVI = applyNDVIfilter(bandSar,band_matchNDVI, etapa)
            rSar, cSar = maskNDVI.shape
            maskNDVI2 = np.zeros((rSar, cSar))
            #print cFactor
            for i in range(0, rSar):
                for j in range(0, cSar):
                    if (maskNDVI[i, j] == 0 ): maskNDVI2[i,j] = 1

            filtWater, maskWater = applyWaterfilter(bandSar,band_matchNDVI)
            #fig, ax = plt.subplots()
            #ax.imshow(filtWater, interpolation='None',cmap=cm.gray)
            #plt.show()


            #sarEnmask, maskCity = applyCityfilter(sarEnmask,L8maskCity)
            #sarEnmask, maskSAR = applyBackfilter(sarEnmask)
            sarEnmask2 = np.copy(sarEnmask)


            r,c = bandSar.shape

            ### los datos para el modelo MLR llevan log

            dataMap_MLR = pd.DataFrame({'Sigma0' :sarEnmask.flatten(),'T_aire' :(np.log10(band_matchTa)).flatten(), 'HR' :(np.log10(band_matchHR)).flatten(),'PP' :band_matchPP.flatten()})

            dataMap_MLR = dataMap_MLR.fillna(0)
            mapSM_MLR = MLRmodel.predict(dataMap_MLR)
            ## debo invertir la funcion flatten()
            mapSM_MLR = mapSM_MLR.reshape(r,c)
            mapSM_MLR  = 10**(mapSM_MLR)
            mapSM_MLR[mapSM_MLR < 0] = 0

            mapSM_MLR = mapSM_MLR*maskNDVI#*maskCity

            #fig, ax = plt.subplots()
            #ax.imshow(mapSM_MLR, interpolation='None',cmap=cm.gray)
            #plt.show()



            OldRange = (np.max(band_matchTa)  - np.min(band_matchTa))
            NewRange = (1 + 1)
            Ta = (((band_matchTa - np.min(band_matchTa)) * NewRange) / OldRange) -1

            OldRange = (np.max(band_matchHR)  - np.min(band_matchHR))
            NewRange = (1 + 1)
            HR = (((band_matchHR - np.min(band_matchHR)) * NewRange) / OldRange) -1

            OldRange = (np.max(band_matchPP)  - np.min(band_matchPP))
            NewRange = (1 + 1)
            PP = (((band_matchPP - np.min(band_matchPP)) * NewRange) / OldRange) -1

            OldRange = (np.max(sarEnmask2)  - np.min(sarEnmask2))
            NewRange = (1 + 1)
            sar2 = (((sarEnmask2 - np.min(sarEnmask2)) * NewRange) / OldRange) -1


            dataMap_MLP = pd.DataFrame({'T_aire' :Ta.flatten(),'Sigma0' :sar2.flatten(), 'HR' :HR.flatten(), 'PP' :PP.flatten()})
            dataMap_MLP = dataMap_MLP[[ 'T_aire', 'PP','Sigma0', 'HR']]

            #print dataMap_MLP
            ###.describe()
            dataMap_MLP = dataMap_MLP.fillna(0)
            mapSM_MLP = MLPmodel.predict(dataMap_MLP)

            mapSM_MLP = mapSM_MLP.reshape(r,c)
            #print mapSM_MLR.shape
            mapSM_MLP[mapSM_MLP < 0] = 0
            mapSM_MLP = mapSM_MLP*maskNDVI
            #fig, ax = plt.subplots()
            #ax.imshow(mapSM_MLP, interpolation='None',cmap=cm.gray)
            #plt.show()


            my_cmap = cm.Blues
            my_cmap.set_under('k', alpha=0)
            my_cmap1 = cm.Greens
            my_cmap1.set_under('k', alpha=0)
            my_cmap2 = cm.OrRd
            my_cmap2.set_under('k', alpha=0)
            my_cmap3 = cm.Oranges
            my_cmap3.set_under('k', alpha=0)


            fig, ax = plt.subplots()


            #im0 = ax.imshow(mapSM_MLR, cmap=my_cmap3)#, vmin=5, vmax=55, extent=[xmin,xmax,ymin,ymax], interpolation='None')
            #maskNDVI2 = ma.masked_where(maskNDVI2 == 0,maskNDVI2)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap1)

            im0 = ax.imshow(mapSM_MLR, cmap=my_cmap1, clim=(5, 45))
            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
            im=ax.imshow(pp, cmap=my_cmap2, interpolation='Bilinear')


            kk = ma.masked_where(filtWater == 0, filtWater)
            im = ax.imshow(kk, cmap=my_cmap, interpolation='Bilinear')

            #oo = ma.masked_where(sarEnmask == 0, sarEnmask)
            #plt.imshow(oo, cmap=my_cmap3, interpolation='None')
            transform = GeoTSar
            xmin,xmax,ymin,ymax=transform[0],transform[0]+transform[1]*src_ds_Sar.RasterXSize,transform[3]+transform[5]*src_ds_Sar.RasterYSize,transform[3]
            print xmin
            print xmax

            ax.xaxis.tick_top()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="5%", pad=0.05)
            cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
            cb.set_label('Volumetric SM (%)')
            cb.set_clim(vmin=5, vmax=50)


            fig, ax = plt.subplots()


            #im0 = ax.imshow(mapSM_MLR, cmap=my_cmap3)#, vmin=5, vmax=55, extent=[xmin,xmax,ymin,ymax], interpolation='None')
            #maskNDVI2 = ma.masked_where(maskNDVI2 == 0,maskNDVI2)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap1)

            im0= ax.imshow(mapSM_MLP, cmap=my_cmap1, clim=(5, 45))
            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
            ax.imshow(pp, cmap=my_cmap2, interpolation='Bilinear')


            kk = ma.masked_where(filtWater == 0, filtWater)
            ax.imshow(kk, cmap=my_cmap, interpolation='Bilinear')

            ax.xaxis.tick_top()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="5%", pad=0.05)
            cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
            cb.set_label('Volumetric SM (%)')
            cb.set_clim(vmin=5, vmax=50)

            plt.show()

            #im1 = ax.imshow(filtWater, cmap=my_cmap)
            #maskNDVI2 = ma.masked_where(filtWater == 0,filtWater)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap)
            #im1 = ax.imshow(band_matchCity, cmap=my_cmap2)

            #mapSM_MLP = mapSM_MLP*maskNDVI

            #functions.createHDFfile(path, nameFileMLR, 'ENVI', mapSM_MLR, c, r, GeoTSar, ProjectSar)
            #functions.createHDFfile(path, nameFileMLP, 'ENVI', mapSM_MLP, c, r, GeoTSar, ProjectSar)

    if (etapa == "etapa2"):

        fechaSMAP.append("2015-04-11")
        fechaSentinel.append("2015-04-08")
        fechaNDVI.append("2015-04-07")
        fechaMYD.append("2015-04-06")


        #fechaSMAP.append("2015-05-02")
        #fechaSentinel.append("2015-05-02")
        #fechaNDVI.append("2015-05-09")
        #fechaMYD.append("2015-05-02")


        #fechaSMAP.append("2015-05-10")
        #fechaSentinel.append("2015-05-12")
        #fechaNDVI.append("2015-05-09")
        #fechaMYD.append("2015-05-12")


        fechaSMAP.append("2015-05-26")
        fechaSentinel.append("2015-05-26")
        fechaNDVI.append("2015-05-25")
        fechaMYD.append("2015-05-25")

        #fechaSMAP.append("2015-06-03")
        #fechaSentinel.append("2015-06-05")
        #fechaNDVI.append("2015-06-10")
        #fechaMYD.append("2015-06-05")

        #fechaSMAP.append("2015-06-19")
        #fechaSentinel.append("2015-06-19")
        #fechaNDVI.append("2015-06-26")
        #fechaMYD.append("2015-06-18")

        #fechaSMAP.append("2015-07-24")
        #fechaSentinel.append("2015-07-23")
        #fechaNDVI.append("2015-07-28")
        #fechaMYD.append("2015-07-20")

        fechaSMAP.append("2015-08-17")
        fechaSentinel.append("2015-08-16")
        fechaNDVI.append("2015-08-13")
        fechaMYD.append("2015-08-13")

        #fechaSMAP.append("2015-08-30")
        #fechaSentinel.append("2015-08-30")
        #fechaNDVI.append("2015-08-29")
        #fechaMYD.append("2015-08-29")

        fechaSMAP.append("2015-09-10")
        fechaSentinel.append("2015-09-09")
        fechaNDVI.append("2015-09-14")
        fechaMYD.append("2015-09-06")

        #fechaSMAP.append("2015-09-23")
        #fechaSentinel.append("2015-09-23")
        #fechaNDVI.append("2015-09-30")
        #fechaMYD.append("2015-09-22")

        #fechaSMAP.append("2015-10-28")
        #fechaSentinel.append("2015-10-27")
        #fechaNDVI.append("2015-10-16")
        #fechaMYD.append("2015-10-24")

        fechaSMAP.append("2015-11-13")
        fechaSentinel.append("2015-11-10")
        fechaNDVI.append("2015-11-17")
        fechaMYD.append("2015-11-09")

        #fechaSMAP.append("2015-11-21")
        #fechaSentinel.append("2015-11-20")
        #fechaNDVI.append("2015-11-17")
        #fechaMYD.append("2015-11-17")

        #fechaSMAP.append("2015-12-18")
        #fechaSentinel.append("2015-12-14")
        #fechaNDVI.append("2015-12-19")
        #fechaMYD.append("2015-12-11")

        fechaSMAP.append("2015-12-28")
        fechaSentinel.append("2015-12-28")
        fechaNDVI.append("2016-01-01")
        fechaMYD.append("2015-12-27")

        #fechaSMAP.append("2016-01-08")
        #fechaSentinel.append("2016-01-07")
        #fechaNDVI.append("2016-01-01")
        #fechaMYD.append("2016-01-09")

        #fechaSMAP.append("2016-01-19")
        #fechaSentinel.append("2016-01-21")
        #fechaNDVI.append("2016-01-17")
        #fechaMYD.append("2016-01-17")

        fechaSMAP.append("2016-01-27")
        fechaSentinel.append("2016-01-31")
        fechaNDVI.append("2016-02-02")
        fechaMYD.append("2016-01-25")

        #fechaSMAP.append("2016-02-14")
        #fechaSentinel.append("2016-02-14")
        #fechaNDVI.append("2016-02-18")
        #fechaMYD.append("2016-02-10")

        #fechaSMAP.append("2016-02-25")
        #fechaSentinel.append("2016-02-24")
        #fechaNDVI.append("2016-02-18")
        #fechaMYD.append("2016-02-26")

        fechaSMAP.append("2016-03-12")
        fechaSentinel.append("2016-03-09")
        fechaNDVI.append("2016-03-21")
        fechaMYD.append("2016-03-13")

        fechaSMAP.append("2016-03-20")
        fechaSentinel.append("2016-03-19")
        fechaNDVI.append("2016-03-21")
        fechaMYD.append("2016-03-21")

        #fechaSMAP.append("2016-04-02")
        #fechaSentinel.append("2016-04-02")
        #fechaNDVI.append("2016-04-06")
        #fechaMYD.append("2016-04-06")

        #fechaSMAP.append("2016-04-13")
        #fechaSentinel.append("2016-04-12")
        #fechaNDVI.append("2016-04-22")
        #fechaMYD.append("2016-04-22")

        #fechaSMAP.append("2016-04-24")
        #fechaSentinel.append("2016-04-26")
        #fechaNDVI.append("2016-04-22")
        #fechaMYD.append("2016-04-22")

        #fechaSMAP.append("2016-05-20")
        #fechaSentinel.append("2016-05-20")
        #fechaNDVI.append("2016-05-08")
        #fechaMYD.append("2016-05-20")

        #fechaSMAP.append("2015-04-19")
        #fechaSentinel.append("2015-04-18")
        #fechaNDVI.append("2015-04-23")
        #fechaMYD.append("2015-04-18")

        #fechaSMAP.append("2015-06-30")
        #fechaSentinel.append("2015-06-29")
        #fechaNDVI.append("2015-06-26")
        #fechaMYD.append("2015-07-03")

        #fechaSMAP.append("2015-10-04")
        #fechaSentinel.append("2015-10-03")
        #fechaNDVI.append("2015-06-26")
        #fechaMYD.append("2015-10-07")

        #fechaSMAP.append("2016-03-20")
        #fechaSentinel.append("2016-03-19")
        #fechaNDVI.append("2016-03-21")
        #fechaMYD.append("2016-03-21")

        #fechaSMAP.append("2015-12-28")
        #fechaSentinel.append("2015-12-28")
        #fechaNDVI.append("2016-01-01")
        #fechaMYD.append("2015-12-27")


        Ts = []
        Ta = []
        HR = []
        PP = []
        Ea = []
        sigma0 = []
        NDVI = []

        dir = "gag"

        for i in range(0,len(fechaSentinel)):
            print fechaSentinel[i]

            #fileTa = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/T_aire.asc"
            #src_ds_Ta, bandTa, GeoTTa, ProjectTa = functions.openFileHDF(fileTa, 1)

            #fileTs = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/Ts.asc"
            #src_ds_Ts, bandTs, GeoTTs, ProjectTs = functions.openFileHDF(fileTs, 1)

            #filePP = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/PP.asc"
            #src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)

            #### temperatura de superficie de SMAP
            fileTs = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[i]+"/subset_reprojected.data/surface_temperature.img"
            src_ds_Ts, bandTs, GeoTTs, ProjectTs = functions.openFileHDF(fileTs, 1)

            ##filePP = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/PP.asc"
            ### PP de GPM
            filePP = "/media/gag/TOURO Mobile/GPM/"+fechaSentinel[i]+"/recorte.img"
            src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)

            fileEt = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MYD16/"+fechaMYD[i]+"/MYD16A_reprojected.data/ET_500m.img"
            src_ds_Et, bandEt, GeoTEt, ProjectEt = functions.openFileHDF(fileEt, 1)

            fileHR = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/HR.asc"
            src_ds_HR, bandHR, GeoTHR, ProjectHR = functions.openFileHDF(fileHR, 1)

            #fileEa = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/Tension_vapor.asc"
            #src_ds_Ea, bandEa, GeoTEa, ProjectEa = functions.openFileHDF(fileEa, 1)


            fileNDVI = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/"+fechaNDVI[i]+"/NDVI_reproyectado_recortado"
            src_ds_NDVI, bandNDVI, GeoTNDVI, ProjectNDVI = functions.openFileHDF(fileNDVI, 1)


            #fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/Sigma0_VV_db.img"

            #fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/1kx1k"
            #nameFileMLR = "mapa_MLR_1Km_"+str(fechaSentinel[i])
            #nameFileMLP = "mapa_MLP_1Km_"+str(fechaSentinel[i])

            fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/5kx5k"
            nameFileMLR = "mapa_MLR_5Km_"+str(fechaSentinel[i])
            nameFileMLP = "mapa_MLP_5Km_"+str(fechaSentinel[i])

            src_ds_Sar, bandSar, GeoTSar, ProjectSar = functions.openFileHDF(fileSar, 1)

            ### se cambian las resoluciones de todas las imagenes a la de la sar
            #type = "Nearest"
            type = "Bilinear"
            nRow, nCol = bandSar.shape

            #data_src = src_ds_Ta
            #data_match = src_ds_Sar
            #match = functions.matchData(data_src, data_match, type, nRow, nCol)
            #band_matchTa = match.ReadAsArray()

            data_src = src_ds_Ts
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchTs = match.ReadAsArray()
            band_matchTs = band_matchTs-273
            #fig, ax = plt.subplots()
            #ax.imshow(band_matchTs, interpolation='None',cmap=cm.gray)


            data_src = src_ds_PP
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchPP = match.ReadAsArray()
            band_matchPP = band_matchPP * 0.1
            #fig, ax = plt.subplots()
            #ax.imshow(band_matchPP, interpolation='None',cmap=cm.gray)
            #plt.show()



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

            data_src = src_ds_Et
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchEt = match.ReadAsArray()
            band_matchEt = band_matchEt * 0.1/8.0




            ### se filtra la imagen SAR
            #print "Se filtran las zonas con NDVI mayores a 0.45 y con NDVI menores a 0"
            sarEnmask, maskNDVI = applyNDVIfilter(bandSar,band_matchNDVI, etapa)
            #fig, ax = plt.subplots()
            #ax.imshow(maskNDVI, interpolation='None',cmap=cm.gray)
            #plt.show()


            #sarEnmask, maskCity = applyCityfilter(sarEnmask,L8maskCity)
            #sarEnmask, maskSAR = applyBackfilter(sarEnmask)
            sarEnmask2 = np.copy(sarEnmask)


            r,c = bandSar.shape

            ### los datos para el modelo MLR llevan log

            #dataMap_MLR = pd.DataFrame({'Sigma0' :sarEnmask.flatten(),'T_s' :(np.log10(band_matchTs)).flatten(),'PP' :band_matchPP.flatten()})
            #dataMap_MLR = pd.DataFrame({'Sigma0' :sarEnmask.flatten(),'T_s' :(np.log10(band_matchTs)).flatten(), 'HR' :(np.log10(band_matchHR)).flatten(),'PP' :band_matchPP.flatten()})
            dataMap_MLR = pd.DataFrame({'Sigma0' :sarEnmask.flatten(),'T_s' :(np.log10(band_matchTs)).flatten(), 'Et' :(np.log10(band_matchEt)).flatten(),'PP' :band_matchPP.flatten()})

            dataMap_MLR = dataMap_MLR.fillna(0)
            mapSM_MLR = MLRmodel.predict(dataMap_MLR)
            ## debo invertir la funcion flatten()
            mapSM_MLR = mapSM_MLR.reshape(r,c)
            mapSM_MLR  = 10**(mapSM_MLR)
            mapSM_MLR[mapSM_MLR < 0] = 0

            mapSM_MLR = mapSM_MLR*maskNDVI#*maskCity

            OldRange = (np.max(band_matchTs)  - np.min(band_matchTs))
            NewRange = (1 + 1)
            Ts = (((band_matchTs - np.min(band_matchTs)) * NewRange) / OldRange) -1

            OldRange = (np.max(band_matchEt)  - np.min(band_matchEt))
            NewRange = (1 + 1)
            Et = (((band_matchEt - np.min(band_matchEt)) * NewRange) / OldRange) -1

            OldRange = (np.max(band_matchHR)  - np.min(band_matchHR))
            NewRange = (1 + 1)
            HR = (((band_matchHR - np.min(band_matchHR)) * NewRange) / OldRange) -1

            OldRange = (np.max(band_matchPP)  - np.min(band_matchPP))
            NewRange = (1 + 1)
            PP = (((band_matchPP - np.min(band_matchPP)) * NewRange) / OldRange) -1

            OldRange = (np.max(sarEnmask2)  - np.min(sarEnmask2))
            NewRange = (1 + 1)
            sar2 = (((sarEnmask2 - np.min(sarEnmask2)) * NewRange) / OldRange) -1



            #dataMap_MLP = pd.DataFrame({'Sigma0' :sar2.flatten(), 'T_s' :Ts.flatten(), 'PP' :PP.flatten()})
            #dataMap_MLP = dataMap_MLP[['Sigma0', 'T_s', 'PP']]

            #dataMap_MLP = pd.DataFrame({'HR' :HR.flatten(), 'PP' :PP.flatten(),'Sigma0' :sar2.flatten(), 'T_s' :Ts.flatten()})
            #dataMap_MLP = dataMap_MLP[['HR', 'PP','Sigma0', 'T_s']]

            dataMap_MLP = pd.DataFrame({'Et' :Et.flatten(),'Sigma0' :sar2.flatten(), 'T_s' :Ts.flatten(), 'PP' :PP.flatten()})
            dataMap_MLP = dataMap_MLP[['Et', 'PP','Sigma0', 'T_s']]

            ### , 'e_a': band_matchEa.flatten(), , ,
            #print dataMap_MLP
            ###.describe()
            #dataMap_MLP = dataMap_MLP.fillna(0)
            mapSM_MLP = MLPmodel.predict(dataMap_MLP)

            mapSM_MLP = mapSM_MLP.reshape(r,c)
            #print mapSM_MLR.shape
            mapSM_MLP[mapSM_MLP < 0] = 0

            #fig, ax = plt.subplots()
            #ax.imshow(maskNDVI, interpolation='None',cmap=cm.gray)
            #plt.show()

            mapSM_MLP = mapSM_MLP*maskNDVI

            #functions.createHDFfile(path, nameFileMLR+"_HR", 'ENVI', mapSM_MLR, c, r, GeoTSar, ProjectSar)
            #functions.createHDFfile(path, nameFileMLP+"_HR", 'ENVI', mapSM_MLP, c, r, GeoTSar, ProjectSar)
            functions.createHDFfile(path, nameFileMLR+"_ET", 'ENVI', mapSM_MLR, c, r, GeoTSar, ProjectSar)
            functions.createHDFfile(path, nameFileMLP+"_ET", 'ENVI', mapSM_MLP, c, r, GeoTSar, ProjectSar)
    print "FIN"


