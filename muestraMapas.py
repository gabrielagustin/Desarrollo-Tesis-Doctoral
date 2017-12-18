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
        lc = srcband.ReadAsArray(xoff, yoff, int(xcount)+1, int(ycount)+1)
    return src_ds, lc, gt1, Project

def applyNDVIfilter(sar,L8):
    #print sar.shape
    #print L8.shape
    result = sar
    rSar, cSar = sar.shape
    mask = np.ones((rSar, cSar))
    #print cFactor
    for i in range(0, rSar):
        for j in range(0, cSar):
            if (L8[i, j] > 0.45 ): mask[i,j] = 0
            #if (L8[i, j] < 0.1 ): mask[i,j] = 0
            if (L8[i, j] < 0.2 ): mask[i,j] = 0
    result = result*mask
    return result, mask


def applyWaterfilter(sar,modis):
    #print sar.shape
    rSar, cSar = sar.shape
    result = np.zeros((rSar, cSar))
    #print modis.shape
    rModis, cModis = modis.shape
    rFactor = rSar/float(rModis)
    #print rFactor
    cFactor = cSar/float(cModis)
    #print cFactor
    for i in range(0, rSar):
        for j in range(0, cSar):
            indexi = int(i/rFactor)
            indexj = int(j/cFactor)
            if (modis[indexi, indexj] < 0.1): result[i,j] = 1
    r,c = sar.shape
    mask = np.ones((r,c))
    mask = mask - result
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



if __name__ == "__main__":

    dir = "ggarcia"

    fechaSMAP = []
    fechaSentinel = []
    fechaNDVI = []
    fechaMYD = []


    fechaSMAP.append("2015-04-11")
    fechaSentinel.append("2015-04-08")
    fechaNDVI.append("2015-04-07")
    fechaMYD.append("2015-04-06")
    i=0
    fileSmap = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[i]+"/recorte/SM.img"
    #src_ds_Smap, bandSmap, GeoTSmap, ProjectSmap = functions.openFileHDF(fileSmap, 1)

    fileEt = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MYD16/"+fechaMYD[i]+"/MYD16A_reprojected.data/ET_500m.img"
    #src_ds_Et, bandEt, GeoTEt, ProjectEt = functions.openFileHDF(fileEt, 1)

    fileTs = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[i]+"/subset_reprojected.data/surface_temperature.img"
    #src_ds_Ts, bandTs, GeoTTs, ProjectTs = functions.openFileHDF(fileTs, 1)

    filePP = "/media/"+dir+"/TOURO Mobile/GPM/"+fechaSentinel[i]+"/recorte.img"
    #src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)

    fileNDVI = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/"+fechaNDVI[i]+"/NDVI_reproyectado_recortado"


    fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/Sigma0_VV_db.img"
    src_ds_Sar, bandSar, GeoTSar, ProjectSar = functions.openFileHDF(fileSar, 1)
    r1 = geospatial_coor(fileSar)
    #print r1
    #r2 = geospatial_coor(fileModis)
    r2 = geospatial_coor(fileEt)
    #print r2
    r3 = geospatial_coor(fileTs)
    r4 = geospatial_coor(filePP)
    #print r3
    intersection = [max(r1[0], r2[0], r3[0], r4[0]), min(r1[1], r2[1], r3[1], r4[1]), min(r1[2], r2[2], r3[2], r4[2]), max(r1[3], r2[3], r3[3], r4[3])]
    print "coordenadas de interseccion: "
    print intersection

    src_dsSmap, Smap, GeoTSmap, ProjectSmap = openImage(fileSmap,intersection)
    src_dsSAR, sar, GeoTSAR, ProjectSAR = openImage(fileSar,intersection)

    src_dsEt, Et, GeoTEt, ProjectEt = openImage(fileEt,intersection)
    src_dsTs, Ts, GeoTTs, ProjectTs = openImage(fileTs,intersection)
    src_dsPP, PP, GeoTPP, ProjectPP = openImage(filePP,intersection)

    src_dsNDVI, NDVI, GeoTNDVI, ProjectNDVI = openImage(fileNDVI,intersection)

    fig, ax = plt.subplots()
    im3 = ax.imshow(Smap, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig, ax = plt.subplots()
    im3 = ax.imshow(sar, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig, ax = plt.subplots()
    im3 = ax.imshow(Et*0.1, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig, ax = plt.subplots()
    im3 = ax.imshow(Ts, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig, ax = plt.subplots()
    im3 = ax.imshow(PP, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig, ax = plt.subplots()
    im3 = ax.imshow(NDVI, interpolation='None',cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.show()