# Image Preprocessing
# 
# %%
import numpy as np
import pandas as pd
import random as rand
from itertools import compress
import skimage
from skimage import io,feature, filters,color
from skimage.exposure import rescale_intensity
import re
import os
import shutil
import matplotlib.pyplot as plt 
import tensorflow as tf
import histomicstk as htk


# %%
#calculate image tissue percentage
def filtering(rgb,thresh_min):
    rgb = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    rgb[:] = [255 - x for x in rgb]
    binary_min = rgb > thresh_min
    return np.sum(binary_min)/float(binary_min.size)

### RGB to HED space. Choose higher H
def HED_filter(rgbimage,threshold=0.5):
    hedimg=color.rgb2hed(rgbimage)[:,:,0]
    h = rescale_intensity(hedimg, out_range=(0, 1))
    h[h < threshold]=0
    return 2*np.sum(h)/h.size


# %%
###Tiling WSI ###
##input: raw svs. files
##output: the best window consists highest tissue percentage in jpg



#---slide_index: index number in filepaths

#---filepaths: list of all svs file paths 
#example: '/Image/SVS/TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A.svs'

#---samplenames: list of sample names without
#example: 'TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A'

#---tile_dir: director to store jpg tiles
#example: '/Image/WSI/'

#---tilesize, overlap: dimension of output is tilesize+2*overlap
#---low_mpp: target mpp. Default is 4 (2.5x magnification)

def tiling_wsi(slide_index, filepaths=filepaths, samplenames=samplenames,
tile_dir=tile_dir_lowreso,tilesize=512,overlap=768,low_mpp=4,save=True):
    tile = [] #save output tile into list
    cell_pct_list = [0] #cell percentage
    h_score=[0]
    ij=[0] #memorize tile location
    slide = openslide.open_slide(filepaths[slide_index])
    #high_mag=int(slide.properties['aperio.AppMag']) #max mag
    high_mpp=round(float(slide.properties['aperio.MPP']),2)
    data_gen = DeepZoomGenerator(slide, tile_size = tilesize, overlap = overlap)
    num_levels = data_gen.level_count #count levels
    lowresolution_level=int(num_levels-int(np.log2(int(low_mpp/high_mpp)))-1)

    #count tiles
    print ('Levels: '+str(num_levels)+'\n'+'Tile Counts: '+ str(data_gen.level_tiles[lowresolution_level]))
    (num_w,num_h)=data_gen.level_tiles[lowresolution_level]
    for i in range(1,num_w-1):
        for j in range(1,num_h-1):
            img = np.array(data_gen.get_tile(lowresolution_level, (i,j)))
            if img.shape[0]==img.shape[1]:
                c_pct=filtering(img,40)
                if c_pct>cell_pct_list[0] and c_pct<0.9:
                    cell_pct_list[0]=c_pct
                    ij[0]=(i,j)
                    tile=img
                if c_pct>=0.9:
                    cell_pct_list[0]=c_pct
                    H=HED_filter(img,0.5)
                    if H>h_score[0]:
                        h_score[0]=H
                        ij[0]=(i,j)
                        tile=img
    #save jpg file
    if save==True:
        imageio.imwrite(tile_dir + samplenames[slide_index] + '_'+str(ij[0][0])+'_'+str(ij[0][1])+'.jpg',tile) #---jpg file
    else:
        return(tile)
    slide.close()





