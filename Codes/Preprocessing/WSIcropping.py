# Image Preprocessing
# Crop WSI into nonoverlapping patches
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
import imageio
import shutil
import matplotlib.pyplot as plt 
import tensorflow as tf
import histomicstk as htk


# %%
#function of filter a list: for a given list, only keep elements containing substr
def Filter(string, substr): 
    fil=[any([bool(re.search(sub,st)) for sub in substr]) for st in string]
    filtered=list(compress(string,fil))
    return filtered



# %%
#2.5x magnification (2048x2048)---tile to 64 nonoverlapping patches(256x256)
#keep patches with tissue percentage>pct (default is 0.8)
#---inputdir: dictionary of WSI
#example: /Image/WSI/
#---targetdir: dictionary of patches
#example: /Image/Patch/ 
def WSIcropping(inputdir,targetdir,pct=0.8):
    wsi_list=Filter(os.listdir(inputdir),['jpg'])
    for index in range(len(wsi_list)):
        print(wsi_list[index],end=', ')
        img=plt.imread(inputdir+ wsi_list[index])
        pattern=re.compile(r"_\d{1,2}_\d{1,2}.jpg")
        img_samplename=re.sub(pattern,'',wsi_list[index])

        for i in range(8):
            for j in range(8):
                sub_img=img[j*256:(j+1)*256,i*256:(i+1)*256,:]
                if filtering(sub_img,40)>pct:
                    temp=targetdir + img_samplename[:12]
                    if not os.path.isdir(temp):
                        os.makedirs(temp)
                    imageio.imwrite(temp+'/'+img_samplename+'_'+str(i)+'_'+str(j)+'.jpg',sub_img) 
