# Feature Extraction through transfer learning
# save bottleneck features
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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D,MaxPool1D,GlobalAveragePooling2D,GlobalMaxPool1D,GlobalMaxPooling1D,BatchNormalization,Activation, Dropout, Flatten, Dense,LeakyReLU,TimeDistributed,Concatenate,Reshape,Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import tensorflow_probability as tfp
import histomicstk as htk
from tensorflow.keras.models import load_model

# %%
##Color Normalization
#---ref_img_path: path of reference image
#example: '/Image/WSI/TCGA-AN-A0FK-01Z-00-DX1.8966A1D5-CE3A-4B08-A1F6-E613BEB1ABD1_2_4.jpg'
#---inputimg: RGB image
def color_norm(inputimg, ref_img_path,
stain_unmixing_routine_params= {'stains': ['hematoxylin', 'eosin'],
'stain_unmixing_method': 'macenko_pca'}):
    refimg=skimage.io.imread(ref_img_path)[:,:,:3]
    mask_img = np.dot(inputimg[...,:3], [0.299, 0.587, 0.114])
    mask_img=np.where(mask_img<=215,False,True)
    img_norm=htk.preprocessing.color_normalization.deconvolution_based_normalization(inputimg,
    im_target=refimg,stain_unmixing_routine_params=stain_unmixing_routine_params,mask_out=mask_img)
    return img_norm


# %%
def Filter(string, substr): 
    fil=[any([bool(re.search(sub,st)) for sub in substr]) for st in string]
    filtered=list(compress(string,fil))
    return filtered

# %%
#Pick defined number of tiles from folder and concatenate to 4D array#
#---path: the path of image folder of one patient
#---allimage_perpatient: when set to False, set num_img number to randomly pick the number of images from one folder
#---image_num_most: the largest number allowed for one patient
def get_input(path,num_img=1,allimage_perpatient=True,image_num_most=64):
    images=Filter(os.listdir(path),['jpg'])#----image candidates list
    if allimage_perpatient==True:
        num_img=min(image_num_most,len(images))
    img=np.zeros(shape=(num_img,256,256,3))#----create empty array
    images=rand.sample(images,num_img)
    for i in range(num_img):
        img0=plt.imread(path+'/'+images[i])
        img[i,:,:,:]=img0
    return(img)





# %%
#feature extractor: Densenet-121
#extract all patients' features from a dataframe

#---WSI_df: a pandas dataframe with all patients' patches will be extracted, 
#should contain column of Barcode_Path with all the paths of patients' folder
#example: /Image/Patch/TCGA-3C-AALJ
#---target: path of features to be stored
#example: '/Bottlenect_Features/features_densenet121.npy'

def Densenet121_extractor(WSI_df,target):
    #define feature extractor CNN model
    Densenet_base=tf.keras.applications.DenseNet121(include_top=False, input_shape=(256,256,3),weights='imagenet')
    for layer in Densenet_base.layers:
        layer.trainable=False
    inputs=tf.keras.Input(shape=(None,256,256,3))
    x=TimeDistributed(Densenet_base)(inputs)
    x=TimeDistributed(GlobalAveragePooling2D())(x)
    x=GlobalMaxPooling1D()(x)
    model=Model(inputs=[inputs],outputs=[x])
    #extract features
    features=[]
    for i in range(WSI_df.shape[0]):
        print('image'+str(i),end=', ')
        img=get_input(WSI_df.Barcode_Path[i],image_num_most=256)
        #color normalization before transfer learning
        for j in range(img.shape[0]):
            img[j,:,:,:]=color_norm(inputimg=img[j,:,:,:])
        img=img/255
        img=np.expand_dims(img,axis=0)
        feature=model.predict(img)[0,:]
        features.append(feature)
    np.save(target,features)






