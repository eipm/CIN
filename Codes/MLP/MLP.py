# Train top MLP and evaluate model performance

# %%
import numpy as np
import pandas as pd
import random as rand
import skimage
from skimage import io,feature, filters,color
from skimage.exposure import rescale_intensity
import re
import os
import shutil
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras import applications,optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D,MaxPool1D,GlobalAveragePooling2D,GlobalMaxPool1D,GlobalMaxPooling1D,BatchNormalization,Activation, Dropout, Flatten, Dense,LeakyReLU,TimeDistributed,GlobalAveragePooling1D,Concatenate,Reshape,Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.model_selection import train_test_split

# %%
#---input: 
#features: patient level features
#WSI_df: summarise file
def feature_split(WSI_df,features,ID,Label,testratio,seed):
    X_train,X_test,y_train,y_test,ID_train,ID_test=train_test_split(features,list(WSI_df[Label]),list(WSI_df[ID]),test_size=testratio,random_state=seed)
    #normalization
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train=np.array([int(y) for y in y_train])
    y_test=np.array([int(y) for y in y_test])
    return(X_train,X_test,y_train,y_test,ID_train,ID_test)



# %%
#train top MLP
#---WSI_df: summarise file, a dataframe
#must have columns indicating: label, patient ID
#---Label, ID: 
#column names in WSI_df. example: ID='Barcode',Label='Cin_Label'
#---features:
#patient level feature array. 
#---testratio: percentage of test dataset, default is 0.15
#---seed, default is 1001
#---Model_Name:
#name to store the model, example: 'Model.h5'
#---Learning parameters:
#layer_num,nodes_num_1,nodes_num_2,dropout,lr
#you can set up to 2 hidden layers. Too many hidden layers is not likely to have good performance by experiments. 
#use layer_num to set number of hidden layers
#use nodes_num_1 and nodes_num_2 to set nodes number of two hidden layers. Only set nodes_num_2 when layer_num=2
#you can set up dropout and learning rate. default setting: dropout=0,lr=0.00001
def MLP_train(WSI_df=WSI_df,features=features,ID='Barcode',Label='Cin_Label',testratio=0.15,seed=1001,
Model_Name='Model.h5',layer_num=1,nodes_num_1=1024,nodes_num_2=512,dropout=0,lr=0.00001):
    #split
    X_train,X_test,y_train,y_test,ID_train,ID_test=feature_split(WSI_df=WSI_df,features=features,
    ID=ID,Label=Label,testratio=testratio,seed=seed)
    #build MLP
    if layer_num==1:
        MLP = Sequential()
        MLP.add(Dense(nodes_num_1,input_shape=(1024,),kernel_initializer=tf.keras.initializers.he_normal(),activation='relu'))
        MLP.add(Dropout(dropout))
        MLP.add(Dense(1,activation='sigmoid'))
    if layer_num==2:
        MLP = Sequential()
        MLP.add(Dense(nodes_num_1,input_shape=(1024,),kernel_initializer=tf.keras.initializers.he_normal(),activation='relu'))
        MLP.add(Dropout(dropout))
        MLP.add(Dense(nodes_num_2,kernel_initializer=tf.keras.initializers.he_normal(),activation='relu'))
        MLP.add(Dense(1,activation='sigmoid'))
    #compile
    MLP.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
    #train
    es=EarlyStopping(monitor='val_loss',mode='min',patience=200)
    mc = ModelCheckpoint(Model_Name, monitor='val_loss', mode='min', verbose=1,save_best_only=True)
    history=MLP.fit(
        X_train,y_train,
        validation_split=0.15,
        epochs=2500,
        callbacks=[es,mc])