import os
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import seaborn as sns

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers as layers

from mpl_toolkits.basemap import Basemap

import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity

from scipy.stats import gaussian_kde

#---------------------------------------------------------------------

def data_preprocessor(train_data, test_data):
    
    # MinMax scaler
    scaler = preprocessing.MinMaxScaler().fit(train_data[0])
    
    scaled_data = []
    for i in train_data:
        scaled_data.append(scaler.transform(i))
    
    scaled_test_data = []
    for i in test_data:
        scaled_test_data.append(scaler.transform(i))
    
    x_train_scaled = np.array(scaled_data)
    x_test_scaled = np.array(scaled_test_data)
    
    return x_train_scaled, x_test_scaled

#---------------------------------------------------------------------

def label_preprocessor(train_label, test_label):
    
    scaler_label = preprocessing.MinMaxScaler().fit(train_label)
    #scaler_label = preprocessing.StandardScaler().fit(train_label)
    
    y_train_scaled = scaler_label.transform(train_label)
    y_test_scaled = scaler_label.transform(test_label)
    
    return y_train_scaled, y_test_scaled, scaler_label

#---------------------------------------------------------------------

def tf_atan2(y, x):

    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle

#---------------------------------------------------------------------

def tf_haversine(latlon1, latlon2):

    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]

    REarth = 6371
    lat = tf.abs(lat1 - lat2) * np.pi / 180
    lon = tf.abs(lon1 - lon2) * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    a = tf.sin(lat / 2) * tf.sin(lat / 2) + tf.cos(lat1) * tf.cos(lat2) * tf.sin(lon / 2) * tf.sin(lon / 2)
    d = 2 * tf_atan2(tf.sqrt(a), tf.sqrt(1 - a))
    return REarth * d

#---------------------------------------------------------------------

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


#---------------------------------------------------------------------

def plot_loss(loss, val_loss, accuracy, val_accuracy):
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss', fontsize=17)
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['Train', 'Val'], loc='upper right', fontsize=12)
    
    plt.subplot(122)
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('Model accuracy', fontsize=17)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['Train', 'Val'], loc='lower right', fontsize=12)
    #plt.savefig('all_2conv_37epocch_2.4h_loc.png')
    plt.show()

#---------------------------------------------------------------------

def distance(s_lat, s_lng, e_lat, e_lng):

   # approximate radius of earth in km
   R = 6373.0

   s_lat = s_lat*np.pi/180.0                      
   s_lng = np.deg2rad(s_lng)     
   e_lat = np.deg2rad(e_lat)                       
   e_lng = np.deg2rad(e_lng)  

   d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

   return 2 * R * np.arcsin(np.sqrt(d))

#---------------------------------------------------------------------

def get_contour(dataframe): 
    
    x = dataframe.long_pred.values
    y = dataframe.lat_pred.values

    k = gaussian_kde(np.vstack([x, y]))

    xi, yi = np.mgrid[-15:95:x.size**1*1j,30:72:y.size**1*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi =zi.reshape(xi.shape)

    #set up plot
    origin = 'lower'
    levels = [0.05, 0.5]
    
    return xi, yi, zi, levels, origin