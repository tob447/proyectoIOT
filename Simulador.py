#manejo de elemeotos del sistema, como carpetas
import scipy.io as sio;
import numpy as np;
from numpy.linalg import matrix_rank
import glob
import os
import base64


# General imports
import matplotlib.pyplot as plt


#importamos la rutina de welch
from scipy.signal import welch as pwelch
from scipy.signal import butter, lfilter



# Import MNE, as well as the MNE sample dataset
import mne

from fieldtrip2mne import read_epoched

# FOOOF imports

from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum



import numpy as np
import matplotlib.pyplot as plt
import os

import itertools
import tensorflow as tf
# Setting seed for reproducibility
n_seed = 0

from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, LSTM,BatchNormalization,Bidirectional


from tensorflow.keras import initializers

from urllib.parse import urlencode
from urllib.request import Request, urlopen

import requests

def get_age(x):
  low,high = x.split('-')
  low = int(low)
  high = int(high)
  return np.mean([low,high])

def get_age_group(x):
  if x >= 60:
    return 1 #'AdultoMayor'
  else:
    return 0 #'Adulto'

def powers(senal_continua,fs):
  nperseg = int(fs*2)
  noverlap = int(nperseg/2)
  max_frequency = 40
  potencia = np.zeros((19,5))

  #%% USANDO WELCH sobre todos los canales
  f, Pxx = pwelch(senal_continua, fs, 'hanning', nperseg, noverlap)
  f=f[0,:]
  deltaIPEG = np.sum(Pxx[:,((f <= 6)& (f > 1.5))],axis=1) #1.5-6
  thetaIPEG = np.sum(Pxx[:,(f <= 8.5) & (f > 6)],axis=1)
  alfaIPEG = np.sum(Pxx[:,(f <= 12.5) & (f > 8.5)],axis=1)
  betaIPEG = np.sum(Pxx[:,(f <= 30) & (f > 12.5)],axis=1)
  gammaIPEG = np.sum(Pxx[:,(f <= 40) & (f > 30)],axis=1)

  total = np.sum(Pxx[:,(f <= max_frequency) & (f > 1.5)],axis=1)
  deltaIPEG_relativo = deltaIPEG/total
  thetaIPEG_relativo = thetaIPEG/total
  alfaIPEG_relativo = alfaIPEG/total
  betaIPEG_relativo = betaIPEG/total
  gammaIPEG_relativo = gammaIPEG/total

  potencia[:,0] = deltaIPEG_relativo
  potencia[:,1] = thetaIPEG_relativo
  potencia[:,2] = alfaIPEG_relativo
  potencia[:,3] = betaIPEG_relativo
  potencia[:,4] = gammaIPEG_relativo

  return (potencia)

def get_model_data(set_path,band=(12,30)):
  dummyset = sio.loadmat(set_path)
  #print(dummyset.keys())
  nchannels_set,npoints_set,nepochs_set = np.dstack(dummyset['segments']).shape
  dummyset_continuo = np.hstack(dummyset['segments'])
  dummyset_S_continuo = dummyset['W'] @ (dummyset_continuo)
  #dummyset_S_continuo = dummyset_continuo
  #print(nchannels_set,npoints_set,nepochs_set)
  dummyset_S_epocas = dummyset_S_continuo.reshape((dummyset_S_continuo.shape[0],npoints_set,nepochs_set),order='F')
  print(dummyset_S_epocas.shape)
  
  ch_types = ['eeg'] * dummyset_S_continuo.shape[0]
  dummyset_info = mne.create_info(dummyset_S_continuo.shape[0], sfreq=dummyset['sfreq'], ch_types=ch_types)
  dummy_raw = mne.io.RawArray(dummyset_S_continuo, dummyset_info)
  #dummy_raw.plot(show_scrollbars=False, show_scalebars=False)
  #dummy_raw.plot_psd()
  #comp_filter = dummy_raw.copy().filter(*band)
  #comp_filter = comp_filter.copy().resample(sfreq=128)
  #comp_filter.plot_psd()
  comp_filter = dummy_raw
  epocas = mne.make_fixed_length_epochs(comp_filter, duration=2, preload=True)
  comp = list(np.arange(epocas._data.shape[1]).astype(int))
  #[3,4,5,6,7,8,9,10,11,12,14,16,17,18]
  #segments = [wavelet_decomposition(epocas._data[e,comp,:])['alpha'] for e in range(epocas._data.shape[0])]
  segments = [powers(epocas._data[e,comp,:],dummyset['sfreq']) for e in range(epocas._data.shape[0])]
  #print(dummyset.keys())
  print(segments[0].shape)
  age = dummyset['Age'].tolist()[0]
  #print(age)
  age = get_age(age)
  age = get_age_group(age)
  #print(age)
  return segments,[age]*len(segments)

def traindata(archivos):
  train_filename = './train_data.mat'
  overwrite = True
  if not os.path.isfile(train_filename) or overwrite:
    ejemplos = []
    grupos = [] 
    for i,archivo in enumerate(archivos[:10]):
      print('**********************************')
      print(i,archivo)
      epocas,edades=get_model_data(archivo)
      ejemplos += epocas
      grupos += edades
      print('**********************************')
    X = np.array(ejemplos)
    Y = np.array(grupos)
    sio.savemat(train_filename,{'X':X,'Y':Y})
  else:
    print('File existed')
    train_data = sio.loadmat(train_filename)
    X = train_data['X']
    Y = np.squeeze(train_data['Y'])
  return X,Y

def balance(X,Y):
  X=X/np.max(X)
  Y_1 = np.sum(Y)
  Y_0 = Y.shape[0] - Y_1
  Y_1p = 100*Y_1/Y.shape[0]
  Y_0p = 100-Y_1p

  print('Y_1',Y_1p,Y_1)
  print('Y_0',Y_0p,Y_0)
  N_remove = np.abs(Y_0 - Y_1)
  print(N_remove)
  N_keep = np.min([Y_0,Y_1])
  print(N_keep)
  idx_0 = np.where(Y==0)
  idx_0
  assert np.sum(Y[idx_0])==0
  idx_1 = np.where(Y==1)[0]
  tf.random.set_seed(n_seed)
  rng = np.random.default_rng(n_seed)
  idx_to_keep = rng.choice(idx_0[0], N_keep, replace=False)
  assert N_keep == len(set(idx_to_keep.tolist()))
  idx_to_keep=np.sort(np.concatenate([idx_to_keep,idx_1]))
  assert np.sum(Y[idx_to_keep])==Y_1 # Adultos mayores antes == Adultos mayores despues
  assert np.sum(Y[idx_to_keep]==0)==Y_1 # Adultos mayores antes == Adultos despues
  X_ = X[idx_to_keep,:]
  Y_ = Y[idx_to_keep]
  X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, stratify=Y_,random_state=n_seed)

  return X_train, X_test, Y_train, Y_test

def inModel(Y_train,Y_test):
  train_1 = 100*np.sum(Y_train)/Y_train.shape[0]
  train_0 = 100-train_1
  test_1 = 100*np.sum(Y_test)/Y_test.shape[0]
  test_0 = 100-test_1
  print('Train:',train_1,train_0)
  print('Test:',test_1,test_0)

def toJSON(model):
  # serialize model to JSON
  model_json = model.to_json()
  with open('modelo.json', "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights('modelo.h5')
  print("Saved model to disk")

def loadModel(path):
  # load json and create model
  json_file = open(path+'.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(path+'.h5')
  print("Loaded model from disk")
  return loaded_model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig=plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# compute precision and recall
def computerprecision(test_label,classes_x):
  precision_test = precision_score(test_label,classes_x)
  recall_test = recall_score(test_label, classes_x)
  f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
  print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )

def vs(scores_test_Pre,scores_test_Cru):

  plt.figure(figsize = (12,7))  
  ## Declaramos valores para el eje x
  x_labels = ['Preprocessed','Raw']  
  ## Declaramos valores para el eje y
  eje_y = [scores_test_Pre[1]*100,scores_test_Cru[1]*100]

  plt.bar(x_labels, eje_y, width= 0.9, align='center',color='0.8')
  for i in range(len(x_labels)):
      plt.annotate(round(eje_y[i]),xy=(i,eje_y[i]),color='Black', weight='bold', ha='center',  size=20)
  plt.legend(labels = ['Accuracy percentage'])
  plt.title("Dataset")
  plt.xlabel('Data')
  plt.ylabel('Accurracy')
  plt.ylim((90,100))
  # Saving the plot as a 'png'
  plt.savefig('vs.png')

def crear_modelo(train_X):
  tf.random.set_seed(n_seed)
  kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=(n_seed))
  nb_features = np.atleast_3d(train_X).shape[2]
  sequence_length = np.atleast_3d(train_X).shape[1]

  model = Sequential()

  model.add(Bidirectional(LSTM( units=256,
          return_sequences=True,kernel_initializer=kernel_initializer),
          input_shape=(sequence_length,nb_features),
          ))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(LSTM(
          units=128,
          return_sequences=True,kernel_initializer=kernel_initializer))

  model.add(BatchNormalization())
  model.add(LSTM(
            units=64,
            return_sequences=False,kernel_initializer=kernel_initializer))

  model.add(BatchNormalization())
  model.add(Dense(units=32,kernel_initializer=kernel_initializer))
  model.add(Dense(units=1, activation='sigmoid',kernel_initializer=kernel_initializer))
  tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  print(model.summary())

  toJSON(model)
    
  return model

def grafica(model,test_X,test_label):
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict_x=model.predict(test_X) 
    classes_x=(predict_x>= 0.5).astype(int)
    cm_test = confusion_matrix(test_label,classes_x)
    computerprecision(test_label,classes_x)
    # Plot non-normalized confusion matrix
    class_names=['AdultoMayor','Adulto']
    plt.figure()
    plot_confusion_matrix(cm_test, classes=class_names,
                          title='Confusion_matrix')
    plt.savefig('Confusion_matrix.png')

def rename(X_train, X_test, Y_train, Y_test):
  train_X = X_train
  train_label = Y_train
  test_X = X_test
  test_label = Y_test
  print(train_X.shape)
  print(train_label.shape)
  print(test_X.shape)
  print(test_label.shape)
  return train_X,test_X,train_label,test_label

def ejecutar():
  path_archivos = './Datos_mat_Preprocesados/*.mat'
  archivos = glob.glob(path_archivos)

  X,Y = traindata(archivos)
  '''
  X_train, X_test, Y_train, Y_test = balance(X,Y)
  inModel(Y_train,Y_test)
  train_X,test_X,train_label,test_label = rename(X_train, X_test, Y_train, Y_test)
  if model == None:
    model=crear_modelo(train_X)
  else:
    '''
  model =loadModel('./modelo') 
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X, Y,validation_split=0.1,epochs=5, batch_size=32,verbose=1)
  toJSON(model)
  grafica(model,X, Y)
    
def returnImageBase64():
  with open('Confusion_matrix.png', "rb") as image_file:
        data = base64.b64encode(image_file.read())
    
  base64String=str(data)
  return base64String[2:len(base64String)-1]
