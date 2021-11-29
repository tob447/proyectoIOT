#manejo de elemeotos del sistema, como carpetas
import scipy.io as sio;
import numpy as np;
from numpy.linalg import matrix_rank
import glob
import os
import copy
import pickle
import h5py
from pandas import HDFStore


# General imports
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
import pandas as pd


#importamos la rutina de welch
from scipy.signal import welch as pwelch
from scipy.signal import butter, lfilter


# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
import tarfile
from fieldtrip2mne import read_epoched

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum
import wget 

import numpy as np
from scipy.stats import mode

import traceback

def dowload_links(links):
  print(links)
  links = [links]
  print(links)
  path_act = os.getcwd()
  prep_path = path_act + './Datos_Originales_Preprocesados/'
  os.makedirs(prep_path,exist_ok=True)
  for l in links:
    filename = prep_path + os.path.split(l)[-1]
    if not os.path.isfile(filename):
      print(filename,'ok')
      try:
        wget.download(l,filename) 
      except Exception:
        continue
    else:
      print(filename,'already existed')
  return path_act



def descomprimir(archivos,folder):
  errores = []
  for g in range(len(archivos)):
    path = './'+folder+'./'
    name = path+os.path.split(archivos[g])[-1].replace('.tar.gz','')
    path_dir = os.path.isdir(name)
    if not path_dir:
      print('Descomprimiendo')
      try:
        file = tarfile.open(archivos[g]) 
        file.extractall(path) 
        file.close() 
      except:
        print('Error con',archivos[g])
        errores.append(archivos[g])

    else:
      print('Ya existe',name)
  return errores

def select_1_event(all_data,events,desired):
    """
    
    Parameters
    ----------
    all_data: Matriz canalesxframes
    events: lista de tuplas tal como se obtiene de mne.events_from_annotations
            (ultimo lugar de la tupla es el identificador del evento)
    desired: identificador del evento que quieres

    Returns
    -------
    lista con las secciones del evento deseado 
        [arreglos[canales,frame], ...]
    """
    segments = []
    desired=210
    for i in range(1,len(events)-1):
        this = events[i][2]
        next = events[i+1][2]
        if this == desired and next == desired:
            segments.append(all_data[:,events[i][0]:events[i+1][0]])

    len_trend = mode([x.shape[-1] for x in segments]).mode[0]

    for i in range(len(segments)):
        if segments[i].shape[-1] > len_trend:
            segments[i] = segments[i][:,:len_trend]
        if segments[i].shape[-1] <len_trend:
            segments[i] = None
    return [x for x in segments if x is not None ]

def fit_spatial_filter(M,all_channels,keep_channels,mode='demixing'):
  """
  # asumir que all_channels tiene el orden original
  # asumir que keep_channels tiene el orden deseado

  all_channels = ['FP1',  'FPZ',  'FP2',  'AF3',  'AF4',  'F7',  'F5',  'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',  'FT7',  'FC5',  'FC3',  'FC1',  'FCZ',  'FC2',  'FC4',  'FC6',  'FT8',  'T7',  'C5',  'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',  'TP7',  'CP5',  'CP3',  'CP1',  'CPZ',  'CP2',  'CP4',  'CP6',  'TP8',  'P7',  'P5',  'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',  'PO7',  'PO5',  'PO3',  'POZ',  'PO4',  'PO6',  'PO8',  'I1',  'O1',  'OZ',  'O2',  'I2']
  keep_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
  fuentes = 19
  A = np.array([[x]*fuentes for x in all_channels],dtype=object)
  W = A.T
  print(indexes)
  print(A)
  print(W)
  """
  # asumir que all_channels tiene el orden original
  # asumir que keep_channels tiene el orden deseado
  indexes = [all_channels.index(x) for x in keep_channels]

  if mode == 'demixing':
    return M[:,indexes]
  else:
    return M[indexes,:]

def preprocesamiento_lemon(filepath,event=210,resample=500,W_lab=None,lab_channels=None,additional_info=None):
  """
  Orden de los canales dado por el filtro espacial del lab, se saca la interseccion con cada eeg al que se le aplique
  event es el evento que se quiere dejar (solo 1 por ahora) (valido para vhdr sin procesar)
  """
  if '.vhdr' in filepath:
    raw=mne.io.read_raw_brainvision(filepath,preload=True)
  elif '.set' in filepath:
    raw=mne.io.eeglab.read_raw_eeglab(filepath,preload=True)
    print('Making fixed lenght epochs of 2 seconds')
    raw = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
  
  # Manejo de canales a los deseados
  raw = raw.rename_channels( lambda x: x.upper())
  if W_lab is not None and lab_channels is not None:
    intersection = list(set(lab_channels).intersection(raw.ch_names))
    intersection.sort()
    #A_lemon = fit_spatial_filter(A,lab_ch_names,intersection,mode='mixing')
    W_lemon = fit_spatial_filter(W_lab,lab_channels,intersection,mode='demixing')
    print(raw.ch_names)
    raw = raw.drop_channels(list(set(raw.ch_names)-set(intersection)))
    print(raw._data.shape)
    raw = raw.reorder_channels(intersection)
    print(raw._data.shape)
    

  # REMUESTREAR?
  if resample is not None:
    raw = raw.copy().resample(sfreq=500)
  if '.vhdr' in filepath:
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    segments = select_1_event(raw.get_data(),events_from_annot,event)
  elif '.set'  in filepath:
    segments = [raw._data[e,:,:] for e in range(raw._data.shape[0])]
  data_epochs = np.dstack(segments) *1e6#canales,puntos,epocas
  data_continuous = np.hstack(segments) *1e6#canales,puntos
  nchannels,npoints,nepochs=data_epochs.shape
  print('nchannels,npoints,nepochs')
  print(nchannels,npoints,nepochs)
  data = {'segments':segments,
          'data_epochs':'np.dstack(segments) #canales,puntos,epocas',
          'data_continuous': 'np.hstack(segments) #canales,puntos',
          'S_continuous' : 'W @ (data_continuous)',
          'S_epochs' : 'S_continuous.reshape((S.shape[0],npoints,nepochs),order=\'F\')',
          'ch_names':copy.deepcopy(raw.ch_names),
          'sfreq':raw.info['sfreq'],
          'W':W_lemon,
          }
          
  if additional_info is not None:
    data = dict(data, **additional_info)
  return data

def get_age(x):
  low,high = x.split('-')
  low = int(low)
  high = int(high)
  return np.mean([low,high])

def get_age_group(x):
  if x >= 60:
    return 'AdultoMayor'
  else:
    return 'Adulto'

def play_dowload(links):
  path_act=dowload_links(links)
  archivos_preprocesados = glob.glob(path_act +'./Datos_Originales_Preprocesados/*.tar.gz')
  path_decompset = path_act +'./Datos_set_Preprocesados/'
  descomprimir(archivos_preprocesados,'Datos_set_Preprocesados')
  condition='EC'
  sujetos_set = os.listdir(path_decompset)
  sujetos_set
  archivos_set = [path_decompset+'./'+name+'./'+name+'_'+condition+'.set' for name in sujetos_set]
  archivos_set
  SELECTED_ICS =[3,4,5,6,7,8,9,10,11,12,14,16,17,18]
  len(SELECTED_ICS)

  matrices = './filtroespacial_lab.mat'
  M = sio.loadmat(matrices)
  A = M["A"]
  W = M["W"]
  eeg_ejemplo=mne.io.read_epochs_eeglab('./EEG_filtroespacial.set')
  lab_ch_names = copy.deepcopy(eeg_ejemplo.ch_names)
  print(len(lab_ch_names),'canales:',lab_ch_names)
  print(W.shape)

  eeg_ejemplo._data.shape

  df=pd.read_csv('./META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv',index_col='Unnamed: 0')

  ages = set(df['Age'].tolist())
  ages
  df['AgeNumber'] = np.vectorize(get_age)(df['Age'])

  df['AgeGroup'] = np.vectorize(get_age_group)(df['AgeNumber'])

  df = df.to_dict(orient='index')

  mats_set_path = './Datos_mat_Preprocesados/'
  mats_vhdr_path = './Datos_mat_Crudos/'


  errores = []
  overwrite = False
  lista_de_archivos = [archivos_set]
  for archivos  in lista_de_archivos:
    for archivo in archivos:
      try:
        if '.vhdr' in archivo:
          mpath = mats_vhdr_path
          sujeto = os.path.split(archivo)[-1].replace('.vhdr','')
        elif '.set' in archivo:
          mpath = mats_set_path
          sujeto = os.path.split(archivo)[-1].replace('.set','').replace('_'+condition,'')
        filename = mpath+sujeto+'.mat'
        if not os.path.isfile(filename) or overwrite:
          additional_info = df[sujeto]
          additional_info['subject'] = sujeto
          sio.savemat(filename,preprocesamiento_lemon(archivo,W_lab=W,lab_channels=lab_ch_names,additional_info=additional_info))
        else:
          print('Ya existe',filename)
      except:
        print('Error para',archivo)
        print(traceback.format_exc())

        errores.append((archivo,      traceback.format_exc()))

      #break

  print('Errores',errores)
