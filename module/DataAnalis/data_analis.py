import pandas as pd
import numpy as np
import os, sys, inspect

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, Input

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"module/DataProcess")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import data_process as DataProcess

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"module/DataProcess")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import data_process as DataProcess

def get_model_name(t, f, model):
    name = model

    for i in t:
        name += i[0]

    name += '-'

    for i in f:
        name += i[0]

    name = DataProcess.get_file_save_dir() + name + '.h5'
    return name 


def load_or_create_model(model_name, create_model_fn, input_shape_s, input_shape_f, output_steps, output_shape):
    if os.path.exists(model_name):
        model = load_model(model_name, compile=False)
        model.compile(optimizer='adam', loss='mse')
        return model, True
    else:
        model = create_model_fn(input_shape_s, input_shape_f)
        return model, False


def create_rnn_model(input_steps=120, input_shape=3, output_steps=24, output_shape=2):
    model = Sequential()
    model.add(Input(shape=(input_steps, input_shape)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    
    model.add(Dense(output_steps * output_shape))  # Выдача 24*2 параметров
    model.add(tf.keras.layers.Reshape((output_steps, output_shape)))  # Преобразование в форму (24, 2)
    
    model.compile(optimizer='adam', loss='mae')
    return model

def train_and_save_model(model, X_train, y_train, model_name, epochs=300, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    model.save(model_name)

def new_arr(cur, new):
    pred_clear = []
    lenss = len(new)
    getvel = len(cur) - lenss
    temp = cur[-getvel:][0].tolist()
    #print(temp)
    for i in new[0]:
        #print(temp[-1:][0][0])
        s = temp[-1:][0][0] + 3600
        w = [s, i[0], i[1]]
        pred_clear.append(w)
        temp.append(w)
        getvel += 1
    return np.array(temp), np.array(pred_clear)

def predict_rnn(model, data, steps):
    predictions = []
    current_input = data
    
    for _ in range(steps):
        #print('---')
        #print(current_input)
        prediction = model.predict(current_input)
        
        current_input, pred = new_arr(current_input, prediction)
        predictions.append(pred) # запоминаем прогнозы
        current_input = current_input[np.newaxis, :, :] 
        #print(f'Для следующего прогноза данные: {current_input}')
    return np.array(predictions)



def split_prediction_rnn(predictions):
    tr_data_flat, tr_temp_flat, tr_pre_flat = [], [], []

    for j in predictions:
        for i in j:
            datas = pd.to_datetime(i[0], unit='s')
            tr_data_flat.append(datas)
            tr_temp_flat.append(i[1])
            tr_pre_flat.append(i[2])

    return  tr_data_flat, tr_temp_flat, tr_pre_flat

