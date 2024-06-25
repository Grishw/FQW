import pandas as pd
import numpy as np
import os, sys, inspect

import tensorflow as tf


from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, Input, Reshape

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
        model = create_model_fn(input_shape_s, input_shape_f, output_steps, output_shape)
        return model, False


def create_mlp_model(input_steps=120, input_shape=3, output_steps=24, output_shape=2):
    model = Sequential()
    model.add(Input(shape=(input_steps, input_shape)))  # (120, 3)
    
    model.add(Flatten())  # Преобразуем (120, 3) в (360,)
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(48, activation='relu'))  # Уменьшаем до 48 элементов, что равно 24 * 2
    
    # Последний слой и преобразование
    model.add(Dense(output_steps * output_shape))  # Выдача 24*2=48 параметров
    model.add(Reshape((output_steps, output_shape)))  # Преобразование в форму (24, 2)
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save_model(model, X_train, y_train, model_name, epochs=300, batch_size=32):
    step_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    model.save(model_name)
    return step_history

def new_arr(cur, new):
    assert cur.shape == (1, 120, 3), "Начальный массив должен иметь форму (1, 120, 3)"
    #print(new.shape)
    assert new.shape == (1, 24, 2), "Массив новых элементов должен иметь форму (1, 24, 3)"

    
    pred_clear = []
    lenss = len(new[0])
    assert lenss == 24, "Длина "

    getvel = len(cur[0]) - lenss
    #print(getvel)
    assert getvel == 96, "количество эдементов"

    
   
    temp = cur[0][-getvel:]
    #print(temp.shape)
    assert temp.shape == (96, 3), "Начальный массив должен иметь форму (1, 96, 3)"
    temp = temp.tolist()
    
    for i in new[0]:
        s = temp[-1][0] + 3600  # Получаем последний элемент времени и добавляем 3600
        w = [s, i[0], i[1]]
        pred_clear.append(w)
        temp.append(w)
        getvel += 1
        
    return np.array(temp), np.array(pred_clear)

def predict(model, data, steps):
    predictions = []
    current_input = data
    
    for _ in range(steps):
        prediction = model.predict(current_input)
        
        new_input, pred = new_arr(current_input, prediction)
        predictions.append(pred) # запоминаем прогнозы
        
        current_input = new_input[np.newaxis, :, :]
    return np.array(predictions)



def split_prediction_rnn(predictions):
    tr_data_flat, tr_temp_flat, tr_pre_flat = [], [], []
    #print(predictions.shape)
    for j in predictions:
        #print(j.shape)
        for i in j:
            #print(i)
            datas = pd.to_datetime(i[0], unit='s')
            tr_data_flat.append(datas)
            tr_temp_flat.append(i[1] / 10**8 )
            tr_pre_flat.append(i[2] /10**6 )

    return  tr_data_flat, tr_temp_flat, tr_pre_flat


'''
print(tr_pre_flat.shape)
    tr_temp_flat_m = tr_temp_flat.reshape(1, 240)
    print(tr_temp_flat_m.shape)
    tr_pre_flat_m = tr_temp_flat.reshape(1, 240)
    assert tr_temp_flat_m.shape == (1, 240), "Начальный массив должен иметь форму (1, 240)"
    assert tr_pre_flat_m.shape == (1, 240), "Начальный массив должен иметь форму (1, 240)"
    
    partitions1 = np.split(tr_temp_flat_m, 10, axis=1)
    partitions2 = np.split(tr_pre_flat_m, 10, axis=1)

    # Объединение соответствующих частей
    for_scaler = np.zeros((10, 24, 2))
    for i in range(10):
        for_scaler[i, :, 0] = partitions1[i].reshape(24)
        for_scaler[i, :, 1] = partitions2[i].reshape(24)


    print(for_scaler.shape)
    assert for_scaler.shape == (10, 24, 2), "Начальный массив должен иметь форму (10, 24, 2)"
    for_scaler = scaler_y.inverse_transform(for_scaler.reshape(-1, for_scaler.shape[-1])).reshape(for_scaler.shape)
    assert for_scaler.shape == (10, 24, 2), "Начальный массив должен иметь форму (10, 24, 2)"
    print(for_scaler)
'''