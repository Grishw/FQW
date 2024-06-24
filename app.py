import os, sys, inspect
import io
import base64
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

from flask import Flask, render_template_string, request, render_template, redirect, url_for

import pandas as pd
import numpy as np

# include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"module")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"module/DataProcess")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"templates")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"module/DataAnalis")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import page_template as PageTemp
import data_process as DataProcess
import data_analis as DataAnalis



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# выбор файла
# загрузка файла
@app.route('/', endpoint='index')
def index():
    temp = PageTemp.get_page_by_name('index')
    return render_template(temp)

# просмотр загруженного файла
@app.route('/data_collection', methods=['POST'], endpoint='data_collection')
def data_collection():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        if not os.path.exists(file_path):
            file.save(file_path)

        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        columns = data.columns
        data_head = data.head(10)

        temp = PageTemp.get_page_by_name('data_collection')
        return render_template(temp, 
                               data_review_id='1', 
                               data_review_name='Обзор сырых загруженных данных', 
                               data_review_data=data_head.to_html(classes='table table-striped table-bordered'), 
                               file_name=file.filename,
                               columns=columns)


# что прогнозируем
# по каким параметрам прогнозируем
@app.route('/data_preprocessing', methods=['POST'], endpoint='data_preprocessing')
def data_preprocessing():

    filename = request.form['filename']
     
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return redirect(url_for('index'))

    data = pd.read_csv(file_path)
    targets = ["Date","Temperature","Pressure"]
    data['Date'] = pd.to_datetime(data['Date'])
    
    new_data = data[targets]
    #print(new_data)
    name = DataProcess.get_file_name_for_target_data()
    new_data.to_csv(name, index=False)

    temp = PageTemp.get_page_by_name('data_preprocessing')
    return render_template(temp, 
                           file_name=name,
                           target="Temperature",
                           main = "Temperature",
                           sub = "Pressure",
                           time="Date")
    '''return render_template(temp, 
                           file_name=filename,
                           columns=columns) lkz select_column_hard'''
    

# преобразование данных
@app.route('/data_preprocessing_step2', methods=['POST'], endpoint='data_preprocessing_step2')
def data_preprocessing_step2():

    filename = request.form['filename']
    target = request.form['target']
    features = request.form.getlist('features')

    #print(filename)
    file_path = filename
    if not os.path.exists(file_path):
        return redirect(url_for('index'))

    #print(features)
    data = pd.read_csv(file_path)
    #print(data)

    target_next = target+DataProcess.get_param_sufix()
    data[target_next] = data[target].shift(-1)
    
    data.dropna(inplace=True)
    X = data[features]

    temp = PageTemp.get_page_by_name('data_preprocessing_step2')
    return render_template(temp, 
                           file_name=filename,
                           features = features, data_features_name="Признаки от которых зависит параметер", data_features_data=X.head(10).to_html(classes='table table-striped table-bordered'),
                           target = target_next, data_target_name="Параметер который будем прогнозировать", data_target_data=target_next)

# подготовка к прогнозированию и обучению моделей
@app.route('/predict', methods=['POST'], endpoint='predict')
def predict_param():
    filename = request.form['filename']
    target = request.form['target']
    features = request.form['features']
    print(request.form)

    features = DataProcess.pars_features(features)
    file_path = filename
    if not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    #load
    data = pd.read_csv(file_path)
    data[target] = data[target.replace(DataProcess.get_param_sufix(),'')].shift(-1)
    data.dropna(inplace=True)

    #normalization
    normalz = DataProcess.normalize_input_data(target, data)
    target_data_normalz = normalz[target]

    # point to chage
    change_points_0 = DataProcess.cusum(target_data_normalz)
    points_0 = DataProcess.get_more_points(change_points_0.indMax, change_points_0.B, 0.2)
    print(len(points_0))
    change_points_01 =DataProcess.get_point_with_max_index(points_0)
    
    print('--')
    print(change_points_01)
    print(data['Date'][change_points_01])

    razladca_point = change_points_01
    razladca_date = data['Date'][change_points_01]
    last_fragment_count = len(normalz[target]) - razladca_point
    print(last_fragment_count)

    #last fragment
    imgcount = last_fragment_count

    lastt = data['Temperature'][-imgcount:].to_numpy()
    lastd = data['Date'][-imgcount:].to_numpy()
    print('-1-')
    print(lastt)

    razladca_plot = DataProcess.create_plot('line', data['Temperature'].to_numpy(), data['Date'].to_numpy(), 'Temperature', 'Date', "Точка разладки",  [razladca_date], filename="localData/plot_1.html")
    last_fragment = DataProcess.create_plot('line', lastt, lastd, 'Temperature', 'Date', "Последний фрагмент", [razladca_date], filename="localData/plot_2.html")

    models = ["Perseptron (MLP)", "Свёрточные нейронные сети (CNN)", 'Рекуррентные нейронные сети (RNN)']
    temp = PageTemp.get_page_by_name('predict')
    return render_template(temp, 
                           file_name=file_path,
                           razladca_point=razladca_point,
                           razladca_plot=razladca_plot,
                           last_fragment_count=last_fragment_count,
                           last_fragment=last_fragment,
                           models=models,
                           features=features,
                           target=target)

@app.route('/results_preprocessing', methods=['POST'], endpoint='results_preprocessing')
def results_preprocessing():
    #get sended data
    filename = request.form['filename']
    target = request.form['target']
    features = request.form['features']
    razladca_point = int(request.form['razladca_point'])
    model = request.form['model']
    mode = request.form['mode']
    
    print(request.form)

    features = DataProcess.pars_features(features)

    file_path = filename
    if not os.path.exists(file_path):
        return redirect(url_for('index'))
    
    #выделение последнего фрагмента
    data = DataProcess.get_prepeared_data(file_path, target)
    data = DataProcess.get_last_sistem(data, razladca_point)
    #print(data)

    #normalize data
    #features_normolazed_data = [DataProcess.normalize_input_data(x, data) for x in features]
    features_normolazed_data = data

    #translate
    trashold = 5
    translatetd_data, time_to_rech_treshold = DataProcess.from_standart_to_time_treshold(features_normolazed_data, trashold, features)
    tr_data, tr_temp, tr_pre =  DataProcess.split_TranslateData_array(translatetd_data)

    plot_translated = DataProcess.create_plot('line',  tr_temp, tr_data,  'Temperature', 'Date', 'график времени достижения порога изменения', [])
    non_translate_plot = DataProcess.create_plot('line', data['Temperature'],data['Date'], 'Date', 'Temperature', 'оригинальный график', [])
    time_to_rech_treshold_plot = DataProcess.create_plot('line', time_to_rech_treshold, tr_data, 'Time to reach (hours)', 'Date', 'график времени достижения заданного изменения', [])


    #prepare data to predict 
    tr_data_flat = np.array(tr_data).flatten()
    tr_temp_flat = np.array(tr_temp).flatten()
    tr_pre_flat = np.array(tr_pre).flatten()

    df1 = pd.DataFrame({
    'Date': tr_data_flat,
    'Temperature': tr_temp_flat,
    'Pressure': tr_pre_flat})

    path = DataProcess.data_for_predict_teach_standart()
    df1.to_csv(path, index=False) 

    df2 = pd.DataFrame({
    'Time': time_to_rech_treshold,
    'Temperature_change': tr_temp_flat,
    'Pressure_mean': tr_pre_flat})

    path = DataProcess.data_for_predict_teach_time_to_reach()
    df2.to_csv(path, index=False) 

    temp = PageTemp.get_page_by_name('results_preprocessing')
    return render_template(temp, 
                           line = "изменение на " + str(trashold)+" градусов",
                           translate_plot = plot_translated,
                           non_translate_plot = non_translate_plot,
                           time_to_rech_treshold_plot = time_to_rech_treshold_plot,
                           model=model,
                           mode=mode)


@app.route('/results_preprocessing_2', methods=['POST'], endpoint='results_preprocessing_2')
def results_preprocessing_2():
   
    df1 = pd.read_csv(DataProcess.data_for_predict_teach_standart())
    df2 = pd.read_csv(DataProcess.data_for_predict_teach_time_to_reach())
    mode = request.form['mode']
    model = request.form['model']
    
    f = ['Date', 'Temperature', 'Pressure']
    t = ['Temperature', 'Pressure']
    name = DataAnalis.get_model_name(t, f, "MLP")
    df1['Date'] = pd.to_datetime(df1['Date']).astype(int) / 10**9  # Преобразуем дату в секунды

    # подготовка данных
    X_train1 = df1[f].values
    y_train1 = df1[t].values

    X_train1, y_train1  = DataProcess.my_array_split(X_train1, y_train1, 120, 24)
    xx = np.array(X_train1)
    yy = np.array(y_train1)

    input_shape_rnn_functions = xx[0].shape[1]
    input_shape_rnn_steps = xx[0].shape[0]

    out_shape_rnn_functions = yy[0].shape[1]
    out_shape_rnn_steps = yy[0].shape[0]
    
    # by рекуррентная нейронная сеть (RNN) с использованием LSTM-слоя и полносвязных (Dense) слоев.
    # создание и обучени или загрузка
    rnn_model, flag = DataAnalis.load_or_create_model(name, DataAnalis.create_rnn_model, input_shape_rnn_steps, input_shape_rnn_functions, out_shape_rnn_steps, out_shape_rnn_functions)

    if flag == False:
        DataAnalis.train_and_save_model(rnn_model, xx, yy, name)

    # подготовка к прогнозу
    ww = xx[-1:]

    #predict
    predictions = DataAnalis.predict_rnn(rnn_model, ww, steps=10)
    #print(predictions)

    tr_data, tr_temp, tr_pre = DataAnalis.split_prediction_rnn(predictions)

    tr_data_flat = np.array(tr_data).flatten()
    tr_temp_flat = np.array(tr_temp).flatten()
    tr_pre_flat = np.array(tr_pre).flatten()

    resultData = pd.DataFrame({
    'Date': tr_data_flat,
    'Temperature': tr_temp_flat,
    'Pressure': tr_pre_flat})

    name_to_save = DataProcess.get_predict_result_save_name()
    resultData.to_csv(name_to_save)

    
    tech_result_plot_1 = DataProcess.create_plot('line', tr_pre_flat, tr_data_flat,  'Pressure', 'Date','Прогноз давления', [])
    tech_result_plot = DataProcess.create_plot('line',  tr_temp_flat, tr_data_flat, 'Temp', 'Date', 'Прогноз температуры', [])
    temp = PageTemp.get_page_by_name('results_preprocessing_2')
    return render_template(temp, 
                           tech_result_plot=tech_result_plot,
                           tech_result_plot_1 = tech_result_plot_1)

if __name__ == '__main__':
    matplotlib.use('Agg')
    plt.style.use('fast') 
    app.run(debug=True)
