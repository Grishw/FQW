import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots



def create_plot_1(plot_type, y, x, ylabel, xlabel, title, veltical_line, figsize=(16, 8)):
    # Создаем график
    fig, ax = plt.subplots(figsize=figsize)
    print('-1-1-')
    if plot_type == 'line':
        ax.plot(x, y)
    elif plot_type == 'bar':
        ax.bar(x, y)
    elif plot_type == 'scatter':
        ax.scatter(x, y)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    print('-1-2-')
    # Настраиваем график
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    for cp in veltical_line:
        ax.axvline(cp, color='r', linestyle='--', label='Change Point')
    print('-1-3-')
    # Сохраняем график в буфер
    buf = io.BytesIO()
    print('-1-3-1-')
    plt.savefig(buf, format='png')
    print('-1-3-2-')
    buf.seek(0)
    plt.close(fig)
    print('-1-4-')
    # Кодируем изображение в base64
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    print('-1-5-')
    # HTML-шаблон для отображения графика
    plot = f'''
        <img src="data:image/png;base64,{plot_data}" alt="Plot">
    '''
    return plot

def create_plot(plot_type, y, x, ylabel, xlabel, title, vertical_lines = []):
    fig = go.Figure()
    
    # Создаем график в зависимости от типа
    if plot_type == 'line':
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Line Plot'))
    elif plot_type == 'bar':
        fig.add_trace(go.Bar(x=x, y=y, name='Bar Plot'))
    elif plot_type == 'scatter':
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Scatter Plot'))
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    # Добавляем вертикальные линии
    for cp in vertical_lines:
        fig.add_vline(x=cp, line=dict(color='red', dash='dash'), name='Change Point')

    # Настраиваем график
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white'
    )
    
   # Сохраняем график в HTML буфер
    buffer = io.StringIO()
    pio.write_html(fig, file=buffer, auto_open=False, include_plotlyjs='cdn')
    html_content = buffer.getvalue()
    buffer.close()

    return html_content

def create_multu_plot(plot_types, ys, xs, ylabels, xlabels, titles, vertical_lines=[], subplot_titles=[]):
    if len(plot_types) != len(ys) or len(plot_types) != len(xs):
        raise ValueError("Number of plot types, y-series, and x-series must be the same")

    fig = make_subplots(rows=len(plot_types), cols=1, subplot_titles=subplot_titles)

    for i, (plot_type, y, x, ylabel, xlabel, title) in enumerate(zip(plot_types, ys, xs, ylabels, xlabels, titles), start=1):
        if plot_type == 'line':
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Line Plot'), row=i, col=1)
        elif plot_type == 'bar':
            fig.add_trace(go.Bar(x=x, y=y, name='Bar Plot'), row=i, col=1)
        elif plot_type == 'scatter':
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Scatter Plot'), row=i, col=1)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # Добавляем вертикальные линии
        for cp in vertical_lines:
            fig.add_vline(x=cp, line=dict(color='red', dash='dash'), name='Change Point', row=i, col=1)

        fig.update_xaxes(title_text=xlabel, row=i, col=1)
        fig.update_yaxes(title_text=ylabel, row=i, col=1)
        fig.update_layout(title=title, template='plotly_white')

    # Сохраняем график в HTML буфер
    buffer = io.StringIO()
    pio.write_html(fig, file=buffer, auto_open=False, include_plotlyjs='cdn')
    html_content = buffer.getvalue()
    buffer.close()

    return html_content


def wind_direction_normalization(df):
    # Нормализация направления ветра
    wind_direction_rad = np.deg2rad(df['Wind_Direction'])
    df['Wind_Direction_sin'] = np.sin(wind_direction_rad)
    df['Wind_Direction_cos'] = np.cos(wind_direction_rad)
    df.drop('Wind_Direction', axis=1)
    return df

def pars_features(features = 'Temperature, Date'):
    features = features.replace('[','')
    features = features.replace(']','')
    features = features.replace("'",'')
    features = features.replace(" ",'')
    features = features.split(',')
    return features

def normalize_input_data(target, data):
    scaler_temp = StandardScaler()
    scaler_humidity = MinMaxScaler()

    normalz = {}

    if target == 'Temperature':
        normalz['Temperature'] = scaler_temp.fit_transform(data['Temperature'].values.reshape(-1, 1))
        return normalz
    elif target == 'Wind_Direction':
        normalz = wind_direction_normalization(data)
        return normalz
    else:
        normalz[target] = scaler_humidity.fit_transform(data[target].values.reshape(-1, 1))
        return normalz


def get_prepeared_data(file_path, target):
    data = pd.read_csv(file_path)

    #new_data = [x for x in range(len(data['Date']))] # переводим дату в числа с 1 до конца +1 шаг
    #data = data.drop('Date', axis=1) 
    #data.insert(0, 'Date', new_data)
    #data['Date'] = pd.to_datetime(data['Date']).astype(int) / 10**9

    data.dropna(inplace=True)
    return data

class TranslateData:
    def __init__(self, start_time = 0,  mean_per = 0, start_temp = 0):
        self.time = start_time
        self.temp = start_temp
        self.per = mean_per

#treshold = изменения на число 
def from_standart_to_time_treshold(data, treshold, features):
    
    data_time = 'Date'
    data_pre = 'Pressure' 
    data_temp = 'Temperature'

    start_time = data[data_time][0]
    start_pre = data[data_pre][0]
    start_temp = data[data_temp][0]

    result_base = []
    result_time_to_treshold = [0]
    a = TranslateData(start_time, start_pre, start_temp)
    result_base.append(a)
    
    for i in range(len(data[data_temp])):
        current_temp = data[data_temp][i]
        current_time = data[data_time][i]
        current_pre = data[data_pre][i]
        
        if abs(current_temp - start_temp) > abs(treshold):
            mean_pre = (start_pre + current_pre)/2
            a = TranslateData(start_time, mean_pre, start_temp)
            result_base.append(a)
            result_time_to_treshold.append((pd.to_datetime(current_time) - pd.to_datetime(start_time))/ pd.Timedelta(hours=1))
            
            start_time = current_time
            start_pre = current_pre
            start_temp = current_temp

    return result_base, result_time_to_treshold

#treshold = изменения в %
def from_standart_to_time_treshold_1(data, treshold, features):
    
    data_time = 'Date'
    data_pre = 'Pressure' 
    data_temp = 'Temperature'

    data_time_index = features.index(data_time)
    data_pre_index = features.index(data_pre)
    data_temp_index = features.index(data_temp)

    start_time = data[data_time_index][data_time][0][0]
    start_pre = data[data_pre_index][data_pre][0][0]
    start_temp = data[data_temp_index][data_temp][0][0]

    result_base = []
    result_time_to_treshold = [0]
    a = TranslateData(start_time, start_pre, start_temp)
    result_base.append(a)
    
    for i in range(len(data[data_temp_index][data_temp])):
        current_temp = data[data_temp_index][data_temp][i][0]
        current_time = data[data_time_index][data_time][i][0]
        current_pre = data[data_pre_index][data_pre][i][0]
        
        if abs(current_temp - start_temp) > abs(start_temp*treshold):
            mean_pre = (start_pre + current_pre)/2
            a = TranslateData(start_time, mean_pre, start_temp, i)
            result_base.append(a)
            result_time_to_treshold.append((current_time - start_time))
            
            start_time = current_time
            start_pre = current_pre
            start_temp = current_temp

    return result_base, result_time_to_treshold

def split_TranslateData_array(a):
    time_arr = []
    temp_arr = []
    per_arr = []

    for e in a:
        time_arr.append(e.time)
        temp_arr.append(e.temp)
        per_arr.append(e.per)


    return time_arr, temp_arr, per_arr

def get_last_sistem(data, razladka):
    new = pd.DataFrame(0, index=np.arange(razladka), columns=data.columns)
    for e in data.columns:
        new[e] = data[e][-razladka:].to_numpy()
    return new

def get_param_sufix():
    return '_next_day'

def get_file_save_dir():
    return 'localData/'

def data_for_predict_teach_standart():
    return get_file_save_dir() + 'to_teach_standart.csv'

def data_for_predict_teach_time_to_reach():
    return get_file_save_dir() + 'to_reach.csv'

def get_file_name_for_target_data():
    return get_file_save_dir() + 'target_data.csv'

def get_predict_result_save_name():
    return get_file_save_dir() + 'predict_result.csv'

def split_result_array_custom(a):
    per_arr = []
    temp_arr = []
    data_arr = []

    for e in a:
        datas = pd.to_datetime(e[0], unit='s')
        data_arr.append(datas)
        temp_arr.append(e[1])
        per_arr.append(e[2])


    return data_arr, temp_arr, per_arr

def split_result_array(a):
    per_arr = []
    temp_arr = []

    for e in a:
        temp_arr.append(e[0])
        per_arr.append(e[1])


    return temp_arr, per_arr

def my_array_split(arr_x, arr_y, step = 120, pred = 24):
    assert arr_x.shape == (4664, 3), "Начальный массив должен иметь форму (1, 120, 3)"
    assert arr_y.shape == (4664, 2), "Массив новых элементов должен иметь форму (1, 24, 3)"
    new_x = []
    new_y = []
    new_j = []
    arr_x_l = arr_x.tolist()
    arr_y_l = arr_y.tolist()
    
    j = 0
    flag = 0
    k = 0
    
    for i in arr_x:
        j += 1
        k += 1
        if flag == 0:
            new_j.append(i)
            if j >= step:
                j = 0
                flag = 1
                new_x.append(new_j)
                new_j = []
        else:
            new_j.append(arr_y[k])
            if j >= pred:
                j = 0
                flag = 0
                new_y.append(new_j)
                new_j = []
            
    return new_x, new_y

def my_array_split_1(arr_x, arr_y, step = 120, pred = 24):
    #assert arr_x.shape == (4664, 3), "Начальный массив должен иметь форму (1, 120, 3)"
    #assert arr_y.shape == (4664, 2), "Массив новых элементов должен иметь форму (1, 24, 3)"
    new_x = []
    new_y = []
    new_jx = []
    new_iy = []
    
    j = 0
    flag = 0
    k = 0
    w=0
    
    for i in arr_x:
        j += 1
        k += 1
        new_jx.append(i)

        if j >= step:
            j = 0
            flag = 1
            new_x.append(new_jx)
            new_jx = []

            
        if flag == 1:
            w += 1
            new_iy.append(arr_y[k])

            if w >= pred:
                w = 0
                flag = 0
                new_y.append(new_iy)
                new_iy = []
    if (len(new_x) > len(new_y)):
        lensss = len(new_y)
        new_x = [new_x[x] for x in range(lensss)]
    return new_x, new_y


'''
     y2 ,x2, z2 = DataProcess.split_result_array_custom(ww[0])
        tech_result_plot_1 = DataProcess.make_subplots(
        plot_types=['line', 'line'],  # Типы графиков для каждого подграфика
        ys=[tr_data_flat, y2],                 # Данные по оси y для каждого подграфика
        xs=[tr_pre_flat, x2],                 # Данные по оси x для каждого подграфика
        ylabels=['Date', 'Date'],  # Метки оси y для каждого подграфика
        xlabels=['Pressure', 'Pressure'],  # Метки оси x для каждого подграфика
        titles=['Прогноз давления', 'Давление'],  # Заголовки для каждого подграфика
        vertical_lines=[10, 20],      # Вертикальные линии для каждого подграфика
        subplot_titles=['Прогноз давления', 'давление']  # Заголовки подграфиков
    )
    '''