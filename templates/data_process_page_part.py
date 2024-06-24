load_data = '''
<section id="data_load">
    <div class="container mt-5" data-aos="fade-up">
        <h1 class="text-center mb-5">Weather Forecasting SPA</h1>
        <div class="card p-3 mb-4">
            <form action="/data_collection" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Upload CSV file:</label>
                    <input type="file" class="form-control-file" id="file" name="file">
                </div>
                <button type="submit" class="btn btn-primary">Upload</button>
            </form>
        </div>
    </div>
</section>'''

file_review = '''
<section id="file_load">
    <div class="container mt-5" data-aos="fade-up">
        <h1 class="text-center mb-5">Weather Forecasting SPA</h1>
        <div class="card p-3 mb-4">
            <div class="form-group">
                <p>Uploaded CSV file: {{ file_name }}</p>
            </div>
            <form action="/" method="get">
                <button type="submit" class="btn btn-simple" >Reload file</button>
            </form>
        </div>
    </div>
</section>'''


data_review = '''
<section id="data_review_{{ data_review_id }}">
    <div class="container mt-5" data-aos="fade-up">
        <div class="card p-3 mb-4">
            <h2>{{ data_review_name | safe }}</h2>
            <div class="table-responsive">
                {{ data_review_data | safe }}
            </div>
        </div>
        <form action="/data_preprocessing" method="post" enctype="multipart/form-data">
            <input type="hidden" name="filename" value="{{ file_name }}">
            <input type="hidden" name="columns" value="{{ columns }}">
            <button type="submit" class="btn btn-primary" >Go to data processing</button>
        </form>
    </div>
</section>'''


select_column = '''
<div class="card p-3 mb-4">
    <form action="/data_preprocessing_step2" method="post">
        <input type="hidden" name="filename" value="{{ file_name }}">
        <div class="form-group">
            <label for="target">Параметр по которому будет строиться прогноз:</label>
            <select class="form-control" id="target" name="target">
                <option value="{{ target }}">{{ target }}</option>
            </select>
        </div>
        <div class="form-group">
            <label for="features">Параметры учавствующие в прогнозе (на основе чего производится прогноз):</label>
            <div id="features">
                <div class="form-check">
                    <input class="form-check-input" type="hidden" id="{{ main }}" name="features" value="{{ main }}">
                    <label class="form-check-label" for="{{ main }}">Основной параметр: {{ main }}</label>
                 </div>
                 <div class="form-check">
                    <input class="input" type="hidden" id="{{ sub }}" name="features" value="{{ sub }}">
                    <label class="form-check-label" for="{{ sub }}">Дополнительный: {{ sub }}</label>
                 </div>
                 <div class="form-check">   
                    <input class="form-check-input" type="hidden" id="{{ time }}" name="features" value="{{ time }}">
                    <label class="form-check-label" for="{{ time }}">Временная шкала: {{ time }}</label>
                </div>
            </div>
        </div>
        <button type="submit" class="btn btn-success">Сhoose</button>
    </form>
</div>'''

select_column_hard = '''
<div class="card p-3 mb-4">
    <form action="/data_preprocessing_step2" method="post">
        <input type="hidden" name="filename" value="{{ file_name }}">
        <div class="form-group">
            <label for="target">Target Variable:</label>
            <select class="form-control" id="target" name="target">
                {% for column in columns %}
                <option value="{{ column }}">{{ column }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="form-group">
            <label for="features">Feature Variables:</label>
            <div id="features">
                {% for column in columns %}
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="{{ column }}" name="features" value="{{ column }}">
                    <label class="form-check-label" for="{{ column }}">{{ column }}</label>
                </div>
                {% endfor %}
            </div>
        </div>
        <button type="submit" class="btn btn-success">Сhoose</button>
    </form>
</div>'''

data_review_target_and_features = '''
<section id="data_target_{{ data_target_id }}">
    <div class="container mt-5" data-aos="fade-up">
        <div class="card p-3 mb-4">
            <h2>{{ data_target_name | safe }}</h2>
            <div class="table-responsive">
                {{ data_target_data | safe }}
            </div>
        </div>
        <div class="card p-3 mb-4">
            <h2>{{ data_features_name | safe }}</h2>
            <div class="table-responsive">
                {{ data_features_data | safe }}
            </div>
        </div>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="hidden" name="filename" value="{{ file_name }}">
            <input type="hidden" name="target" value="{{ target }}">
            <input type="hidden" name="features" value="{{ features }}">
            <button type="submit" class="btn btn-primary" >Go to predict</button>
        </form>
    </div>
</section>'''

razladca_graphiks = '''
<div class="card p-3 mb-4">
    <h1 class="text-center mb-5">Поиск точки разладки по температуре</h1>
    <div class="card p-3 mb-4">
        <h2>Индекс: {{ razladca_point }}</h2>
        {{ razladca_plot | safe }}
    </div>
</div>'''

last_fragment_graphiks = '''
<div class="card p-3 mb-4">
    <h1 class="text-center mb-5">Выделение последнего фрагмента по температуре</h1>
    <div class="card p-3 mb-4">
        <h2>Количество данных в последнем фрагменте: {{ last_fragment_count }}</h2>
        {{ last_fragment | safe }}
    </div>
</div>'''

predict_param = '''
<div class="card p-3 mb-4">
    <form action="/results_preprocessing" method="post">
        <input type="hidden" name="filename" value="{{ file_name }}">
        <input type="hidden" name="razladca_point" value="{{ razladca_point }}">
        <input type="hidden" name="features" value="{{ features }}">
        <input type="hidden" name="target" value="{{ target }}">
        <div class="form-group">
            <label for="model">Select predict model:</label>
            <select class="form-control" id="model" name="model">
                {% for model in models %}
                    <option value="{{ model }}">{{ model }}</option>
                {% endfor %}
            </select>
            <label for="data_transform">Select transform mode:</label>
            <select class="form-control" id="data_transform" name="mode">
                <option value="yes">Преобразовать в в ряд времени достижения порога изменения</option>
            </select>
        </div>
        <button type="submit" class="btn btn-success">Gooo</button>
    </form>
</div>'''

translete_data_plot = '''
<h1 class="text-center mb-5">Translation result</h1>
<div class="card p-3 mb-4">
    <h2>График времени достижения порога заданного изменения: {{ line }}</h2>
    {{ translate_plot | safe }}
    <h2>Оригинальный график: </h2>
    {{ non_translate_plot | safe }}
    <h2>График времени за которое достигалась точка достижения порогового изменения: </h2>
    {{ time_to_rech_treshold_plot | safe }}
</div>'''

tech_result = '''
<h1 class="text-center mb-5">Model teach result</h1>
<div class="card p-3 mb-4">
    <h2>График прогноза по Температура</h2>
    {{ tech_result_plot | safe }}
    <h2>График  прогноза по Давлению</h2>
    {{ tech_result_plot_1 | safe }}
</div>'''

predict_result = '''
<div class="card p-3 mb-4">
    <h2>Predict Data Plot</h2>
    <div class="table-responsive">
        {{ predict_result_predict | safe }}
    </div>
</div>'''

data_last= '''
<section id="data_review_{{ data_review_id }}">
    <div class="container mt-5" data-aos="fade-up">
        <div class="card p-3 mb-4">
            <h2>{{ data_review_name | safe }}</h2>
            <div class="table-responsive">
                {{ data_review_data | safe }}
            </div>
        </div>
        <form action="/results_preprocessing_2" method="post" enctype="multipart/form-data">
            <input type="hidden" name="mode" value="{{ mode }}">
            <input type="hidden" name="model" value="{{ model }}">
            <button type="submit" class="btn btn-primary" >Go to nex resulp page</button>
        </form>
    </div>
</section>'''

