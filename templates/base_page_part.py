header = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/aos@2.3.4/dist/aos.css">
    <link rel="stylesheet" href="/static/style.css">
    <title>Weather Forecasting SPA</title>
</head>
<body>'''

footer = '''
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/aos@2.3.4/dist/aos.js"></script>
    <script>
        AOS.init();
    </script>
</body>
</html>''' 

navegation = '''
<!-- Sidebar -->
<nav id="sidebar" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
    <div class="sidebar-sticky pt-3">
        <h5>Steps</h5>
        <ul class="nav flex-column">
            <li class="nav-item">
                <a class="nav-link {{ data_collection | safe}}" href="{{ data_collection_link | safe}}">Upload Data owerview</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {{ data_preprocessing  | safe}}" href="{{ data_preprocessing_link | safe}}">Data process</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {{ model_preprocessing  | safe}}" href="{{ model_preprocessing_link | safe}}">Predict</a>
            </li>
            <li class="nav-item">
                <a class="nav-link {{ results_preprocessing  | safe}}" href="{{ results_preprocessing_link| safe}}">Results process</a>
            </li>
        </ul>
    </div>
</nav>'''

main_content = '''
<!-- Main content -->
<main role="main" class="col-md-9 ml-sm-auto col-lg-10 px-md-4">
    {{ main_content | safe }}
</main>'''


body = '''
<div class="container-fluid mt-5" data-aos="fade-up">
    <div class="row">
        {{ navegation | safe }}
        {{ main | safe }}
    </div>
</div>'''

pageExaple = header + body + footer

