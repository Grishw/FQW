from jinja2 import Environment, FunctionLoader, BaseLoader
import data_process_page_part as Section
import base_page_part as Part




def __page_loader(name):
    '''
    loader

    Page part:
        header - main page
        navegation
        main_content
        body
    '''
    page = ''''''
    if name == 'header':
        return Part.header
    elif name == 'navegation':
        return Part.navegation
    elif name == 'main_content':
        return Part.main_content
    if name == 'body':
        return Part.body
__env = Environment(loader=FunctionLoader(__page_loader))






#export
def get_page_by_name(name):
    '''
    index - main page
    data_collection -
    data_preprocessing -
    predict
    
    '''
    head = Part.header
    footer = Part.footer
    body = ""
    
    main = ""
    main_block = __env.get_template('body')
    main_content_component = __env.get_template('main_content')

    if name == 'index':
        main_content = Section.load_data

        content = main_content_component.render(main_content=main_content)
        body = main_block.render(main=content)
    else:
        navegation_component = __env.get_template('navegation')

        if name == 'data_collection':
            navegation = navegation_component.render(
                data_collection="active", data_collection_link="#file_load",
                data_preprocessing_link="/data_preprocessing") 
                
            main_content = Section.file_review + Section.data_review 
            
        elif name == "data_preprocessing":
            navegation = navegation_component.render(
                data_collection_link="/data_collection", 
                data_preprocessing = "active", data_preprocessing_link="/data_preprocessing",
                model_preprocessing_link = "/model_processing")
            
            main_content = Section.select_column

        elif name == "data_preprocessing_step2":
            navegation = navegation_component.render(
                data_collection_link="/data_collection", 
                data_preprocessing = "active", data_preprocessing_link="/data_preprocessing",
                model_preprocessing_link = "/model_processing")
            
            main_content = Section.data_review_target_and_features

        elif name == 'predict':
            navegation = navegation_component.render(
                data_collection_link="/data_collection", 
                model_preprocessing="active",
                model_preprocessing_link = "/model_processing")
            
            main_content = Section.razladca_graphiks + Section.last_fragment_graphiks + Section.predict_param

        elif name == 'results_preprocessing':
            navegation = navegation_component.render(
                results_preprocessing="active")
            
            main_content = Section.translete_data_plot
            
        elif name == 'results_preprocessing_2':
            navegation = navegation_component.render(
                results_preprocessing="active")
            
            main_content =  Section.tech_result + Section.predict_result
            
        content = main_content_component.render(main_content=main_content)
        body = main_block.render(main=content, navegation=navegation)
    

    page_str = head + body + footer
    return Environment(loader=BaseLoader()).from_string(page_str)
    

    
    







