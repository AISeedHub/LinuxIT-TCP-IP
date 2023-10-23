from model import DLModel
# from utils import inference

def stop_classification(task_manager, response, raw_request):
    print("stop_classification")
    task_manager.dl_model.ease_model()
    return stop_response(task_manager, response)

def stop_response(task_manager, response):
    print("response_classification")
    func_str = stop_response.__name__
    response["cmd"] = task_manager[func_str]
    response["result"] = 2


def request_classification(task_manager, response, raw_request):
    print("request_classification")
    # Classification result is returned here
    raw_data_directory =  raw_request
    response["result"] = task_manager.dl_model.inference(raw_data_directory)
    if response["result"] is -1:
        raise Exception
    return response_classification(task_manager, response)


def response_classification(task_manager, response):
    print("response_classification")
    func_str = response_classification.__name__
    response["cmd"] = task_manager[func_str]


def request_download(task_manager, response, raw_request):
    print("request_download")
    # TODO: copy or download?
    response["result"] = 1
    return response_download(task_manager, response)


def response_download(task_manager, response):
    print("response_download")
    func_str = response_download.__name__
    response["cmd"] = task_manager[func_str]


def request_model_select(task_manager, response, raw_request):
    print("request_model_select")
    model_name = raw_request
    try:
        task_manager.dl_model.set_model_name(model_name)
        response["result"] = 2
    except Exception as e:
        print(f"Error handling request: {e}")
        response["result"] = 1
    finally:
        response_model_select(task_manager, response)


def response_model_select(task_manager, response):
    print("response_model_select")
    func_str = response_model_select.__name__
    response["cmd"] = task_manager[func_str]


def request_current_model(task_manager, response, raw_request):
    print("request_current_model")
    response["result"] = task_manager.dl_model.model_name
    return response_current_model(task_manager, response)


def response_current_model(task_manager, response):
    print("response_current_model")
    func_str = response_current_model.__name__
    response["cmd"] = task_manager[func_str]


def request_model_delete(task_manager, response, raw_request):
    print("request_model_delete")
    # TODO: delete model file or change model_name
    response["result"] = 1
    return response_model_delete(task_manager, response)


def response_model_delete(task_manager, response):
    print("response_model_delete")
    func_str = response_model_delete.__name__
    response["cmd"] = task_manager[func_str]
