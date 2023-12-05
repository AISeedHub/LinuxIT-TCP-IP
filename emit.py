from model import PearDetectionModel


def response_structure():
    return {
        "file_name": None,
        "result": None,
        "error_code": 0
    }


def request_classification(task_manager, request_data):
    """
    expected input request_data = ["001.jpg","002.jpg","003.jpg"]
    """
    print("request_classification")
    response_data = []
    for file_name in request_data:
        response_element = response_structure()
        response_element["file_name"] = file_name
        result = task_manager.dl_model.inference(file_name)
        if result is -1:
            response_element["error_code"] = 1
        else:
            response_element["result"] = result
        response_data.append(response_element)
    return response_data


def stop_classification(task_manager, request_data):
    """
    expected input request_data = None
    """
    print("stop_classification")
    task_manager.dl_model.ease_model()
    response_element = response_structure()
    response_element["result"] = 2
    return [response_element]


def request_download(task_manager, request_data):
    """
    expected input request_data = ["URL"]
    """
    print("request_download")
    # TODO: copy or download?
    response_element = response_structure()
    import wget
    try:
        wget.download(request_data[0], out=task_manager.config["DIR_MODEL_DETECTION"])
        response_element["result"] = 2  # success
    except:
        response_element["result"] = 1
    return [response_element]


def request_model_select(task_manager, request_data):
    """
    expected input request_data = ["model1.pt"]
    """
    print("request_model_select")
    response_element = response_structure()
    response_element["file_name"] = request_data[0]
    try:
        task_manager.dl_model.set_model_name(request_data[0])
        response_element["result"] = 2
    except:
        response_element["result"] = 1
        response_element["error_code"] = 2  # no file
    return [response_element]


def request_current_model(task_manager, request_data):
    """
    expected input request_data = null
    """
    print("request_current_model")
    response_element = response_structure()
    response_element["result"] = task_manager.dl_model.model_name
    return [response_element]


def request_list_model(task_manager, request_data):
    """
    expected input request_data = null
    """
    print("request_list_model")
    # check the list file in dir_model_detection
    import os
    try:
        list_file = os.listdir(task_manager.config["DIR_MODEL_DETECTION"])
    except:
        list_file = []
    if len(list_file) is 0:  # no files
        response_element = response_structure()
        response_element["result"] = 1
        response_element["error_code"] = 2
        return [response_element]
    response_data = []
    for file_name in list_file:
        response_element = response_structure()
        response_element["file_name"] = file_name
        response_element["result"] = 2
        response_data.append(response_element)
    return response_data


def request_model_delete(task_manager, request_data):
    """
    expected input request_data = ["model1.pt"]
    """
    print("request_model_delete")
    response_element = response_structure()
    response_element["file_name"] = request_data[0]
    import os
    try:
        os.remove(task_manager.config["DIR_MODEL_DETECTION"] + "/" + request_data[0])
        response_element["result"] = 2
    except:
        response_element["result"] = 1
        response_element["error_code"] = 2  # no file
    return [response_element]
