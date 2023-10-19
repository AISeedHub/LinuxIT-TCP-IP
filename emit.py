from model import DLModel


def stop_classification():
    print("stop_classification")
    pass


def request_classification(command_codes, response, json_data):
    print("request_classification")
    DLModel.name = "test"
    DLModel.dir = "test/model.properties"
    print("Done Start")
    # Classification result is returned here
    response["result"] = 1
    return response_classification(command_codes, response)


def response_classification(command_codes, response):
    print("response_classification")
    func_str = response_classification.__name__
    response["cmd"] = command_codes[func_str]


def request_download(command_codes, response, json_data):
    print("request_download")
    response["result"] = 1
    return response_download(command_codes, response)


def response_download(command_codes, response):
    print("response_download")
    func_str = response_download.__name__
    response["cmd"] = command_codes[func_str]


def request_model_select(command_codes, response, json_data):
    print("request_model_select")
    response["result"] = 1
    return response_model_select(command_codes, response)


def response_model_select(command_codes, response):
    print("response_model_select")
    func_str = response_model_select.__name__
    response["cmd"] = command_codes[func_str]


def request_current_model(command_codes, response, json_data):
    print("request_current_model")
    response["result"] = 1
    return response_current_model(command_codes, response)


def response_current_model(command_codes, response):
    print("response_current_model")
    func_str = response_current_model.__name__
    response["cmd"] = command_codes[func_str]


def request_model_change(command_codes, response, json_data):
    print("request_model_change")
    response["result"] = 1
    return response_model_change(command_codes, response)


def response_model_change(command_codes, response):
    print("response_model_change")
    func_str = response_model_change.__name__
    response["cmd"] = command_codes[func_str]


def request_model_delete(command_codes, response, json_data):
    print("request_model_delete")
    response["result"] = 1
    return response_model_delete(command_codes, response)


def response_model_delete(command_codes, response):
    print("response_model_delete")
    func_str = response_model_delete.__name__
    response["cmd"] = command_codes[func_str]
