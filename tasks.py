from model import PearDetectionModel
from tcpip import Server
from util import *
from emit import *


class Manager:
    """
    - Task Manager: Coordinate and manage actions between the server and the DL model.
    - Only one task is allowed to be performed at a time, the process is updated in the `state`
    - Manager class should be initial when the server starts
    """

    def __init__(self, cnf):
        self.cmd = cnf["COMMAND"].keys()
        self.config = cnf
        self.server = None  # TCP-IP server
        self.command_codes = {value: key for key, value in self.config["COMMAND"].items()}
        self.dl_model = PearDetectionModel(self.config)

    def start_server(self):
        self.server = Server(self)
        self.server.start()

    def distribute_task(self, data):
        print("Distribute Task")

        response = {"cmd": None,
                    "response_data": [],
                    "request_data": None}

        try:
            task_func, json_data = validate_task(data, self.config["COMMAND"])
            # Doing request task
            response_data = task_func(self, json_data["request_data"])
            # Generate response
            response["cmd"] = self.config["CORRESPONDING_COMMAND"][json_data["cmd"]]
            response["response_data"] = response_data

        except Exception as e:
            print(f"Error handling request: {e}")
            # response["cmd"] = 0x29
            response = {"cmd": None,
                        "response_data": [{
                            "file_name": None,
                            "result": None,
                            "error_code": self.config["ERROR_CODE"]["not_exist_command"]
                        }],
                        "request_data": None}

        finally:
            # response to client
            print("Distribute Task Done")
            print(response)
            return json.dumps(response)
