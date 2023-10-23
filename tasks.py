from model import DLModel
from tcpip import Server
from utils import *
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
        self.dl_model = DLModel(self.config["DIR_MODEL_PREPROCESSING"])

    def start_server(self):
        self.server = Server(self)
        self.server.start()

    def distribute_task(self, data):
        print("Distribute Task")
        response = {"cmd": "",
                    "result": ""}
        try:
            task_func, json_data = validate_task(data, self.config["COMMAND"])
            raw_data_request = json_data["request_data"]
            task_func(self, response, json_data)

        except Exception as e:
            print(f"Error handling request: {e}")
            response["cmd"] = 0x29
        finally:
            # response to client
            print("Distribute Task Done")
            return json.dumps(response)
