from asyncio import Task, get_event_loop
from tcpip import Server
import time
import json
from utils import *
import asyncio


def start():
    print("Start")
    time.sleep(5)
    print("Done Start")
    return 10


def resume():
    print("Resume")
    pass


def suspend():
    print("Suspend")
    pass


def stop():
    print("Stop")
    pass


async def request_detective():
    print("Request Detective")
    time.sleep(10)


def response_detective():
    print("Response Detective")
    pass


def request_device_change():
    print("Request Device Change")
    pass


def response_device_change():
    print("Response Device Change")
    pass


def request_model_change():
    print("Request Model Change")
    pass


def response_model_change():
    print("Response Model Change")
    pass


def request_model_name():
    print("Request Model Name")
    pass


def response_model_name():
    print("Response Model Name")
    pass


def request_model_delete():
    print("Request Model Delete")
    pass


def response_model_delete():
    print("Response Model Delete")
    pass


def request_download():
    print("Request Download")
    pass


def response_download():
    print("Response Download")
    pass


class Manager:
    """
    - Task Manager: Coordinate and manage actions between the server and the DL model.
    - Only one task is allowed to be performed at a time, the process is updated in the `state`
    - Manager class should be initial when the server starts
    """

    def __init__(self, cnf, loop):
        self.cmd = cnf["COMMAND"].keys()
        self.config = cnf
        self.loop = loop
        self._state = 0
        self.server = None  # TCP-IP server
        self.tasks = {}  # {name: Task}
        self.dl_model = None  # adding DL model here

    def start_server(self):
        self.server = Server(self)
        # Register the input listener task
        self.tasks["input_listener"] = self.server.start_input()
        self.tasks["server"] = self.server.start()

    async def distribute_task(self, data):
        print("Distribute Task")
        await asyncio.sleep(3)
        try:
            # check format of data
            data = convert_str_to_dict(data)
            command_type = data.get("cmd", None)
            if command_type is not None:
                # check command type
                func_str = self.config["COMMAND"][command_type]
                task = eval(func_str)
                result = self.execute_task(task)
                response = {"cmd": command_type,
                            "result": "Success"}

                # Add more command_type cases here
                # check state
            time.sleep(5)
        except json.JSONDecodeError:
            print("Received invalid JSON")
            # TODO: make the ERROR CODE
        except Exception as e:
            print(f"Error handling request: {e}")
            # TODO: make the ERROR CODE
        # response to client
        print("Distribute Task Done")

    def execute_task(self, task):
        result = task()
        return result
