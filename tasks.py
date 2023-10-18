from asyncio import Task, get_event_loop
from tcpip import Server
import time
from model import DLModel
from utils import *
import asyncio


async def start():
    print("Start")
    DLModel.name = "test"
    DLModel.dir = "test/model.properties"
    await asyncio.sleep(10)
    print("Done Start")
    return 10

def start_model():
    print("Start Model")
    return Task(start(), name="start")

async def resume():
    print("Resume")
    pass


def suspend():
    print("Suspend")
    pass


async def stop():
    print("Stop")
    pass


async def request_detective():
    print("Request Detective")
    time.sleep(10)
    return 3  # result


def response_detective(detective_task):
    print("Response Detective")
    if detective_task.done():
        print(detective_task.result())
        return detective_task.result()
    else:
        return 1


async def request_device_change():
    print("Request Device Change")
    pass


def response_device_change():
    print("Response Device Change")
    pass


async def request_model_change():
    print("Request Model Change")
    pass


def response_model_change():
    print("Response Model Change")
    pass


async def request_model_name():
    print("Request Model Name")
    pass


def response_model_name():
    print("Response Model Name")
    pass


async def request_model_delete():
    print("Request Model Delete")
    pass


def response_model_delete():
    print("Response Model Delete")
    pass


async def request_download():
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

    def assign_task(self, task_name, func):
        self.tasks[task_name] = Task(func(), name=task_name)
        # self.tasks[task_name] = self.loop.create_task(func())
    def check_task(self):
        print("Number of tasks: ", len(self.tasks))
        for name, task in self.tasks.items():
            print(f"Task name:", task.get_name())
            print(f"Task {name} is stopped: ", task.done())
            if task.done():
                print(f"Task result", task.result())

    def distribute_task(self, data):
        print("Distribute Task")
        response = {"cmd": "",
                    "result": ""}
        try:
            # check format of data
            data = convert_str_to_dict(data)
            print("parse:", data)
            command_type = data.get("cmd", None)
            if command_type is not None:
                response["cmd"] = command_type
                # check command type
                func_str = self.config["COMMAND"][command_type]
                task = eval(func_str)
                if "response" in func_str:  # checking progress of func or getting result
                    if self.tasks.get(func_str, None) is None:  # if task is not requested before
                        data["cmd"] = 0x29
                    else:  # if task is requested before
                        data["result"] = task(self.tasks[func_str])
                else:  # assign the task to run background
                    self.assign_task(func_str, task)

            else:
                print("Received invalid JSON")
                response["cmd"] = 0x2A
        except json.JSONDecodeError:
            print("Received invalid JSON")
            response["cmd"] = 0x2A
        # except Exception as e:
        #     print(f"Error handling request: {e}")
        #     response["cmd"] = 0x29
        finally:
            # response to client
            print("Distribute Task Done")
            return response
