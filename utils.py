import yaml
import json
import re

from emit import *


# load the configuration YAML file: server-config.yaml
def load_server_config():
    print("Loading the server's configuration file...")
    with open("config/server-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["IP"] = get_ip_address()
    return config


def load_model_config():
    print("Loading the model's configuration file...")
    with open("config/yolo-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def create_json_format(data):
    return json.dumps(data)


def convert_str_to_dict(data) -> dict:
    # Convert '0x' hexadecimal values to decimal
    def convert_hex_to_dec(match):
        hex_value = match.group(0)
        return str(int(hex_value, 16))

    input_string = re.sub(r'0x[0-9A-Fa-f]+', convert_hex_to_dec, data)

    # Now parse the modified string
    return json.loads(input_string)


def get_ip_address():
    """Get the owner ip address"""
    print("Getting the local IP address...")
    return "127.0.0.1"
    try:
        import socket

        def get_local_ip():
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address

        local_ip = get_local_ip()
        print("Local IP address:", local_ip)
        return local_ip

    except Exception as e:
        print(e)
        print("Cannot get the local IP address")
        print("Using the default IP address: 127.0.0.1")
        return "127.0.0.1"


def validate_task(data, valid_cmd):
    # check format of data
    json_data = convert_str_to_dict(data)
    print("parse:", json_data)
    command_type = json_data["cmd"]
    # check command type
    task_str = valid_cmd[command_type]
    task_func = eval(task_str)
    return task_func, json_data
