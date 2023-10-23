import yaml
import json
import re

from emit import *

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2



# load the configuration YAML file: server-config.yaml
def load_config():
    print("Loading the configuration file...")
    with open("server-config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["IP"] = get_ip_address()
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



def load_image(file_or_url_or_path):
    return Image.open(file_or_url_or_path).convert('RGB')

def convert_image(pillow_image, input_size=320):
    pillow_image = pillow_image.convert("RGB")  # Convert to RGB
    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    image_np = np.array(pillow_image)
    transformed = transform(image=image_np)
    return transformed["image"].unsqueeze(0)

def segmentaion_to_crop(image, mask_np, out_name):
    mask_np = (mask_np.squeeze() * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_np, (image.width, image.height))
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        image.save(out_name)
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    img_np = np.array(image)
    img_np = cv2.merge([img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], mask_resized])
    cropped_image_np = img_np[y:y+h, x:x+w]
    cropped_image = Image.fromarray(cropped_image_np).resize((320, 320))
    return cropped_image



def segmentaion_to_crop(image, mask_np):
    mask_np = (mask_np.squeeze() * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_np, (image.width, image.height))
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    img_np = np.array(image)
    img_np = cv2.merge([img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], mask_resized])
    cropped_image_np = img_np[y:y+h, x:x+w]
    cropped_image = Image.fromarray(cropped_image_np).resize((320, 320))
    return cropped_image