## Installation
- Navigate to the directory containing your requirements.txt file. Run the following command:
```
pip install -r requirements.txt
```


## Run the Application

### Server:
- Configurate server `config/server-config.yaml`
+ `PORT` and `IP` must check carefully
```yaml
PORT: 9090
IP: 192.168.0.40
```
- Run `python main.py` to start the Server service , expected output:
```shell
$ python main.py 
Loading the server's configuration file...
Loading the model's configuration file...
weights/best.pt
Using cache found in /home/linuxit/.cache/torch/hub/ultralytics_yolov5_master
YOLOv5 🚀 2023-12-6 Python-3.9.18 torch-2.1.1+cu121 CUDA:0 (NVIDIA GeForce RTX 4070 Ti, 12007MiB)

Fusing layers... 
YOLOv5s summary: 157 layers, 7023610 parameters, 0 gradients, 15.8 GFLOPs
Adding AutoShape... 
Loaded model successfully
Server Address: 192.168.0.40:9090
Start the server...
Waiting for connection...
```
### Client 
- Establish connection to server:
> In Server, expected output:
```shell
WELLCOME bro: (('192.168.0.22', 64438))
Waiting for connection...
```
- Request server
```
{"cmd" : 0x01, "request_data": ["test.jpg", "test1.jpg", "test2.jpg"]}
```

> In Server, expected output:
```shell
Peer handler
<----------- ('192.168.0.22', 64438): {"cmd" : 0x01, "request_data": ["test.jpg", "test1.jpg", "test2.jpg"]}
Distribute Task
parse: {'cmd': 1, 'request_data': ['test.jpg', 'test1.jpg', 'test2.jpg']}
request_classification
Distribute Task Done
json_response:  {"cmd": 2, "response_data": [{"file_name": "test.jpg", "result": 5, "error_code": 0}, {"file_name": "test1.jpg", "result": 6, "error_code": 0}, {"file_name": "test2.jpg", "result": 6, "error_code": 0}]}
Broadcasting message...
----------> ('192.168.0.22', 64438) : "{"cmd": 2, "response_data": [{"file_name": "test.jpg", "result": 5, "error_code": 0}, {"file_name": "test1.jpg", "result": 6, "error_code": 0}, {"file_name": "test2.jpg", "result": 6, "error_code": 0}], "request_data": null}"
Done sending
```

***

## Other request:
#### Model Download Request
```
{"cmd" : 0x20, "request_data": ["https://github.com/AISeedHub/pretrained-models/releases/download/PearDetection/best.pt"]}
```
Expected Response:
```
"{"cmd": 33, "response_data": [{"file_name": null, "result": 2, "error_code": 0}], "request_data": null}"
```

#### Current Model Request
```
{"cmd" : 0x22, "request_data": null}
```
Expected Response:
```
"{"cmd": 35, "response_data": [{"file_name": null, "result": "best.pt", "error_code": 0}], "request_data": null}"
```


#### List All Model Request
```
{"cmd" : 0x24, "request_data": null}
```
Expected Response:
```
"{"cmd": 37, "response_data": [{"file_name": "best.pt", "result": 2, "error_code": 0}, {"file_name": "best (1).pt", "result": 2, "error_code": 0}, {"file_name": "bk.pt", "result": 2, "error_code": 0}], "request_data": null}"
```

#### Change Model Request
```
{"cmd" : 0x26, "request_data": ["bk.pt"]}
```
Expected Response:
```
"{"cmd": 39, "response_data": [{"file_name": "bk.pt", "result": 2, "error_code": 0}], "request_data": null}"
```

#### Current Img Folder Request
```
{"cmd" : 0x32, "request_data": null}
```
Expected Response:
```
"{"cmd": 51, "response_data": [{"file_name": null, "result": "img", "error_code": 0}], "request_data": null}"
```

#### Change Img Folder Request
```
{"cmd" : 0x30, "request_data": ["img2"]}
```
Expected Response:
```
"{"cmd": 49, "response_data": [{"file_name": "img2", "result": 2, "error_code": 0}], "request_data": null}"
```

#### Delete Model Request
```
{"cmd" : 0x28, "request_data": ["best (1).pt"]}
```
Expected Response:
```
"{"cmd": 41, "response_data": [{"file_name": "best (1).pt", "result": 2, "error_code": 0}], "request_data": null}"
```

##### @opyRight: AISeedCorp


