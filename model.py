import numpy as np
import onnxruntime as ort
from utils import segmentaion_to_crop, load_image, convert_image

import cv2
import torch
import yaml

class DLModel:
    def __init__(self, dir_model_preprocessing, root_dir) -> None:
        self.model_name = None
        self.__dir_model_pre_process = dir_model_preprocessing
        self.root_dir = root_dir
        self.model_classification = None
        self.__model_pre_process = self.__load_model(self.__dir_model_pre_process)

    def set_model_name(self, model_name):
        self.model_classification = self.__load_model(model_name)
        self.model_name = model_name

    def __load_model(self, model_name):
        return ort.InferenceSession(self.root_dir + "/" + model_name)

    
    def _preprocessing(self, filename):
        pillow_image = load_image(filename)
        torch_image = convert_image(pillow_image)
        d1 = self.__model_pre_process.run(None, {'input_image': torch_image.numpy()})[0]
        return segmentaion_to_crop(pillow_image, d1[0,0,:,:])

    def inference(self, filename):
        if self.model_classification is None:
            return -1
        
        seg_result = self._preprocessing(filename)
        ort_inputs = {self.model_classification.get_inputs()[0].name: convert_image(seg_result).numpy()}
        ort_outs = self.model_classification.run(None, ort_inputs)
        class_idx = np.argmax(ort_outs[0], axis=1)[0]

        # TODO: Post-processing
        return 1
        # if class_idx == 0:
        #     return 'NG'
        # else:
        #     return 'OK'
    
    def ease_model(self):
        self.model_name = None
        self.model_classification = None

class PearDetectionModel():

    def __init__(self, config) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=config['model_path'])
        self.model.to(self.device)

        self.names = config['classes']

    def detect(self, img):
        img = self._preporcess(img)
        results = self.model(img)
        return results.pred[0].numpy()
    
    def inference(self, img):
        pred = self.detect(img)
        labels = [self.names[int(cat)] for _,_,_,_,_,cat in pred]
        if 'normal_pear_bbox' in labels:
            return 1
        else:
            return 0
    
    def _preporcess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    
if __name__ == '__main__':

    with open('yolo_config.yml', 'r') as f:
        config = yaml.safe_load(f)

    model = PearDetectionModel(config)
    for image in []:
        img = cv2.imread(image)
       
        print(model.inference(img))

