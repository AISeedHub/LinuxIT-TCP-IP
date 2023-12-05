import os
import cv2
import torch
import yaml


class PearDetectionModel:
    def __init__(self, config) -> None:
        self.__dir_model_detection = config["DIR_MODEL_DETECTION"]
        self.__dir_img = config["DIR_IMG"]
        self.model_name = None
        self.model = None

        # self._download_weight() # for testing

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.__load_model(config['model_name']) # for testing
        self.names = config['classes']

    def set_model_name(self, model_name):
        self.model = self.__load_model(model_name)
        self.model_name = model_name

    def __load_model(self, model_name):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.__dir_model_detection + '/' + model_name)
        model.to(self.device)
        return model

    def _download_weight(self):  # for testing
        if os.path.exists(self.__dir_model_detection + '/' + config['model_name']) is False:
            if os.path.exists(config['dir_model_detection']) is False:
                os.mkdir(config['dir_model_detection'])
            self.__dir_model_detection = self.__dir_model_detection + '/' + config['model_name']
            torch.hub.download_url_to_file(config['weight_url'], dst=self.__dir_model_detection)
        else:
            self.__dir_model_detection = self.__dir_model_detection + '/' + config['model_name']

    def __load_img(self, img_path):
        return cv2.imread(self.__dir_img + "/" + img_path)

    def detect(self, img):
        img = self._preporcess(img)
        results = self.model(img)
        return results.pred[0].numpy()

    def inference(self, file_name):
        try:
            img = self.__load_img(file_name)
        except:
            return -1  # no img file
        pred = self.detect(img)
        labels = [self.names[int(cat)] for _, _, _, _, _, cat in pred]
        if 'normal_pear_bbox' in labels:
            return 1
        else:
            return 0

    def ease_model(self):
        self.model_name = None
        self.model = None

    def _preporcess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


# for testing
if __name__ == '__main__':
    with open('config/yolo_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    config['DIR_MODEL_DETECTION'] = './weights'
    config['DIR_IMG'] = './img'

    model = PearDetectionModel(config)
    model.set_model_name(config['model_name'])

    # loop the list of images to do inference
    for file_name in os.listdir(config['DIR_IMG']):
        print(model.inference(file_name))
