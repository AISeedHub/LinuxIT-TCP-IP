import os
import cv2
import torch
import yaml


class PearDetectionModel:
    def __init__(self, config) -> None:
        self.__dir_model_detection = config["DIR_MODEL_DETECTION"]
        self.__dir_img = config["DIR_IMG"]
        self.__default_weight_url = config["weight_url"]
        self.model_name = None
        self.model = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # initialize
        self._download_weight(config["model_name"])  # for testing
        self.set_model_name(config["model_name"])

        self.names = config['classes']

    def set_model_name(self, model_name):
        self.model = self.__load_model(model_name)
        self.model_name = model_name

    def set_dir_img(self, directory):
        self.__dir_img = directory

    def get_current_img_dir(self):
        return self.__dir_img

    def __load_model(self, model_name):
        print(self.__dir_model_detection + '/' + model_name)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.__dir_model_detection + '/' + model_name)
        model.to(self.device)
        print("Loaded model successfully")
        return model

    def _download_weight(self, model_name):  # for testing
        if os.path.exists(self.__dir_model_detection + '/' + model_name) is False:
            if os.path.exists(self.__dir_model_detection) is False:
                os.mkdir(self.__dir_model_detection)
            torch.hub.download_url_to_file(self.__default_weight_url, dst=self.__dir_model_detection + '/' + model_name)

    def __load_img(self, img_path):
        return cv2.imread(self.__dir_img + "/" + img_path)

    def detect(self, img):
        img = self._preporcess(img)
        results = self.model(img)
        return results.pred[0].cpu().numpy()

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
    with open('config/yolo-config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['DIR_MODEL_DETECTION'] = './weights'
    config['DIR_IMG'] = './img'

    model = PearDetectionModel(config)
    model.set_model_name(config['model_name'])

    # loop the list of images to do inference
    for file_name in os.listdir(config['DIR_IMG']):
        print(model.inference(file_name))
