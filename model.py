import numpy as np
import onnxruntime as ort
from utils import segmentaion_to_crop, load_image, convert_image


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

    


