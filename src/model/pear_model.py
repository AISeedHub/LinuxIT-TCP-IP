import torch
from torchvision import transforms
from ultralytics import YOLO
import timm
from typing import Tuple,Optional
import numpy as np
import time
import sys
from PIL import Image
import os

from src.model import ModelConfig

class Logger:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = time.time() - self.start_time
        print(f"[{timestamp:.2f}s] {level} - {self.name}: {message}")
        sys.stdout.flush()


class PearModel:
    """
    A class for detecting pears and their defects using a YOLO model.
    Attributes:
        preprocessor: Preprocessor for the model. This model first detect pear area and crop for better performance of 
        classification model.
        model: Classification model for classifying defects.
        names (list): List of class names for the model.
    Args:
        config (dict): Configuration dictionary containing model path and class names.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.logger = Logger("Model")
        self.logger.log("Initializing model...")

        self.device = config.device
        self.logger.log(f"Using device: {self.device}")

        try:
            self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=8)
            #self.model = timm.create_model('convnextv2_base', pretrained=False, num_classes=8)
            state_dict = torch.load(config.model_path, map_location=self.device, weights_only=False).state_dict()
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # check existence of preprocessor model
            if os.path.exists("../weights/best-2cls.pt"):

                self.preprocessor = YOLO("../weights/best-2cls.pt", task="detect")
                self.preprocessor.to(self.device)
                self.preprocessor.eval()
            else:
                raise FileNotFoundError("Preprocessor model file not found.")
            
            self.names = config.classes
            self.confidence = config.confidence_threshold

            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])

            self.logger.log("Model loaded successfully")
            self.logger.log(f"Classes: {self.names}")
        except Exception as e:
            self.logger.log(f"Error loading model: {e}", "ERROR")
            raise

    def predict(self, img: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        This is the main function to run inference on the input image. This function will be called outside the class.
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[int, np.ndarray]: (class_idx, pred_array) : where -1 indicates no detection.
        """
        return self.__inference(img)
    
    def __detect(self, img: np.ndarray, conf: float) -> np.ndarray:
        """Run detection on image for pear area detection
        Args:
            img (np.ndarray): Input image in BGR format.
            conf (float): Confidence threshold for detection.
        Returns:
            np.ndarray: Array of detected bounding boxes and classes.
                    Shape: (N, 5) → [x1, y1, x2, y2, cls]
        """
        try:
            results = self.preprocessor.predict(img)
            # Extract bounding boxes, classes, and scores
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            # Filter results based on confidence threshold
            mask = scores >= conf
            bboxes = bboxes[mask]
            classes = classes[mask]
            scores = scores[mask]

            return np.hstack((bboxes, classes[:, None], scores[:, None]))
        except Exception as e:
            self.logger.log(f"Detection error: {e}", level="ERROR")
            return np.array([])

    def __crop_object_area(self, img: np.ndarray, conf: float) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Finds the first class-0 bounding box (after confidence filtering)
        and returns a single cropped image for that region.

        Args:
            img (np.ndarray): Input image in BGR format (H x W x 3).
            conf (float): Confidence threshold.

        Returns:
            Tuple[np.ndarray | None, np.ndarray]: Only 1 cropped image corresponding to the selected box and its bounding box.
                            Returns (None, empty array) if no valid class-0 box is found.
        """
        try:
            detections = self.__detect(img, conf)
            # detections shape: (N, 6) → [x1, y1, x2, y2, cls, score]
            detections = detections.astype(int)
            # Select the first class-0 box
            x1, y1, x2, y2 = detections[detections[:, 4] == 0][0, :4]  # [x1, y1, x2, y2]
            score = detections[detections[:, 4] == 0][0, 5]
            crop = img[y1:y2, x1:x2].copy()

            return crop, np.array([x1, y1, x2, y2, score])

        except Exception as e:
            self.logger.log(f"Cropping error: {e}", level="ERROR")
            return None, np.array([])
        
    def __classify(self, img: np.ndarray) -> int:
        """Run classification on the input image
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            int: Class index.
        """
     
        pil_img = Image.fromarray(img).convert("RGB")
        
        input_tensor = self.transform(pil_img)      # (3, 512, 512)
        input_tensor = input_tensor.unsqueeze(0)    # (1, 3, 512, 512)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)       # assume (1, num_classes)
            # if logits.ndim == 1:
            #     logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)[0] # (num_classes,)

        probs[0] = probs[0] * 2.0
        probs = probs / probs.sum() # re-normalize confidences of normal class
   
        class_idx = int(torch.argmax(probs).item())
        return class_idx

    def __inference(self, img: np.ndarray) -> Tuple[int, np.ndarray]:
        """Run one step inference and return classification-based result
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[int, np.ndarray]: (class_idx, pred_array) where pred_array holds class index.
        """
        cropped, bbox = self.__crop_object_area(img, self.confidence)

        if cropped is None:
            self.logger.log("No valid cropped area found.", level="ERROR")
            
            return -1, np.array([])

        class_idx = self.__classify(cropped)
        # Log predicted class index
        self.logger.log(f"Predicted class index: {class_idx}", level="INFO")
        
        return class_idx, bbox