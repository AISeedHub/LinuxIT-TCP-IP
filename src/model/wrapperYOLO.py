import torch
from torchvision import transforms
from ultralytics import YOLO
import timm
from typing import Tuple
import numpy as np
import time
import sys

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
            self.model.to(self.device)
            self.model.eval()
            self.preprocessor = YOLO(config["model_path"], task="detect")
            self.preprocessor.to(self.device)
            self.preprocessor.eval()
            self.names = config.classes
            self.confidence = config.confidence_threshold

            self.transform = transforms.Compose([
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])

            self.logger.log("Model loaded successfully")
            self.logger.log(f"Classes: {self.names}")
        except Exception as e:
            self.logger.log(f"Error loading model: {e}", "ERROR")
            raise
    
    def __crop_by_bounding_boxes(
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """
        Crop image by given bounding boxes.

        Args:
            image (np.ndarray): Input image in BGR format
            bboxes (List[Tuple[int, int, int, int]]): List of bounding boxes in format (x1, y1, x2, y2)
            save_dir (str, optional): Directory to save cropped images. If None, only returns crops.
            image_name (str): Base name for saved images

        Returns:
            List[np.ndarray]: List of cropped image regions
        """
        crops = []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Crop the image
            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        return crops

    def __detect(self, img: np.ndarray, conf: float) -> np.ndarray:
        """Run detection on image for pear area detection
        Args:
            img (np.ndarray): Input image in BGR format.
            conf (float): Confidence threshold for detection.
        Returns:
            np.ndarray: Array of detected bounding boxes, classes, and scores.
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

            return np.hstack((bboxes, classes[:, None]))
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return np.array([])


    def __crop_object_area(self, img: np.ndarray, conf: float) -> list[np.ndarray]:
        """Detect and crop pears from the image
        Args:
            img (np.ndarray): Input image in BGR format.
            conf (float): Confidence threshold for detection.
        Returns:
            List[np.ndarray]: List of cropped pear images.
        """
        try:
            detections = self.__detect(img, conf)
            bboxes = detections[:, :4].astype(int).tolist()
            crops = __crop_by_bounding_boxes(img, bboxes)
            return crops
        except Exception as e:
            self.logger.error(f"Cropping error: {e}")
            return []
    

    def __post_process(self, pred: np.ndarray) -> np.ndarray:
        """Post-process the predictions"""
        # ensure that defect boxes are inside the fruit boxes
        # extract the defect boxes and fruit boxes

        return pred

    def predict(self, img: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        This is the main function to run inference on the input image. This function will be called outside the class.
        """
        return self.__one_step_inference(img)

    def __one_step_inference(self, img: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Run one step inference and return the predictions
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[int, np.ndarray]: A tuple containing the result (1 for defect detected, 0 for no defect) and the predictions.
        """
        croped_area = self.__crop_object_area(img, self.confidence)

        pred = self.model(self.transform(croped_area))
        # Post-process the predictions
        pred = self.__post_process(pred)

        labels = [self.names[int(cat)] for cat in pred[:, 4]]

        if any([label == "defect" for label in labels]):
            return False, pred  # Return False if any defect is detected
        else:
            return True, pred  # Return True if no defect is detected

    def __two_step_inference(self, img: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Run inference and return result and boxes
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[int, np.ndarray]: A tuple containing the result (1 for defect detected, 0 for no defect) and the predictions."""
        pred = self.detect(img)
        # check the fruit boxes appeared in the image, if yes, crop the fruit boxes and run prediction on the cropped image
        for box in pred:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1, x2, y2 = x1 - 10, y1 - 10, x2 + 10, y2 + 10  # Add padding
            if box[4] == 0:  # Assuming class 0 is the fruit class
                fruit_box = img[y1:y2, x1:x2]
                fruit_pred = self.detect(fruit_box)
                # Check if any defect is detected in the fruit box, if yes, recalibrate the defect boxes
                if len(fruit_pred) > 0:
                    defect_boxes = fruit_pred[fruit_pred[:, 4] == 1]
                    if len(defect_boxes) > 0:
                        defect_boxes[:, :4] += np.array([x1, y1, x1, y1])
                        pred = np.vstack((pred, defect_boxes))  # Append defect boxes to the original prediction

        labels = [self.names[int(cat)] for cat in pred[:, 4]]

        if any([label == "defect" for label in labels]):
            return False, pred
        else:
            return True, pred
