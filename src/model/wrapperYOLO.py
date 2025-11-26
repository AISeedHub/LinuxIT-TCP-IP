import torch
from ultralytics import YOLO
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
        model (YOLO): The YOLO model for detection.
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
            self.model = YOLO(config.model_path, task="detect")
            self.names = config.classes
            self.confidence = config.confidence_threshold
            self.logger.log("Model loaded successfully")
            self.logger.log(f"Classes: {self.names}")
        except Exception as e:
            self.logger.log(f"Error loading model: {e}", "ERROR")
            raise

    def __detect_objects(self, img: np.ndarray) -> np.ndarray:
        """Run detection on image
        Args:
            img (np.ndarray): Input image in BGR format.
            conf (float): Confidence threshold for detection.
        Returns:
            np.ndarray: Array of detected bounding boxes, classes, and scores.
        """
        try:
            with torch.no_grad():
                results = self.model.predict(img)
            # Extract bounding boxes, classes, and scores
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            # Filter results based on confidence threshold
            mask = scores >= self.confidence
            bboxes = bboxes[mask]
            classes = classes[mask]
            scores = scores[mask]

            return np.hstack((bboxes, classes[:, None], scores[:, None]))
        except Exception as e:
            self.logger.log(f"Detection error: {e}", "ERROR")
            return np.array([])

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
        pred = self.__detect_objects(img)

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
