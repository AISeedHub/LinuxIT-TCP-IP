import torch
from torchvision import transforms
from ultralytics import YOLO
import timm
from typing import Tuple,Optional
import numpy as np
import time
import sys
import cv2
from PIL import Image

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
            self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=config.num_classes)
            prev_ = torch.load(config.model_path, map_location='cpu')
            ckpt = prev_.state_dict()
            self.model.load_state_dict(ckpt)
            self.model.to(self.device)
            self.model.eval()

            self.preprocessor = YOLO(config.preprocessor_path, task="detect")
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

            return np.hstack((bboxes, classes[:, None]))
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return np.array([])

    def predict(self, img: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        This is the main function to run inference on the input image. This function will be called outside the class.
        """
        return self.__one_step_inference(img)
        
    def __crop_object_area(self, img: np.ndarray, conf: float) -> Optional[np.ndarray]:
        """
        Finds the first class-0 bounding box (after confidence filtering)
        and returns a single cropped image for that region.

        Args:
            img (np.ndarray): Input image in BGR format (H x W x 3).
            conf (float): Confidence threshold.

        Returns:
            np.ndarray | None: Cropped image corresponding to the selected box.
                            Returns None if no valid class-0 box is found.
        """
        try:
            detections = self.__detect(img, conf)
            # detections shape: (N, 5) → [x1, y1, x2, y2, cls]
            if detections is None or detections.size == 0:
                self.logger.warning("No detections found.")
                return None

            # Filter class-0 detections
            cls_ids = detections[:, 4].astype(int)
            mask_cls0 = cls_ids == 0
            cls0_dets = detections[mask_cls0]

            if cls0_dets.size == 0:
                self.logger.warning("No class-0 detections found.")
                return None

            # Select the first class-0 box
            best_box = cls0_dets[0, :4].astype(int)  # [x1, y1, x2, y2]

            x1, y1, x2, y2 = best_box

            # Clamp to image bounds
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                self.logger.error(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")
                return None

            crop = img[y1:y2, x1:x2].copy()
            return crop

        except Exception as e:
            self.logger.error(f"Cropping error: {e}")
            return None


    def __one_step_inference(self, img: np.ndarray) -> Tuple[bool, int]:
        """Run one step inference and return classification-based result
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[bool, np.ndarray]: (is_normal, pred_array) where pred_array holds class index.
        """
        croped_area = self.__crop_object_area(img, self.confidence)

        # ----------------------------------------
        # 1) If no valid cropped area: use default (assume normal class 0)
        # ----------------------------------------
        if croped_area is None:
            self.logger.log("No valid cropped area found. Using default prediction.", level="WARNING")
            # Treat class 0 as normal
            return True, np.array([0], dtype=np.int32)

        else:
            # If a list is returned, use the first element
            if isinstance(croped_area, list):
                if len(croped_area) == 0:
                    self.logger.log("Empty crop list. Using default prediction.", level="WARNING")
                    return True, 0
                else:
                    crop_img = croped_area[0]
            else:
                crop_img = croped_area

            # Step 2: Convert BGR ndarray -> RGB -> PIL -> transform -> Tensor
            img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            input_tensor = self.transform(pil_img)      # (3, 512, 512)
            input_tensor = input_tensor.unsqueeze(0)    # (1, 3, 512, 512)
            input_tensor = input_tensor.to(self.device)

            # Step 3: Run classification and decide normal/abnormal by class index
            with torch.no_grad():
                logits = self.model(input_tensor)       # assume (1, num_classes)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                probs = torch.softmax(logits, dim=1)[0] # (num_classes,)

            class_idx = int(torch.argmax(probs).item())

            # Log predicted class index
            self.logger.log(f"Predicted class index: {class_idx}", level="INFO")

            # Class 0 is normal; others are abnormal
            is_normal = (class_idx == 0)
            return is_normal, class_idx

