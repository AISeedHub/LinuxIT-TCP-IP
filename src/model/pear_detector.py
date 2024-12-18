import torch
import numpy as np
import logging
import cv2
import os
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional, Tuple
from src.utils.exceptions import ModelError, InferenceError
from src.utils.visualization import save_predictions_image

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str
    img_path: str
    confidence_threshold: float = 0.9
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size: int = 640
    classes: Tuple[str] = ('defective', 'non-defective')


@dataclass
class Error:
    error: bool = False
    non_model: bool = False
    non_img: bool = False
    non_detect: bool = False


@dataclass
class DetectionResult:
    error: Error
    is_normal: int
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]


class PearDetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device(config.device)

    def change_model(self, model_path: str):
        bk_model = self.model
        bk_model_name = self.config.model_path
        try:
            self.config.model_path = model_path
            self.load_model()
            logger.info(f"Model changed to {model_path}")
        except Exception as e:
            self.model = bk_model
            self.config.model_path = bk_model_name
            logger.error(f"Failed to change model: {e}")
            raise ModelError(f"Model change failed: {e}")

    def load_model(self, model_path: Optional[str] = None):
        try:
            path = model_path or self.config.model_path
            # self.model = None
            self.model = YOLO(path, task="detect")
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {path}")
            logger.info(f"Model device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Model loading failed: {e}")

    # inferencing
    async def inference(self, img_path) -> DetectionResult:
        try:
            dir_img = os.path.join(self.config.img_path, img_path)
            if os.path.exists(dir_img) is False:
                logger.error(f"Image not found: {dir_img}")
                return DetectionResult(Error(non_img=True),
                                       is_normal=False, confidence=0.0, bbox=None)
            img = cv2.imread(dir_img)
            return await self.detect(img)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return DetectionResult(Error(error=True, non_detect=True), is_normal=False, confidence=0.0, bbox=None)

    async def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise ModelError("Model not loaded")

        try:
            # Run inference
            with torch.no_grad():
                results = self.model.predict(image)

            # Process results
            pred = results[0].boxes.cpu().numpy()
            pred = pred[pred.conf > self.config.confidence_threshold]
            predictions = np.array(pred.data)

            # Save image with predictions
            await save_predictions_image(image, predictions)

            best_pred = predictions[predictions[:, 5] == 0]

            if len(best_pred) == 0:
                return DetectionResult(
                    error=Error(),
                    is_normal=1,  # No detection usually means 1
                    confidence=0.0,
                    bbox=None
                )
            else:
                best_pred = best_pred[0]  # only one detection
                return DetectionResult(
                    error=Error(),
                    is_normal=0,  # Defective
                    confidence=float(best_pred[4]),
                    bbox=tuple(best_pred[:4])
                )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise InferenceError(f"Detection failed: {e}")

    def _preprocess_image(self, image: np.ndarray):  # -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return img

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def __str__(self):
        str_format = f"Model path: {self.config.model_path}\n" \
                     f"Image path: {self.config.img_path}\n" \
                     f"Confidence threshold: {self.config.confidence_threshold}\n" \
                     f"Device: {self.config.device}\n" \
                     f"Image size: {self.config.img_size}\n" \
                     f"Classes: {self.config.classes}"
        return str_format
