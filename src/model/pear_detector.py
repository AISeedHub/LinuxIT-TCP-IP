import torch
import numpy as np
import logging
import cv2
import os
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional, Tuple
from src.utils.exceptions import ModelError, InferenceError

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
    is_defective: bool
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
                return DetectionResult(Error(non_img=True), is_defective=False, confidence=0.0, bbox=None)
            img = cv2.imread(dir_img)
            return await self.detect(img)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return DetectionResult(Error(error=True, non_detect=True), is_defective=False, confidence=0.0, bbox=None)

    async def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise ModelError("Model not loaded")

        try:
            # Preprocess image
            img = self._preprocess_image(image)

            # fix return output (temporary
            import random
            return DetectionResult(
                    error=Error(),
                    is_defective=random.choice([True, False]),
                    confidence=0.0,
                    bbox=None
                )

            # Run inference
            # TODO: check the model output
            with torch.no_grad():
                results = self.model(img)

            # Process results
            predictions = results.pred[0].cpu().numpy()

            # pred = (
            #     predictions[predictions.conf > self.config.confidence_threshold]
            #     if all([pred != "burn_bbox" for pred in self.names])
            #     else predictions[predictions.conf > 0.7]
            # )
            # labels = [self.names[int(cat)] for cat in pred.cls]
            #
            # if any([label == "burn_bbox" for label in labels]):
            #     return 1
            # else:
            #     return 0

            if len(predictions) == 0:
                return DetectionResult(
                    is_defective=True,  # No detection usually means defective
                    confidence=0.0,
                    bbox=None
                )

            # Get highest confidence prediction
            best_pred = predictions[predictions[:, 4].argmax()]

            return DetectionResult(
                is_defective=bool(best_pred[5] == 0),  # Assume class 0 is defective
                confidence=float(best_pred[4]),
                bbox=tuple(best_pred[:4])
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise InferenceError(f"Detection failed: {e}")

    def _preprocess_image(self, image: np.ndarray):  # -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return img
        # # Convert BGR to RGB
        # image = image[:, :, ::-1]
        #
        # # Resize
        # image = cv2.resize(image, (self.config.img_size, self.config.img_size))
        #
        # # Normalize and convert to tensor
        # image = image / 255.0
        # image = torch.from_numpy(image).permute(2, 0, 1).float()
        # image = image.unsqueeze(0).to(self.device)
        #
        # return image

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded")
