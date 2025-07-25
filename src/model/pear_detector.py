import torch
import numpy as np
import logging
import cv2
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from .PearDectionModel import PearDetectionModel
from src.utils.exceptions import ModelError, InferenceError
from src.utils.visualization import save_predictions_image

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str
    classes: list[str]
    confidence_threshold: float = 0.5

@dataclass
class Error:
    error: bool = False
    non_model: bool = False
    non_img: bool = False
    non_detect: bool = False


@dataclass
class DetectionResult:
    error: Error
    is_normal: bool
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]


class PearDetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = PearDetectionModel({
            "model_path": config.model_path,
            "classes": config.classes,
            "confidence": config.confidence_threshold
        })
        # self.device = torch.device(config.device)

    # def change_model(self, model_path: str):
    #     bk_model = self.model
    #     bk_model_name = self.config.model_path
    #     try:
    #         self.config.model_path = model_path
    #         self.load_model()
    #         logger.info(f"Model changed to {model_path}")
    #     except Exception as e:
    #         self.model = bk_model
    #         self.config.model_path = bk_model_name
    #         logger.error(f"Failed to change model: {e}")
    #         raise ModelError(f"Model change failed: {e}")


    # inferencing
    async def inference(self, img_path) -> DetectionResult:
        try:
            if os.path.exists(img_path) is False:
                logger.error(f"Image not found: {img_path}")
                return DetectionResult(Error(non_img=True),
                                       is_normal=False, confidence=0.0, bbox=None)
            img = cv2.imread(img_path)
            return await self.detect(img)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return DetectionResult(Error(error=True, non_detect=True), is_normal=False, confidence=0.0, bbox=None)

    async def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise ModelError("Model not loaded")

        try:
            # Run inference
            is_normal, predictions = self.model.one_step_inference(image)

            # Save image with predictions
            await save_predictions_image(image, predictions)

            if len(predictions) == 0:
                return DetectionResult(
                    error=Error(non_detect=True),
                    is_normal=False,
                    confidence=0.0,
                    bbox=None
                )
            else:
                return DetectionResult(
                    error=Error(),
                    is_normal=is_normal,  # 1 for normal, 0 for defect
                    confidence=float(predictions[0][5]),
                    bbox=tuple(predictions[0][:4])
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
                     f"Classes: {self.config.classes}"
        return str_format
