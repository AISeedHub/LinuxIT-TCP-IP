import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from ..utils.exceptions import ModelError, InferenceError

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str
    confidence_threshold: float
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size: int = 640


@dataclass
class DetectionResult:
    is_defective: bool
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]


class PearDetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device(config.device)

    async def load_model(self, model_path: Optional[str] = None) -> None:
        try:
            path = model_path or self.config.model_path
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Model loading failed: {e}")

    async def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise ModelError("Model not loaded")

        try:
            # Preprocess image
            img = self._preprocess_image(image)

            # Run inference
            with torch.no_grad():
                results = self.model(img)

            # Process results
            predictions = results.pred[0].cpu().numpy()
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

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        # Convert BGR to RGB
        image = image[:, :, ::-1]

        # Resize
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))

        # Normalize and convert to tensor
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0).to(self.device)

        return image

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded")