import torch
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_path: str
    img_path: str
    confidence_threshold: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 640
    classes: Tuple[str] = ("defective", "non-defective")


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
