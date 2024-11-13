import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import cv2
from src.model.pear_detector import PearDetector, ModelConfig, DetectionResult


class TestPearDetector:
    @pytest.fixture
    def test_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def detector(self):
        config = ModelConfig(
            model_path="../weights/best.pt",
            img_path="img/test.jpg",
            confidence_threshold=0.5
        )
        detector = PearDetector(config)
        detector.load_model()
        return detector

    def test_model_load(self, detector):
        assert detector.model is not None
        assert detector.config.confidence_threshold == 0.5

    @pytest.mark.asyncio
    async def test_inference(self, detector, test_image):
        result = await detector.detect(test_image)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_defective, bool)
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference_async(self, detector):
        img_path = "../img/test.jpg"
        result = await detector.inference(img_path)  # Added await here
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_defective, bool)
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_empty_image(self, detector):
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await detector.detect(empty_image)
        assert isinstance(result, DetectionResult)
        assert not result.is_defective
        assert result.confidence == 0.0
