import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import cv2
from src.model.detector import PearDetector, ModelConfig, DetectionResult


class TestPearDetector:
    @pytest.fixture
    def test_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def detector(self):
        config = ModelConfig(
            model_path="../weights/best-2cls.pt",
            img_path="./test_photos",
            classes=('defect', 'non-defective'),
            confidence_threshold=0.8
        )
        detector = PearDetector(config)
        detector.load_model()
        return detector

    def test_model_load(self, detector):
        assert detector.model is not None
        assert detector.config.confidence_threshold == 0.8

    @pytest.mark.asyncio
    async def test_detection(self, detector, test_image):
        result = await detector.detect(test_image)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, int)
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference_async(self, detector):
        img_path = "test1.jpg"
        result = await detector.inference(img_path)  # Added await here
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, int)
        assert result.error.non_img is False
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("img_path, is_normal", [
        ("test1.jpg", 1),
        ("test2.jpg", 0),
        ("test3.jpg", 0),
        ("test4.jpg", 0),
        ("test5.jpg", 0),
    ])
    @pytest.mark.asyncio
    async def test_multiple_images(self, detector, img_path, is_normal):
       
        result = await detector.inference(img_path)
        assert isinstance(result, DetectionResult)
        assert result.is_normal == is_normal


    @pytest.mark.asyncio
    async def test_empty_image(self, detector):
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await detector.detect(empty_image)
        assert isinstance(result, DetectionResult)
        assert result.is_normal
        assert result.error.non_detect is True
        assert result.confidence == 0.0
