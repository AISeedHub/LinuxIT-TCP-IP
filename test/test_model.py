import os
import sys
import pytest
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
            model_path="../weights/pear.pt",
            classes=["pear", "apple"],
            confidence_threshold=0.8
        )
        detector = PearDetector(config)
    
        return detector

    def test_model_load(self, detector):
        assert detector.model is not None
        assert detector.config.confidence_threshold == 0.8

    
    @pytest.mark.asyncio
    async def test_inference_async(self, detector):
        img_path = "./img/test5.jpg"
        result = await detector.inference(img_path)  # Added await here
        print(result)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, bool)
        assert result.is_normal
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference(self, detector, test_image):
        result = await detector.detect(test_image)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, bool)
        # assert not result.is_normal
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_empty_image(self, detector):
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await detector.detect(empty_image)
        assert isinstance(result, DetectionResult)
        assert result.error.non_detect == True