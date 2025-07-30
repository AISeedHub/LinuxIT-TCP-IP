import os
import sys
import time
import pytest
import numpy as np
import cv2
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.detector import PearDetector, ModelConfig, DetectionResult
from src.utils.exceptions import ModelError

# CONSTANTS
CONFIDENCE_THRESHOLD = 0.9  # Default confidence threshold for model inference
CLASSES = ("normal", "defective")  # Classes for the model
MODEL_PATH = r"C:\Users\Andrew\Desktop\LinuxIT-TCP-IP\weights"  # Path to the model weights
MODEL_DEFAULT = "best-2cls.pt"  # Default model file name
IMG_PATH = r"C:\Users\Andrew\Desktop\LinuxIT-TCP-IP\test_photos"  # Path to a sample image for testing
LIST_IMAGES_TEST = ["test1.jpg", "test2.jpg", "test3.jpg"]  # List of test images


class TestPearDetector:
    @pytest.fixture
    def test_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def detector(self):
        config = ModelConfig(
            model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
            classes=CLASSES,
            img_path=IMG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        detector = PearDetector(config)
        detector.load_model()
        return detector

    @pytest.fixture
    def available_models(self) -> List[str]:
        """Get a list of available model files"""
        models_dir = os.path.join(MODEL_PATH)
        return [f for f in os.listdir(models_dir) if f.endswith('.pt')]

    @pytest.fixture
    def test_image_paths(self):
        from pathlib import Path
        """Create list of test image paths"""
        base_path = Path(IMG_PATH)
        return [base_path / img for img in LIST_IMAGES_TEST if (base_path / img).exists()]

    def test_model_load(self, detector):
        assert detector.model is not None
        assert detector.config.confidence_threshold == CONFIDENCE_THRESHOLD

    def test_model_unload(self, detector):
        # Verify model is loaded
        assert detector.model is not None

        # Unload model
        detector.unload_model()

        # Verify model is unloaded
        assert detector.model is None

    @pytest.mark.asyncio
    async def test_model_change(self, detector, available_models):
        # Skip test if no alternative models available
        if len(available_models) < 2:
            pytest.skip("Not enough models available for testing model change")

        # Get current model path
        original_model_path = detector.config.model_path

        # Find an alternative model
        alternative_model = None
        for model in available_models:
            full_path = os.path.join("../weights", model)
            if full_path != original_model_path:
                alternative_model = full_path
                break

        if not alternative_model:
            pytest.skip("No alternative model found")

        # Change model
        detector.change_model(alternative_model)

        # Verify model was changed
        assert detector.config.model_path == alternative_model
        assert detector.model is not None

        # Change back to original model
        detector.change_model(original_model_path)
        assert detector.config.model_path == original_model_path

    @pytest.mark.asyncio
    async def test_model_change_invalid(self, detector):
        # Try to change to non-existent model
        with pytest.raises(ModelError):
            detector.change_model("non_existent_model.pt")

        # Verify original model is still loaded
        assert detector.model is not None

    @pytest.mark.asyncio
    async def test_inference_performance(self, detector, test_image):
        # Perform multiple inferences and measure time
        num_iterations = 5
        total_time = 0

        for _ in range(num_iterations):
            start_time = time.time()
            result = await detector.detect(test_image)
            end_time = time.time()

            # Verify result is valid
            assert isinstance(result, DetectionResult)

            # Add to total time
            total_time += (end_time - start_time)

        # Calculate average time
        avg_time = total_time / num_iterations
        print(f"Average inference time: {avg_time:.4f} seconds")

        # No strict assertion on performance, but log it for monitoring
        assert avg_time > 0  # Just ensure time is measured

    @pytest.mark.asyncio
    async def test_inference(self, detector, test_image):
        result = await detector.detect(test_image)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, int)
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference_async(self, detector):
        img_path = os.path.join(IMG_PATH, LIST_IMAGES_TEST[0])
        result = await detector.inference(img_path)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, int)
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_empty_image(self, detector):
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await detector.detect(empty_image)
        assert isinstance(result, DetectionResult)
        assert result.is_normal
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_model_reload(self, detector):
        # Unload model
        detector.unload_model()
        assert detector.model is None

        # Reload model
        detector.load_model()
        assert detector.model is not None

        # Test inference after reload
        img_path = os.path.join(IMG_PATH, LIST_IMAGES_TEST[0])
        result = await detector.inference(img_path)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, int)
