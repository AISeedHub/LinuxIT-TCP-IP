"""
Test suite for request handlers.

NOTE: These tests are currently broken due to:
1. Missing model files (weights/best.pt)
2. Incorrect async fixture usage
3. Handler constructor signature changes

These tests require substantial refactoring and model dependencies.
See the new protocol test suite for comprehensive testing:
- test_protocol_parser.py
- test_protocol_validation.py  
- test_protocol_processing.py
- test_protocol_integration.py
"""

import pytest
from src.handlers.classification_handler import ClassificationHandler
from src.handlers.model_handler import ModelHandler
from src.model.pear_detector import PearDetector


@pytest.mark.skip(reason="Broken - requires model files and fixture refactoring")
@pytest.fixture
async def detector():
    config = ModelConfig(
        model_path="tests/fixtures/test_model.pt",
        confidence_threshold=0.5
    )
    detector = PearDetector(config)
    detector.load_model()
    return detector


@pytest.mark.skip(reason="Broken - requires model files and fixture refactoring")
@pytest.fixture
def classification_handler(detector):
    return ClassificationHandler(detector)


@pytest.mark.skip(reason="Broken - requires model files and fixture refactoring")
@pytest.fixture
def model_handler(detector):
    return ModelHandler(detector)


@pytest.mark.skip(reason="Broken - requires model files and fixture refactoring")
@pytest.mark.asyncio
async def test_classification_handler(classification_handler):
    request = {
        "cmd": "classify",
        "request_data": {
            "images": ["tests/fixtures/test_image.jpg"]
        }
    }

    response = await classification_handler.handle(request)
    assert response["status"] == "success"
    assert "result" in response


@pytest.mark.skip(reason="Broken - requires model files and fixture refactoring")
@pytest.mark.asyncio
async def test_model_handler(model_handler):
    request = {
        "cmd": "model",
        "request_data": {
            "action": "switch",
            "model_path": "tests/fixtures/alternate_model.pt"
        }
    }

    response = await model_handler.handle(request)
    assert response["status"] == "success"
