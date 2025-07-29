import pytest
from src.handlers.classification_handler import ClassificationHandler
from src.handlers.model_handler import ModelHandler
from src.model.detector import PearDetector


@pytest.fixture
async def detector():
    config = ModelConfig(
        model_path="tests/fixtures/test_model.pt",
        confidence_threshold=0.5
    )
    detector = PearDetector(config)
    detector.load_model()
    return detector


@pytest.fixture
def classification_handler(detector):
    return ClassificationHandler(detector)


@pytest.fixture
def model_handler(detector):
    return ModelHandler(detector)


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
