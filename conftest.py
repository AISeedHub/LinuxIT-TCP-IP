"""
Test configuration and fixtures for protocol tests.
Handles mocking of heavy ML dependencies for testing.
"""

import sys
import numpy as np
from unittest.mock import Mock, MagicMock
import pytest

# Mock heavy dependencies before they are imported
if 'torch' not in sys.modules:
    torch_mock = Mock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.device = Mock()
    torch_mock.cuda.empty_cache = Mock()
    torch_mock.no_grad = MagicMock()
    sys.modules['torch'] = torch_mock

if 'ultralytics' not in sys.modules:
    ultralytics_mock = Mock()
    sys.modules['ultralytics'] = ultralytics_mock

if 'cv2' not in sys.modules:
    cv2_mock = Mock()
    cv2_mock.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2_mock.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2_mock.COLOR_BGR2RGB = 4
    sys.modules['cv2'] = cv2_mock

@pytest.fixture(autouse=True)
def mock_save_predictions_image():
    """Mock the save_predictions_image function"""
    import sys
    if 'src.utils.visualization' not in sys.modules:
        viz_mock = Mock()
        viz_mock.save_predictions_image = Mock()
        sys.modules['src.utils.visualization'] = viz_mock
    return sys.modules['src.utils.visualization']

@pytest.fixture(autouse=True)
def mock_torch():
    """Auto-use fixture to ensure torch is mocked for all tests"""
    return sys.modules['torch']

@pytest.fixture(autouse=True) 
def mock_cv2():
    """Auto-use fixture to ensure cv2 is mocked for all tests"""
    return sys.modules['cv2']

@pytest.fixture(autouse=True)
def mock_ultralytics():
    """Auto-use fixture to ensure ultralytics is mocked for all tests"""
    return sys.modules['ultralytics']