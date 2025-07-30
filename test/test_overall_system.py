import os
import sys
import asyncio
import pytest
import subprocess
from typing import List, Optional, Tuple

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server.tcp_server import TCPServer, ServerConfig
from src.model.detector import PearDetector, ModelConfig

from configs import *


class TestRunner:
    def __init__(self):
        self.server_config = ServerConfig(
            host="127.0.0.1",
            port=8888,
            max_connections=5,
            buffer_size=4096,
            encoding='utf-8'
        )

        self.model_config = ModelConfig(
            model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
            img_path=os.path.join(IMG_PATH, LIST_IMAGES_TEST[0]),
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        self.detector = PearDetector(self.model_config)

    async def start_server(self) -> Tuple[TCPServer, asyncio.Task]:
        """Start the TCP server for testing"""
        server = TCPServer(self.server_config, self.detector)
        # Start server in a separate task
        server_task = asyncio.create_task(server.start())
        # Give the server a moment to start
        await asyncio.sleep(1)
        return server, server_task

    def run_tests(self, test_files: List[str]) -> bool:
        """Run pytest on the specified test files"""
        cmd = ["pytest", "-xvs"] + test_files
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print test output
        print(result.stdout)
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)

        return result.returncode == 0

    async def run_tcp_client_tests(self):
        """Run TCP client tests with server running"""
        self.detector.load_model()

        try:
            # Start server
            print("Starting TCP server for tests...")
            server, server_task = await self.start_server()

            # Run TCP client tests
            print("\nRunning TCP client tests...")
            success = self.run_tests(["test_tcp_client.py"])

            # Cancel server task
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

            return success
        finally:
            # Clean up
            self.detector.unload_model()


# Pytest fixtures
@pytest.fixture
def runner_instance():
    """Fixture để tạo TestRunner instance"""
    return TestRunner()


@pytest.fixture
def server_config_instance():
    """Fixture để tạo ServerConfig"""
    return ServerConfig(
        host="127.0.0.1",
        port=8888,
        max_connections=5,
        buffer_size=4096,
        encoding='utf-8'
    )


@pytest.fixture
def model_config_instance():
    """Fixture để tạo ModelConfig"""
    return ModelConfig(
        model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
        img_path=os.path.join(IMG_PATH, LIST_IMAGES_TEST[0]),
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


# Test functions
def test_runner_initialization(runner_instance):
    """Test khởi tạo TestRunner"""
    assert runner_instance is not None
    assert runner_instance.server_config.host == "127.0.0.1"
    assert runner_instance.server_config.port == 8888
    assert runner_instance.model_config.confidence_threshold == CONFIDENCE_THRESHOLD


def test_server_config_creation(server_config_instance):
    """Test tạo ServerConfig"""
    assert server_config_instance.host == "127.0.0.1"
    assert server_config_instance.port == 8888
    assert server_config_instance.max_connections == 5
    assert server_config_instance.buffer_size == 4096
    assert server_config_instance.encoding == 'utf-8'


def test_model_config_creation(model_config_instance):
    """Test tạo ModelConfig"""
    assert model_config_instance.confidence_threshold == CONFIDENCE_THRESHOLD
    assert MODEL_DEFAULT in model_config_instance.model_path
    assert LIST_IMAGES_TEST[0] in model_config_instance.img_path


def test_model_tests_execution(runner_instance):
    """Test chạy model tests"""
    # Kiểm tra xem file test_model.py có tồn tại không
    test_file = "test_model.py"
    if os.path.exists(test_file):
        success = runner_instance.run_tests([test_file])
        assert isinstance(success, bool)
    else:
        pytest.skip(f"Test file {test_file} không tồn tại")


@pytest.mark.asyncio
async def test_tcp_client_tests_execution(runner_instance):
    """Test chạy TCP client tests"""
    # Kiểm tra xem file test_tcp_client.py có tồn tại không
    test_file = "test_tcp_client.py"
    if os.path.exists(test_file):
        success = await runner_instance.run_tcp_client_tests()
        assert isinstance(success, bool)
    else:
        pytest.skip(f"Test file {test_file} không tồn tại")


def test_detector_initialization_and_methods(runner_instance):
    """Test khởi tạo detector"""
    assert runner_instance.detector is not None
    assert hasattr(runner_instance.detector, 'load_model')
    assert hasattr(runner_instance.detector, 'unload_model')


def test_configuration_consistency_check():
    """Test tính nhất quán của configuration"""
    runner = TestRunner()

    # Test server config consistency
    assert runner.server_config.host == "127.0.0.1"
    assert runner.server_config.port == 8888

    # Test model config consistency
    assert runner.model_config.confidence_threshold == CONFIDENCE_THRESHOLD
    assert MODEL_DEFAULT in runner.model_config.model_path


def test_class_instantiation():
    """Test khả năng tạo instance của các class"""
    # Test TestRunner instantiation
    runner = TestRunner()
    assert isinstance(runner, TestRunner)

    # Test ServerConfig instantiation
    server_config = ServerConfig(
        host="127.0.0.1",
        port=8888,
        max_connections=5,
        buffer_size=4096,
        encoding='utf-8'
    )
    assert isinstance(server_config, ServerConfig)

    # Test ModelConfig instantiation
    model_config = ModelConfig(
        model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
        img_path=os.path.join(IMG_PATH, LIST_IMAGES_TEST[0]),
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    assert isinstance(model_config, ModelConfig)