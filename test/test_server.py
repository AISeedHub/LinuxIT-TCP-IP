import pytest
import asyncio
from src.server.tcp_server import TCPServer
from src.model.detector import PearDetector
from src.utils.config import ServerConfig, ModelConfig


@pytest.fixture
async def server_config():
    return ServerConfig(
        host="localhost",
        port=8080,
        max_connections=10,
        buffer_size=1024
    )


@pytest.fixture
async def model_config():
    return ModelConfig(
        model_path="tests/fixtures/test_model.pt",
        confidence_threshold=0.5
    )


@pytest.fixture
def detector(model_config):
    detector = PearDetector(model_config)
    detector.load_model()
    return detector


@pytest.fixture
async def server(server_config, detector):
    server = TCPServer(server_config, detector)
    yield server
    # Cleanup
    await server._server.close()
    await server._server.wait_closed()


@pytest.mark.asyncio
async def test_server_start(server):
    # Start server in background task
    task = asyncio.create_task(server.start())
    await asyncio.sleep(0.1)  # Give server time to start

    assert server._server is not None
    assert server._server.is_serving()

    # Cleanup
    server._server.close()
    await server._server.wait_closed()
    await task

