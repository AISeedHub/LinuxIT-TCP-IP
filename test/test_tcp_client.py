import os
import sys
import json
import asyncio
import pytest
from typing import Dict, Any, Optional, List

import pytest_asyncio

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server.tcp_server import ServerConfig
from src.model.detector import ModelConfig, PearDetector
from configs import *


class TCPClient:
    """
    TCP Client for testing the TCP Server
    
    This client can connect to the server, send requests, and receive responses.
    """

    def __init__(self, host: str, port: int, encoding: str = 'utf-8'):
        self.host = host
        self.port = port
        self.encoding = encoding
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Connect to the server"""
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port
        )
        print(f"Connected to {self.host}:{self.port}")

    async def disconnect(self) -> None:
        """Disconnect from the server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            print(f"Disconnected from {self.host}:{self.port}")

    async def send_request(self, cmd: int, request_data: List[Any]) -> Dict[str, Any]:
        """
        Send a request to the server and receive the response
        
        Args:
            cmd: Command code (e.g., 0x01 for classification)
            request_data: List of data to send with the request
            
        Returns:
            Response from the server as a dictionary
        """
        if not self.writer or not self.reader:
            raise ConnectionError("Not connected to server")

        # Create request
        request = {
            "cmd": cmd,
            "request_data": request_data
        }

        # Encode and send request
        encoded_request = json.dumps(request).encode(self.encoding)
        self.writer.write(encoded_request)
        await self.writer.drain()
        print(f"Sent request: {request}")

        # Receive and decode response
        response_data = await self.reader.read(4096)  # Adjust buffer size as needed
        response = json.loads(response_data.decode(self.encoding))
        print(f"Received response: {response}")

        return response


@pytest.fixture
async def server_config():
    """Fixture for server configuration"""
    return ServerConfig(
        host="127.0.0.1",
        port=8888,
        max_connections=5,
        buffer_size=4096,
        encoding='utf-8'
    )


@pytest.fixture
async def model_config():
    """Fixture for model configuration"""
    return ModelConfig(
        model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
        img_path=IMG_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


@pytest.fixture
async def detector(model_config):
    """Fixture for detector"""
    detector = PearDetector(model_config)
    detector.load_model()
    yield detector
    detector.unload_model()


@pytest_asyncio.fixture
async def tcp_client():
    """Create TCP client fixture"""
    # Tạo và khởi tạo client
    client = TCPClient("127.0.0.1", 9090)  # Thay bằng tên class thực tế của bạn

    # Kết nối nếu cần thiết
    await client.connect()  # nếu có method connect

    yield client  # Sử dụng yield để trả về client

    # Cleanup sau khi test xong
    await client.disconnect()  # nếu có method disconnect


@pytest.mark.asyncio
async def test_classification_request(tcp_client):
    """Test classification request"""
    # Send classification request with test image
    response = await tcp_client.send_request(
        cmd=0x01,  # request_classification
        request_data=[os.path.join(IMG_PATH, LIST_IMAGES_TEST[0])]
    )

    # Verify response structure
    assert "cmd" in response
    assert response["cmd"] == 0x02  # response_classification
    assert "response_data" in response
    assert isinstance(response["response_data"], list)
    assert len(response["response_data"]) > 0

    # Verify response data
    result = response["response_data"][0]
    assert "file_name" in result
    assert LIST_IMAGES_TEST[0] in result['file_name']
    assert "result" in result
    assert "error_code" in result
    assert result["error_code"] is 0  # No error
    # result should be 0 or 1 for classification
    assert result["result"] in [0, 1]  # Assuming binary classification


@pytest.mark.asyncio
async def test_model_change_request(tcp_client):
    """Test model change request"""
    # Send model change request
    response = await tcp_client.send_request(
        cmd=0x26,  # model_change_request
        request_data=[r"pear.pt"]
    )

    # Verify response structure
    assert "cmd" in response
    assert response["cmd"] == 0x27  # model_change_response
    assert "response_data" in response
    assert isinstance(response["response_data"], list)
    assert len(response["response_data"]) > 0

    # Verify response data
    result = response["response_data"][0]
    assert "file_name" in result
    assert "result" in result
    assert result["result"] is 2  # 2 is success for model change, 1 is failure


@pytest.mark.asyncio
async def test_current_model_request(tcp_client):
    """Test current model request"""
    # Send current model request
    response = await tcp_client.send_request(
        cmd=0x22,  # request_current_model
        request_data=[]
    )

    # Verify response structure
    assert "cmd" in response
    assert response["cmd"] == 0x23  # response_current_model
    assert "response_data" in response
    assert isinstance(response["response_data"], list)
    assert len(response["response_data"]) > 0

    # Verify response data
    result = response["response_data"][0]
    assert "file_name" in result
    assert "result" in result
    assert "error_code" in result
    # success get current model: result should be 2 and error_code should be 0
    assert result["result"] is 2
    assert result["error_code"] is 0  # No error


@pytest.mark.asyncio
async def test_list_models_request(tcp_client):
    """Test list models request"""
    # Send list models request
    response = await tcp_client.send_request(
        cmd=0x24,  # request_list_model
        request_data=[]
    )

    # Verify response structure
    assert "cmd" in response
    assert response["cmd"] == 0x25  # response_list_model
    assert "response_data" in response
    assert isinstance(response["response_data"], list)
    assert len(response["response_data"]) > 0

    # Verify response data
    for model in response["response_data"]:
        assert "file_name" in model
        assert "result" in model
        assert "error_code" in model
        assert model["error_code"] is 0  # No error


@pytest.mark.asyncio
async def test_invalid_request(tcp_client):
    """Test invalid request"""
    # Send invalid request (non-existent command)
    response = await tcp_client.send_request(
        cmd=0xFF,  # Invalid command
        request_data=[]
    )

    # Verify error response
    assert "cmd" in response
    assert response["cmd"] is None  # response_error
    assert "response_data" in response
    assert isinstance(response["response_data"], list)
    assert len(response["response_data"]) > 0

    # Verify error data
    error = response["response_data"][0]
    assert "error_code" in error
    assert error["error_code"] is not None  # Error code present
    assert "message" in error


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
