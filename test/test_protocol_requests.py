"""
TCP Client Implementation for Testing Server Connections

This module provides a comprehensive TCP client for testing the LinuxIT TCP/IP server.
It supports all protocol commands and provides both automated and manual testing capabilities.
"""

import os
import sys
import asyncio
import json
import logging
import socket
import time
import pytest
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server.tcp_server import TCPServer, ServerConfig
from src.model.detector import PearDetector, ModelConfig
from configs import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Represents the result of a test case"""
    test_name: str
    success: bool
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]
    error: Optional[str]
    response_time: float


class TCPTestClient:
    """
    TCP client for testing the LinuxIT server protocol.
    Supports connection testing, protocol validation, and performance measurement.
    """

    def __init__(self, host: str = "localhost", port: int = 9090, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False

    async def connect(self) -> bool:
        """
        Establish connection to the server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
            self.connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Close connection to server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.connected = False
            logger.info("Disconnected from server")

    async def send_request(self, request: Dict[str, Any]) -> TestResult:
        """
        Send a request and measure response time

        Args:
            request: Request dictionary with cmd and request_data

        Returns:
            TestResult with response data and timing information
        """
        test_name = f"cmd_{hex(request.get('cmd', 0))}"
        start_time = time.time()

        try:
            if not self.connected:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    request=request,
                    response=None,
                    error="Not connected to server",
                    response_time=0.0
                )

            # Convert request to JSON and send
            request_json = json.dumps(request)
            request_bytes = request_json.encode('utf-8')

            self.writer.write(request_bytes)
            await self.writer.drain()

            # Read response
            response_data = await asyncio.wait_for(
                self.reader.read(4096),
                timeout=self.timeout
            )

            response_time = time.time() - start_time

            if not response_data:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    request=request,
                    response=None,
                    error="No response received",
                    response_time=response_time
                )

            # Parse response
            response_str = response_data.decode('utf-8')
            response = json.loads(response_str)

            return TestResult(
                test_name=test_name,
                success=True,
                request=request,
                response=response,
                error=None,
                response_time=response_time
            )

        except asyncio.TimeoutError:
            return TestResult(
                test_name=test_name,
                success=False,
                request=request,
                response=None,
                error="Request timeout",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                request=request,
                response=None,
                error=str(e),
                response_time=time.time() - start_time
            )

    def get_test_requests(self) -> List[Dict[str, Any]]:
        """
        Get a comprehensive list of test requests covering all protocol commands

        Returns:
            List of test request dictionaries
        """
        return [
            # Classification requests
            {"cmd": 0x01, "request_data": [LIST_IMAGES_TEST[0]]},
            {"cmd": 0x01, "request_data": LIST_IMAGES_TEST},

            # Stop classification
            {"cmd": 0x03, "request_data": None},

            # # Model download request
            # {"cmd": 0x20, "request_data": ["https://github.com/user/model.pt"]},

            # Current model request
            {"cmd": 0x22, "request_data": None},

            # List all models request
            {"cmd": 0x24, "request_data": None},

            # Change model request
            {"cmd": 0x26, "request_data": ["new_model.pt"]},

            # Delete model request
            {"cmd": 0x28, "request_data": ["old_model.pt"]},

            # Change image folder request
            {"cmd": 0x30, "request_data": ["new_img_folder"]},

            # Current image folder request
            {"cmd": 0x32, "request_data": None},

            # Change model folder request
            {"cmd": 0x34, "request_data": ["new_model_folder"]},
        ]

    def get_error_test_requests(self) -> List[Dict[str, Any]]:
        """
        Get test requests that should trigger errors for testing error handling

        Returns:
            List of test request dictionaries that should cause errors
        """
        return [
            # Invalid command code
            {"cmd": 0x99, "request_data": None},

            # Invalid file extensions
            {"cmd": 0x01, "request_data": ["document.txt", "archive.zip"]},

            # Invalid URL
            {"cmd": 0x20, "request_data": ["not-a-valid-url"]},

            # Multiple URLs for single download
            {"cmd": 0x20, "request_data": ["url1.pt", "url2.pt"]},

            # Invalid model extension
            {"cmd": 0x26, "request_data": ["model.txt"]},

            # Dangerous directory path
            {"cmd": 0x30, "request_data": ["../../../etc"]},

            # Empty file list
            {"cmd": 0x01, "request_data": []},
        ]

    async def run_comprehensive_tests(self) -> List[TestResult]:
        """
        Run a comprehensive set of protocol tests

        Returns:
            List of TestResult objects
        """
        results = []

        # Test valid requests
        logger.info("Testing valid protocol requests...")
        for request in self.get_test_requests():
            result = await self.send_request(request)
            results.append(result)

            if result.success:
                logger.info(f"✓ {result.test_name}: {result.response_time:.3f}s")
            else:
                logger.error(f"✗ {result.test_name}: {result.error}")

        # Test error cases
        logger.info("Testing error handling...")
        for request in self.get_error_test_requests():
            result = await self.send_request(request)
            results.append(result)

            # For error tests, we expect either an error or an error response
            if result.success and result.response and "error_code" in str(result.response):
                logger.info(f"✓ {result.test_name}: Error handled correctly")
            elif not result.success:
                logger.info(f"✓ {result.test_name}: Connection error as expected")
            else:
                logger.warning(f"? {result.test_name}: Unexpected success")

        return results

    async def test_connection_stability(self, num_requests: int = 100) -> List[TestResult]:
        """
        Test connection stability with multiple sequential requests

        Args:
            num_requests: Number of requests to send

        Returns:
            List of TestResult objects
        """
        logger.info(f"Testing connection stability with {num_requests} requests...")
        results = []
        test_request = {"cmd": 0x22, "request_data": None}  # Simple current model request

        for i in range(num_requests):
            result = await self.send_request(test_request)
            result.test_name = f"stability_test_{i + 1}"
            results.append(result)

            if not result.success:
                logger.error(f"Connection failed at request {i + 1}: {result.error}")
                break

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_requests} requests")

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Connection stability: {success_count}/{len(results)} successful")

        return results

    async def test_concurrent_connections(self, num_concurrent: int = 5) -> List[TestResult]:
        """
        Test multiple concurrent connections to the server

        Args:
            num_concurrent: Number of concurrent connections

        Returns:
            List of TestResult objects from all connections
        """
        logger.info(f"Testing {num_concurrent} concurrent connections...")

        async def single_connection_test(connection_id: int) -> List[TestResult]:
            """Test function for a single concurrent connection"""
            client = TCPTestClient(self.host, self.port, self.timeout)
            results = []

            try:
                if await client.connect():
                    # Send a few test requests
                    test_requests = [
                        {"cmd": 0x22, "request_data": None},
                        {"cmd": 0x24, "request_data": None},
                        {"cmd": 0x32, "request_data": None},
                    ]

                    for i, request in enumerate(test_requests):
                        result = await client.send_request(request)
                        result.test_name = f"concurrent_{connection_id}_{i}"
                        results.append(result)

                        # Small delay between requests
                        await asyncio.sleep(0.1)
                else:
                    # Connection failed
                    results.append(TestResult(
                        test_name=f"concurrent_{connection_id}_connect",
                        success=False,
                        request={},
                        response=None,
                        error="Failed to connect",
                        response_time=0.0
                    ))
            finally:
                await client.disconnect()

            return results

        # Run concurrent tests
        tasks = [single_connection_test(i) for i in range(num_concurrent)]
        all_results = await asyncio.gather(*tasks)

        # Flatten results
        results = []
        for connection_results in all_results:
            results.extend(connection_results)

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Concurrent test results: {success_count}/{len(results)} successful")

        return results

    def print_test_summary(self, results: List[TestResult]) -> None:
        """
        Print a summary of test results

        Args:
            results: List of TestResult objects
        """
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests

        if successful_tests > 0:
            avg_response_time = sum(r.response_time for r in results if r.success) / successful_tests
        else:
            avg_response_time = 0.0

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests:      {total_tests}")
        print(f"Successful:       {successful_tests}")
        print(f"Failed:           {failed_tests}")
        print(f"Success Rate:     {(successful_tests / total_tests) * 100:.1f}%")
        print(f"Avg Response:     {avg_response_time:.3f}s")
        print("=" * 60)

        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in results:
                if not result.success:
                    print(f"  {result.test_name}: {result.error}")


# This function is replaced by pytest test functions


# Pytest fixtures
@pytest.fixture
def server_config():
    """Fixture for server configuration"""
    return ServerConfig(
        host="127.0.0.1",
        port=8888,
        max_connections=5,
        buffer_size=4096,
        encoding='utf-8'
    )


@pytest.fixture
def model_config():
    """Fixture for model configuration"""
    return ModelConfig(
        model_path=os.path.join(MODEL_PATH, MODEL_DEFAULT),
        img_path=os.path.join(IMG_PATH, LIST_IMAGES_TEST[0]),
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


@pytest.fixture
async def detector(model_config):
    """Fixture for detector"""
    detector = PearDetector(model_config)
    detector.load_model()
    yield detector
    detector.unload_model()


@pytest.fixture
async def tcp_server(server_config, detector):
    """Fixture for TCP server"""
    server = TCPServer(server_config, detector)
    server_task = asyncio.create_task(server.start())
    # Give the server a moment to start
    await asyncio.sleep(1)
    yield server
    # Clean up
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.fixture
async def tcp_test_client():
    """Fixture for TCP test client"""
    client = TCPTestClient(host="127.0.0.1", port=8888)
    await client.connect()
    yield client
    await client.disconnect()


# Test functions
@pytest.mark.asyncio
async def test_valid_protocol_requests(tcp_server, tcp_test_client):
    """Test valid protocol requests"""
    for request in tcp_test_client.get_test_requests():
        result = await tcp_test_client.send_request(request)

        # Check if the request was successful
        assert result.success, f"Request {result.test_name} failed: {result.error}"

        # Check if we got a response
        assert result.response is not None, f"No response received for {result.test_name}"

        # Check if the response has the expected structure
        assert "cmd" in result.response, f"Response for {result.test_name} missing 'cmd' field"
        assert "response_data" in result.response, f"Response for {result.test_name} missing 'response_data' field"


@pytest.mark.asyncio
async def test_error_handling(tcp_server, tcp_test_client):
    """Test error handling for invalid requests"""
    for request in tcp_test_client.get_error_test_requests():
        result = await tcp_test_client.send_request(request)

        # For error tests, we expect either an error or an error response
        if result.success and result.response:
            # If successful, the response should contain an error code
            response_str = str(result.response)
            assert "error_code" in response_str, f"Error response for {result.test_name} missing 'error_code'"
        else:
            # If not successful, there should be an error message
            assert result.error is not None, f"Failed request {result.test_name} missing error message"


@pytest.mark.asyncio
async def test_connection_stability(tcp_server, tcp_test_client):
    """Test connection stability with multiple sequential requests"""
    num_requests = 10  # Reduced for faster testing
    test_request = {"cmd": 0x22, "request_data": None}  # Simple current model request

    for i in range(num_requests):
        result = await tcp_test_client.send_request(test_request)
        assert result.success, f"Connection failed at request {i + 1}: {result.error}"


@pytest.mark.asyncio
async def test_concurrent_connections(tcp_server):
    """Test multiple concurrent connections to the server"""
    num_concurrent = 3

    async def single_connection_test(connection_id: int) -> List[TestResult]:
        """Test function for a single concurrent connection"""
        client = TCPTestClient(host="127.0.0.1", port=8888)
        results = []

        try:
            if await client.connect():
                # Send a few test requests
                test_requests = [
                    {"cmd": 0x22, "request_data": None},
                    {"cmd": 0x24, "request_data": None},
                    {"cmd": 0x32, "request_data": None},
                ]

                for i, request in enumerate(test_requests):
                    result = await client.send_request(request)
                    results.append(result)

                    # Check if the request was successful
                    assert result.success, f"Request failed for connection {connection_id}: {result.error}"

                    # Small delay between requests
                    await asyncio.sleep(0.1)
            else:
                # Connection failed
                assert False, f"Failed to connect for connection {connection_id}"
        finally:
            await client.disconnect()

        return results

    # Run concurrent tests
    tasks = [single_connection_test(i) for i in range(num_concurrent)]
    all_results = await asyncio.gather(*tasks)

    # Check results
    for connection_results in all_results:
        for result in connection_results:
            assert result.success, f"Request {result.test_name} failed: {result.error}"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
