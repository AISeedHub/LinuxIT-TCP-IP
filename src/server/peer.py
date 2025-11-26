<<<<<<< HEAD
import asyncio
import json
import logging
from typing import Optional, Dict, Any

from ..handlers import get_handler, COMMAND_CODES
from ..utils.parsers import convert_str_to_dict
from ..utils.validators import RequestValidator
from ..utils.exceptions import ValidationError
from ..model.detector import PearDetector

logger = logging.getLogger(__name__)


class RequestProcessor:
    """Helper class to process and validate requests"""

    def __init__(self):
        self.validator = RequestValidator()

    def process_request(self, raw_data: bytes, encoding: str) -> Dict[str, Any]:
        """
        Process and validate raw request data

        Args:
            raw_data: Raw bytes from client
            encoding: Encoding to use for decoding

        Returns:
            Validated request dictionary

        Raises:
            ValidationError: If request is invalid
        """
        try:
            # Decode bytes to string
            data_str = raw_data.decode(encoding)
            logger.debug(f"Received raw request: {data_str}")

            # Convert string with hex values to dict
            request = convert_str_to_dict(data_str)
            logger.debug(f"Parsed request: {request}")

            # Validate basic structure
            self.validator.validate_basic_structure(request)

            # Validate command code
            cmd = request.get('cmd')
            if cmd not in COMMAND_CODES:
                raise ValidationError(f"Invalid command code: {hex(cmd)}")

            # Validate request data based on command
            self._validate_request_data(cmd, request.get('request_data'))

            return request

        except UnicodeDecodeError as e:
            raise ValidationError(f"Failed to decode request: {e}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")

    def _validate_request_data(self, cmd: int, request_data: Any) -> None:
        """
        Validate request data based on command type

        Args:
            cmd: Command code
            request_data: Request data to validate

        Raises:
            ValidationError: If request data is invalid
        """
        # Classification request
        if cmd == 0x01:  # request_classification
            self.validator.validate_file_list(
                request_data,
                allowed_extensions=['.jpg', '.jpeg', '.png']
            )

        # Model download request
        elif cmd == 0x20:  # request_download
            if not request_data or len(request_data) != 1:
                raise ValidationError("Download request must contain exactly one URL")
            self.validator.validate_url(request_data[0])

        # Model change request
        elif cmd == 0x26:  # model_change_request
            if not request_data or len(request_data) != 1:
                raise ValidationError("Model change request must contain exactly one model name")
            self.validator.validate_file_list(
                request_data,
                allowed_extensions=['.pt']
            )

        # Directory change request
        elif cmd == 0x30:  # request_change_img_folder
            if not request_data or len(request_data) != 1:
                raise ValidationError("Directory change request must contain exactly one path")
            self.validator.validate_directory_path(request_data[0])


class Peer:
    def __init__(
            self,
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
            config: 'ServerConfig',
            detector: PearDetector
    ):
        self.reader = reader
        self.writer = writer
        self.config = config
        self.detector = detector
        self.addr = writer.get_extra_info('peername')
        self.request_processor = RequestProcessor()

    async def handle_connection(self) -> None:
        """Handle client connection and process requests"""
        logger.info(f"New connection from {self.addr}")

        try:
            while True:
                # Read data from client
                data = await self.reader.read(self.config.buffer_size)
                if not data:
                    break

                try:
                    # Process and validate request
                    request = self.request_processor.process_request(
                        data,
                        self.config.encoding
                    )

                    logger.info(f"Received request from {self.addr}: {request}")

                    # Get appropriate handler and process request
                    handler = get_handler(request['cmd'], self.detector)
                    response = await handler.handle(request)

                    # Send response back to client
                    await self._send_response(response)

                except ValidationError as e:
                    logger.warning(f"Validation error from {self.addr}: {e}")
                    await self._send_error(str(e), error_code=1)  # invalid_message

                except Exception as e:
                    logger.error(f"Error processing request from {self.addr}: {e}")
                    await self._send_error("Internal server error", error_code=3)  # not_exist_command

        except asyncio.CancelledError:
            logger.info(f"Connection cancelled for {self.addr}")
        except asyncio.IncompleteReadError:
            logger.warning(f"Connection closed unexpectedly by {self.addr}")
        except Exception as e:
            logger.error(f"Unexpected error for {self.addr}: {e}")
        finally:
            await self._cleanup()

    async def _send_response(self, response: Dict[str, Any]) -> None:
        """Send response to client"""
        try:
            encoded_response = json.dumps(response).encode(self.config.encoding)
            self.writer.write(encoded_response)
            await self.writer.drain()
            logger.info(f"Sent response to {self.addr}: {response}")

        except Exception as e:
            logger.error(f"Error sending response to {self.addr}: {e}")
            raise

    async def _send_error(self, message: str, error_code: int) -> None:
        """Send error response to client"""
        error_response = {
            "cmd": None,  # response_error
            "response_data": [{
                "file_name": None,
                "result": None,
                "error_code": error_code,
                "message": message
            }],
            "request_data": None
        }
        await self._send_response(error_response)

    async def _cleanup(self) -> None:
        """Clean up connection"""
        try:
            self.writer.close()
            await self.writer.wait_closed()
            logger.info(f"Connection closed for {self.addr}")
        except Exception as e:
=======
import asyncio
import json
import logging
from typing import Optional, Dict, Any

from ..handlers import get_handler, COMMAND_CODES
from ..utils.parsers import convert_str_to_dict
from ..utils.validators import RequestValidator
from ..utils.exceptions import ValidationError
from ..model.detector import PearDetector

logger = logging.getLogger(__name__)


class RequestProcessor:
    """Helper class to process and validate requests"""

    def __init__(self):
        self.validator = RequestValidator()

    def process_request(self, raw_data: bytes, encoding: str) -> Dict[str, Any]:
        """
        Process and validate raw request data

        Args:
            raw_data: Raw bytes from client
            encoding: Encoding to use for decoding

        Returns:
            Validated request dictionary

        Raises:
            ValidationError: If request is invalid
        """
        try:
            # Decode bytes to string
            data_str = raw_data.decode(encoding)
            logger.debug(f"Received raw request: {data_str}")

            # Convert string with hex values to dict
            request = convert_str_to_dict(data_str)
            logger.debug(f"Parsed request: {request}")

            # Validate basic structure
            self.validator.validate_basic_structure(request)

            # Validate command code
            cmd = request.get('cmd')
            if cmd not in COMMAND_CODES:
                raise ValidationError(f"Invalid command code: {hex(cmd)}")

            # Validate request data based on command
            self._validate_request_data(cmd, request.get('request_data'))

            return request

        except UnicodeDecodeError as e:
            raise ValidationError(f"Failed to decode request: {e}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")

    def _validate_request_data(self, cmd: int, request_data: Any) -> None:
        """
        Validate request data based on command type

        Args:
            cmd: Command code
            request_data: Request data to validate

        Raises:
            ValidationError: If request data is invalid
        """
        # Classification request
        if cmd == 0x01:  # request_classification
            self.validator.validate_file_list(
                request_data,
                allowed_extensions=['.jpg', '.jpeg', '.png']
            )

        # Model download request
        elif cmd == 0x20:  # request_download
            if not request_data or len(request_data) != 1:
                raise ValidationError("Download request must contain exactly one URL")
            self.validator.validate_url(request_data[0])

        # Model change request
        elif cmd == 0x26:  # model_change_request
            if not request_data or len(request_data) != 1:
                raise ValidationError("Model change request must contain exactly one model name")
            self.validator.validate_file_list(
                request_data,
                allowed_extensions=['.pt', '.pth']
            )

        # Directory change request
        elif cmd == 0x30:  # request_change_img_folder
            if not request_data or len(request_data) != 1:
                raise ValidationError("Directory change request must contain exactly one path")
            self.validator.validate_directory_path(request_data[0])


class Peer:
    def __init__(
            self,
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
            config: 'ServerConfig',
            detector: PearDetector
    ):
        self.reader = reader
        self.writer = writer
        self.config = config
        self.detector = detector
        self.addr = writer.get_extra_info('peername')
        self.request_processor = RequestProcessor()

    async def handle_connection(self) -> None:
        """Handle client connection and process requests"""
        logger.info(f"New connection from {self.addr}")

        try:
            while True:
                # Read data from client
                data = await self.reader.read(self.config.buffer_size)
                if not data:
                    break

                try:
                    # Process and validate request
                    request = self.request_processor.process_request(
                        data,
                        self.config.encoding
                    )

                    logger.info(f"Received request from {self.addr}: {request}")

                    # Get appropriate handler and process request
                    handler = get_handler(request['cmd'], self.detector)
                    response = await handler.handle(request)

                    # Send response back to client
                    await self._send_response(response)

                except ValidationError as e:
                    logger.warning(f"Validation error from {self.addr}: {e}")
                    await self._send_error(str(e), error_code=1)  # invalid_message

                except Exception as e:
                    logger.error(f"Error processing request from {self.addr}: {e}")
                    await self._send_error("Internal server error", error_code=3)  # not_exist_command

        except asyncio.CancelledError:
            logger.info(f"Connection cancelled for {self.addr}")
        except asyncio.IncompleteReadError:
            logger.warning(f"Connection closed unexpectedly by {self.addr}")
        except Exception as e:
            logger.error(f"Unexpected error for {self.addr}: {e}")
        finally:
            await self._cleanup()

    async def _send_response(self, response: Dict[str, Any]) -> None:
        """Send response to client"""
        try:
            encoded_response = json.dumps(response).encode(self.config.encoding)
            self.writer.write(encoded_response)
            await self.writer.drain()
            logger.info(f"Sent response to {self.addr}: {response}")

        except Exception as e:
            logger.error(f"Error sending response to {self.addr}: {e}")
            raise

    async def _send_error(self, message: str, error_code: int) -> None:
        """Send error response to client"""
        error_response = {
            "cmd": None,  # response_error
            "response_data": [{
                "file_name": None,
                "result": None,
                "error_code": error_code,
                "message": message
            }],
            "request_data": None
        }
        await self._send_response(error_response)

    async def _cleanup(self) -> None:
        """Clean up connection"""
        try:
            self.writer.close()
            await self.writer.wait_closed()
            logger.info(f"Connection closed for {self.addr}")
        except Exception as e:
>>>>>>> origin/main
            logger.error(f"Error during cleanup for {self.addr}: {e}")