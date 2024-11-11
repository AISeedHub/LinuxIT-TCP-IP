import asyncio
import json
import logging
from typing import Optional
from ..handlers import get_handler
from ..utils.exceptions import ValidationError
from ..model.pear_detector import PearDetector

logger = logging.getLogger(__name__)


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

    async def handle_connection(self) -> None:
        logger.info(f"New connection from {self.addr}")
        try:
            while True:
                data = await self.reader.read(self.config.buffer_size)
                if not data:
                    break

                request = json.loads(data.decode(self.config.encoding))
                handler = get_handler(request['cmd'], self.detector)
                response = await handler.handle(request)

                await self._send_response(response)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {self.addr}: {e}")
            await self._send_error("Invalid JSON format")

        except ValidationError as e:
            logger.error(f"Validation error from {self.addr}: {e}")
            await self._send_error(str(e))

        except Exception as e:
            logger.error(f"Error handling connection from {self.addr}: {e}")
            await self._send_error("Internal server error")

        finally:
            logger.info(f"Connection closed from {self.addr}")

    async def _send_response(self, data: dict) -> None:
        response = json.dumps(data).encode(self.config.encoding)
        self.writer.write(response)
        await self.writer.drain()

    async def _send_error(self, message: str) -> None:
        error_response = {
            "error": True,
            "message": message
        }
        await self._send_response(error_response)
