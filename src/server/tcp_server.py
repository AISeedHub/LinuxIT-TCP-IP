import asyncio
import logging
from typing import Optional, List
from dataclasses import dataclass
from .peer import Peer
from .connection_manager import ConnectionManager
from ..model.pear_detector import PearDetector
from ..utils.exceptions import ServerError

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    host: str
    port: int
    max_connections: int
    buffer_size: int
    encoding: str = 'utf-8'


class TCPServer:
    def __init__(self, config: ServerConfig, detector: PearDetector):
        self.config = config
        self.detector = detector
        self.connection_manager = ConnectionManager()
        self._server: Optional[asyncio.Server] = None

    async def start(self) -> None:
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self.config.host,
                self.config.port,
                backlog=self.config.max_connections
            )

            addr = self._server.sockets[0].getsockname()
            logger.info(f'Server running on {addr}')

            async with self._server:
                await self._server.serve_forever()

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise ServerError(f"Failed to start server: {e}")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = Peer(reader, writer, self.config, self.detector)
        self.connection_manager.add_peer(peer)
        try:
            await peer.handle_connection()
        finally:
            self.connection_manager.remove_peer(peer)
            writer.close()
            await writer.wait_closed()