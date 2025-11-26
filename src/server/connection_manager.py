import logging
from typing import Set
import asyncio
from .peer import Peer

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._peers: Set[Peer] = set()

    def add_peer(self, peer: Peer) -> None:
        self._peers.add(peer)
        logger.info(f"Added peer {peer.addr}. Total connections: {len(self._peers)}")

    def remove_peer(self, peer: Peer) -> None:
        self._peers.remove(peer)
        logger.info(f"Removed peer {peer.addr}. Total connections: {len(self._peers)}")

    def broadcast(self, message: dict) -> None:
        for peer in self._peers:
            asyncio.create_task(peer._send_response(message))
