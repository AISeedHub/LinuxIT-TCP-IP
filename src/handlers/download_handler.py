import logging
import wget
from typing import Dict, Any
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


class DownloadHandler(BaseHandler):
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self.command_code != 0x20:
            raise ValueError(f"Invalid command code for DownloadHandler: {hex(self.command_code)}")

        try:
            url = request["request_data"][0]
            wget.download(
                url,
                out=self.model.config["DIR_MODEL_DETECTION"]
            )
            return self.create_response([{"result": 2}])  # success
        except Exception as e:
            logger.error(f"Download error: {e}")
            return self.create_response([{"result": 1}])
