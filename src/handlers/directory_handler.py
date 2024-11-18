import logging
from typing import Dict, Any
from .base_handler import BaseHandler, ResponseData
from ..model.pear_detector import PearDetector
from ..utils.exceptions import ValidationError
from ..utils.directory import verify_directory

logger = logging.getLogger(__name__)


class DirectoryHandler(BaseHandler):

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle directory management requests"""
        command_handlers = {
            0x30: self._handle_img_directory_change,
            0x32: self._handle_current_directory,
            0x34: self._handle_model_directory_change
        }

        handler = command_handlers.get(self.command_code)
        if not handler:
            raise ValueError(f"Invalid command code for DirectoryHandler: {hex(self.command_code)}")

        return await handler(request)

    async def _handle_img_directory_change(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle directory change requests"""
        new_dir = request.get("request_data")[0]
        if not new_dir:
            raise ValidationError("No directory provided")

        self.model.config.img_path = new_dir
        response = ResponseData(file_name=new_dir, result=2)
        return self.create_response([vars(response)])

    async def _handle_current_directory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get current directory"""
        response = ResponseData(file_name=self.model.config.img_path,
                                result=2)
        return self.create_response([vars(response)])

    async def _handle_model_directory_change(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model directory change requests"""
        try:
            new_dir = request.get("request_data")[0]
            new_model_dir = new_dir + "/" + self.model.config.model_path.split("/")[-1]  # get the model name
            if not verify_directory(new_dir):
                raise ValidationError("Invalid directory provided")

            self.model.config.model_path = new_model_dir  # TODO: Risky, should be validated
            response = ResponseData(file_name=new_model_dir, result=2)
            return self.create_response([vars(response)])
        except Exception as e:
            logger.error(f"Model directory change error: {e}")
            response = ResponseData(file_name=None, result=1, error_code=2)
            return self.create_response([vars(response)])
