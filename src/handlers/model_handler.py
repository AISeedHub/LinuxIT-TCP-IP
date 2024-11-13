import logging
from typing import Dict, Any, List
import os
from .base_handler import BaseHandler, ResponseData

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command_handlers = {
            0x22: self._handle_current_model,
            0x24: self._handle_list_model,
            0x26: self._handle_model_change,
            0x28: self._handle_delete_model
        }

        handler = command_handlers.get(self.command_code)
        if not handler:
            raise ValueError(f"Invalid command code for ModelHandler: {hex(self.command_code)}")

        return await handler(request)

    async def _handle_current_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.create_response([{
            "result": self.model.config.model_path,
            "error_code": 0
        }])

    async def _handle_list_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            models = os.listdir(self.model.config.model_path.split("/")[0])
            if not models:
                return self.create_response([{
                    "result": None,
                    "error_code": 2  # no models found
                }])

            return self.create_response([
                {"file_name": model, "result": None}
                for model in models
            ])
        except Exception as e:
            logger.error(f"List models error: {e}")
            return self.create_error_response(2)

    async def _handle_model_change(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model_name = request["request_data"][0]
            self.model.change_model(model_name)
            return self.create_response([{
                "file_name": model_name,
                "result": 2
            }])
        except Exception as e:
            logger.error(f"Model change error: {e}")
            return self.create_response([{
                "file_name": request["request_data"][0],
                "result": 1,  # failure
                "error_code": 2
            }])

    async def _handle_delete_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model_name = request["request_data"][0]
        if model_name == self.model.config.model_path.split("/")[1]:
            return self.create_response([{
                "file_name": model_name,
                "result": 1,  # failure, cannot delete current model
                "error_code": 1
            }])

        try:
            os.remove(os.path.join(
                self.model.config.model_path.split("/")[0],
                model_name
            ))
            return self.create_response([{
                "file_name": model_name,
                "result": 2,
                "error_code": 0
            }])
        except Exception as e:
            logger.error(f"Delete model error: {e}")
            return self.create_response([{
                "file_name": model_name,
                "result": 1,
                "error_code": 2
            }])
