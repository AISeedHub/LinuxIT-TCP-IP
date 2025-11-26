import logging
from typing import Dict, Any, List
from .base_handler import BaseHandler, ResponseData

logger = logging.getLogger(__name__)


class ClassificationHandler(BaseHandler):
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self.command_code == 0x01:  # request_classification
            return await self._handle_classification(request)
        elif self.command_code == 0x03:  # stop_classification
            return await self._handle_stop()
        else:
            raise ValueError(f"Invalid command code for ClassificationHandler: {hex(self.command_code)}")

    async def _handle_classification(self, request: Dict[str, Any]) -> Dict[str, Any]:
        response_data = []
        files = request.get("request_data", [])

        for file_name in files:
            response = ResponseData(file_name=file_name)
            try:
                result = await self.model.inference(file_name)
                error = result.error.error
                if error:
                    response.error_code = 2
                    response.result = None
                else:
                    result = result.is_normal
                    response.error_code = 0
                    response.result = result
            except Exception as e:
                logger.error(f"Classification error: {e}")
                response.error_code = 4  # timeout
                response.result = None

            response_data.append(vars(response))

        return self.create_response(response_data)

    async def _handle_stop(self) -> Dict[str, Any]:
        try:
            #await self.model.ease_model()
            return self.create_response([{"result": 2, "error_code": 0}])
        except Exception as e:
            logger.error(f"Stop classification error: {e}")
            return self.create_error_response(4)
