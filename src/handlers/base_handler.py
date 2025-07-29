from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..model.detector import PearDetector
from ..utils.exceptions import ValidationError


@dataclass
class ResponseData:
    file_name: Optional[str] = None
    result: Optional[Any] = None
    error_code: int = 0


class BaseHandler(ABC):
    def __init__(self, model: 'PearDetector', command_code: int):
        self.model = model
        self.command_code = command_code
        from . import RESPONSE_CODES
        self.response_code = RESPONSE_CODES.get(command_code)

    @abstractmethod
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the request and return appropriate response"""
        pass

    def create_response(self, response_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a properly formatted response"""
        return {
            "cmd": self.response_code,
            "response_data": response_data,
            "request_data": None
        }

    def create_error_response(self, error_code: int) -> Dict[str, Any]:
        """Create an error response"""
        return self.create_response([{
            "file_name": None,
            "result": None,
            "error_code": error_code
        }])
