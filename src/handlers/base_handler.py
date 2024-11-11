from abc import ABC, abstractmethod
from typing import Dict, Any
from ..utils.validators import validate_request
from ..utils.exceptions import ValidationError


class BaseHandler(ABC):
    @abstractmethod
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def validate(self, request: Dict[str, Any]) -> None:
        if not validate_request(request):
            raise ValidationError("Invalid request format")

    def _create_response(self, result: Any, error: str = None) -> Dict[str, Any]:
        if error:
            return {
                "status": "error",
                "error": error
            }
        return {
            "status": "success",
            "result": result
        }
