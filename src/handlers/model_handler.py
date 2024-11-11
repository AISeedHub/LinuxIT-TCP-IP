import logging
from typing import Dict, Any
from .base_handler import BaseHandler
from ..model.pear_detector import PearDetector
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    def __init__(self, detector: PearDetector):
        self.detector = detector

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.validate(request)

            action = request.get('action')
            if action not in ['load', 'unload', 'switch']:
                raise ValidationError(f"Invalid action: {action}")

            if action == 'load':
                await self._load_model(request.get('model_path'))
            elif action == 'unload':
                self._unload_model()
            elif action == 'switch':
                await self._switch_model(request.get('model_path'))

            return self._create_response({"action": action, "status": "completed"})

        except Exception as e:
            logger.error(f"Model handling error: {e}")
            return self._create_response(None, str(e))

    async def _load_model(self, model_path: str) -> None:
        if not model_path:
            raise ValidationError("Model path not provided")
        await self.detector.load_model(model_path)

    def _unload_model(self) -> None:
        self.detector.unload_model()

    async def _switch_model(self, model_path: str) -> None:
        self._unload_model()
        await self._load_model(model_path)
