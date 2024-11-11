import logging
from typing import Dict, Any, List
from .base_handler import BaseHandler
from ..model.pear_detector import PearDetector
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ClassificationHandler(BaseHandler):
    def __init__(self, detector: PearDetector):
        self.detector = detector

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.validate(request)

            if 'images' not in request:
                raise ValidationError("No images provided")

            results = []
            for image_path in request['images']:
                result = await self._process_image(image_path)
                results.append(result)

            return self._create_response(results)

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._create_response(None, str(e))

    async def _process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            result = await self.detector.detect(image)

            return {
                "path": image_path,
                "is_defective": result.is_defective,
                "confidence": result.confidence,
                "bbox": result.bbox
            }

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "path": image_path,
                "error": str(e)
            }
