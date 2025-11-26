from .base_handler import BaseHandler
from .classification_handler import ClassificationHandler
from .model_handler import ModelHandler
from .download_handler import DownloadHandler
from .directory_handler import DirectoryHandler
from ..model.detector import PearDetector

# Command codes as integers (0x01 -> 1, etc.)
COMMAND_CODES = {
    0x01: "request_classification",
    0x03: "stop_classification",
    0x20: "request_download",
    0x22: "request_current_model",
    0x24: "request_list_model",
    0x26: "model_change_request",
    0x28: "request_delete_model",
    0x30: "request_change_img_folder",
    0x32: "request_current_img_folder",
    0x34: "request_change_model_folder"
}

# Response codes mapping
RESPONSE_CODES = {
    0x01: 0x02,  # request_classification -> response_classification
    0x03: 0x04,  # stop_classification -> stop_response
    0x20: 0x21,  # request_download -> response_download
    0x22: 0x23,  # request_current_model -> response_current_model
    0x24: 0x25,  # request_list_model -> response_list_model
    0x26: 0x27,  # model_change_request -> model_change_response
    0x28: 0x29,  # request_delete_model -> response_delete_model
    0x30: 0x31,  # request_change_img_folder -> response_change_img_folder
    0x32: 0x33,  # request_current_img_folder -> response_current_img_folder
    0x34: 0x35  # request_change_model_folder -> response_change_model_folder
}

# Map commands to handlers
HANDLER_MAPPING = {
    0x01: ClassificationHandler,  # request_classification
    0x03: ClassificationHandler,  # stop_classification
    0x20: DownloadHandler,  # request_download
    0x22: ModelHandler,  # request_current_model
    0x24: ModelHandler,  # request_list_model
    0x26: ModelHandler,  # model_change_request
    0x28: ModelHandler,  # request_delete_model
    0x30: DirectoryHandler,  # request_change_img_folder
    0x32: DirectoryHandler,  # request_current_img_folder
    0x34: DirectoryHandler  # request_change_model_folder
}


def get_handler(command_code: int, model: PearDetector) -> BaseHandler:
    """
    Get appropriate handler for a command code.

    Args:
        command_code: Integer command code (e.g., 0x01)
        model: PearDetector instance

    Returns:
        Appropriate handler instance

    Raises:
        ValueError: If command is not recognized
    """
    handler_class = HANDLER_MAPPING.get(command_code)
    if not handler_class:
        raise ValueError(f"Unknown command code: {hex(command_code)}")

    return handler_class(model, command_code)


# Export all handlers and utilities
__all__ = [
    'BaseHandler',
    'ClassificationHandler',
    'ModelHandler',
    'DownloadHandler',
    'DirectoryHandler',
    'get_handler',
    'COMMAND_CODES',
    'RESPONSE_CODES'
]