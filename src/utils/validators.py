import os
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from .exceptions import ValidationError


class RequestValidator:
    @staticmethod
    def validate_basic_structure(request: Dict[str, Any]) -> None:
        """Validate basic request structure"""
        if not isinstance(request, dict):
            raise ValidationError("Request must be a dictionary")

        if "cmd" not in request:
            raise ValidationError("Request must contain 'cmd' field")

        if "request_data" not in request:
            raise ValidationError("Request must contain 'request_data' field")

    @staticmethod
    def validate_file_list(files: List[str], allowed_extensions: Optional[List[str]] = None) -> None:
        """Validate list of files"""
        if not isinstance(files, list):
            raise ValidationError("File list must be a list")

        if not files:
            raise ValidationError("File list cannot be empty")

        if allowed_extensions:
            for file in files:
                if not any(file.lower().endswith(ext) for ext in allowed_extensions):
                    raise ValidationError(f"Invalid file extension for {file}. Allowed: {allowed_extensions}")

    @staticmethod
    def validate_url(url: str) -> None:
        """Validate URL format"""
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")

        if not url.startswith(('http://', 'https://')):
            raise ValidationError("URL must start with http:// or https://")

    @staticmethod
    def validate_directory_path(path: str) -> None:
        """Validate directory path"""
        if not isinstance(path, str):
            raise ValidationError("Directory path must be a string")

        if not path:
            raise ValidationError("Directory path cannot be empty")

    @staticmethod
    def validate_command_code(cmd: int) -> None:
        """Validate command code"""
        valid_commands = {
            0x01,  # request_classification
            0x03,  # stop_classification
            0x20,  # request_download
            0x22,  # request_current_model
            0x24,  # request_list_model
            0x26,  # model_change_request
            0x28,  # request_delete_model
            0x30,  # request_change_img_folder
            0x32  # request_current_img_folder
        }

        if cmd not in valid_commands:
            raise ValidationError(f"Invalid command code: {hex(cmd)}")

    @staticmethod
    def validate_file_list(files: List[str], allowed_extensions: Optional[List[str]] = None) -> None:
        """
        Validate list of files

        Args:
            files: List of file names
            allowed_extensions: List of allowed file extensions (e.g., ['.jpg', '.pt'])

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(files, list):
            raise ValidationError("File list must be a list")

        if not files:
            raise ValidationError("File list cannot be empty")

        if allowed_extensions:
            for file in files:
                if not isinstance(file, str):
                    raise ValidationError(f"Invalid file name: {file}")

                ext = os.path.splitext(file)[1].lower()
                if ext not in allowed_extensions:
                    raise ValidationError(
                        f"Invalid file extension for {file}. "
                        f"Allowed: {', '.join(allowed_extensions)}"
                    )

    @staticmethod
    def validate_url(url: str) -> None:
        """
        Validate URL format

        Args:
            url: URL string to validate

        Raises:
            ValidationError: If URL is invalid
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")

        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValidationError("Invalid URL format")

            if result.scheme not in ['http', 'https']:
                raise ValidationError("URL must use HTTP or HTTPS protocol")

        except Exception as e:
            raise ValidationError(f"Invalid URL: {e}")

    @staticmethod
    def validate_directory_path(path: str) -> None:
        """
        Validate directory path

        Args:
            path: Directory path to validate

        Raises:
            ValidationError: If path is invalid
        """
        if not isinstance(path, str):
            raise ValidationError("Directory path must be a string")

        if not path:
            raise ValidationError("Directory path cannot be empty")

        # Remove any potentially dangerous characters
        if re.search(r'[<>:"|?*]', path):
            raise ValidationError("Directory path contains invalid characters")

        # Normalize path
        try:
            normalized_path = os.path.normpath(path)
            if normalized_path.startswith('..'):
                raise ValidationError("Directory path cannot navigate above root")
        except Exception as e:
            raise ValidationError(f"Invalid directory path: {e}")

    @classmethod
    def validate_request(cls, request: Dict[str, Any]) -> None:
        """
        Comprehensive request validation

        Args:
            request: Request dictionary to validate

        Raises:
            ValidationError: If validation fails
        """
        cls.validate_basic_structure(request)
        cmd = request.get('cmd')
        cls.validate_command_code(cmd)

        request_data = request.get('request_data')

        # Validate based on command type
        if cmd == 0x01:  # request_classification
            cls.validate_file_list(request_data, ['.jpg', '.jpeg', '.png'])
        elif cmd == 0x20:  # request_download
            cls.validate_url(request_data[0])
        # elif cmd == 0x26:  # model_change_request
        #     cls.validate_file_list(request_data, ['.pt'])
        elif cmd == 0x30:  # request_change_img_folder
            cls.validate_directory_path(request_data[0])