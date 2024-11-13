class ServerError(Exception):
    """Base exception for server-related errors."""
    pass

class ConfigurationError(ServerError):
    """Raised when there's an error in configuration."""
    pass

class ModelError(ServerError):
    """Raised when there's an error with the ML model."""
    pass

class ValidationError(ServerError):
    """Raised when there's an error in request validation."""
    pass

class ConnectionError(ServerError):
    """Raised when there's an error in network connection."""
    pass

class InferenceError(ServerError):
    """Raised when there's an error in model inference."""
    pass