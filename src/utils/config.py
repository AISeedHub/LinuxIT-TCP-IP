import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any
from ..server.tcp_server import ServerConfig
from ..model.detector import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str


@dataclass
class ApplicationConfig:
    server: ServerConfig
    model: ModelConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ApplicationConfig':
        return cls(
            server=ServerConfig(**config_dict['server']),
            model=ModelConfig(**config_dict['model']),
            logging=LoggingConfig(**config_dict['logging'])
        )


class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> ApplicationConfig:
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return ApplicationConfig.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

# src/