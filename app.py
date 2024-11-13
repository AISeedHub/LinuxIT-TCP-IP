# main.py
import asyncio
import logging
from src.utils.config import ConfigLoader
from src.server.tcp_server import TCPServer
from src.model.pear_detector import PearDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    try:
        # Load configuration
        config = ConfigLoader.load('config/server_config.yaml')

        # Initialize model
        detector = PearDetector(config.model)
        detector.load_model(config.model.model_path)

        # Initialize and start server
        server = TCPServer(config.server, detector)
        await server.start()

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise