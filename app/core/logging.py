import logging
import sys
from .config import settings

def setup_logging():
    level = logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

logger = logging.getLogger("ai_service")
