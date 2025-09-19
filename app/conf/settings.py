import os
from loguru import logger

from dotenv import load_dotenv

# API configs:
load_dotenv(".prod.env")

# Version
__VERSION__ = "0.0.1"

# Pathing:
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API URL GET
API_URL = os.getenv("API_URL", "http://localhost")

# API Key
API_KEY = os.getenv("API_KEY", "xyz")

if API_URL is None:
    raise ValueError("API_URL not found in environment variables")

# Logs Configs:
LOGS_DIR = os.path.join(BASE_PATH, "files", "logs")

# -- Initialize Logger:
logs_kw = dict(
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    rotation="1 week",
    compression="zip",
    backtrace=True,
)
logger.add(os.path.join(LOGS_DIR, "info_log.log"), level="INFO", **logs_kw)
logger.add(os.path.join(LOGS_DIR, "debug_log.log"), level="DEBUG", **logs_kw)
