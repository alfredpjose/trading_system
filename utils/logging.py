# utils/logging.py
import sys
from loguru import logger
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup structured logging"""
    
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handlers
    logger.add(
        f"{log_dir}/system.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="100 MB",
        retention="7 days"
    )
    
    # Separate error log
    logger.add(
        f"{log_dir}/errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="50 MB",
        retention="30 days"
    )
    
    # Trading-specific log
    logger.add(
        f"{log_dir}/trading.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        filter=lambda record: "trading" in record["name"] or "strategy" in record["name"],
        rotation="50 MB",
        retention="30 days"
    )