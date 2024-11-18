from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import setup_logger
import logging

def main():
    # Load configuration
    config = load_config()
    logger = setup_logger(config['paths']['logs_dir'])
    logger.info("Starting demand forecasting pipeline")
    
    try:
        # TODO: Implement main pipeline
        pass
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
