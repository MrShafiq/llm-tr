import logging
from main import run_trading, set_mt5_env
from src.utils.config import MODEL_DIR
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Check if model exists
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        logger.error(f"No trained model found in {MODEL_DIR}. Please train the model first.")
    else:
        logger.info("Starting trading test...")
        # Set environment variables
        set_mt5_env()
        
        # Get the latest model
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
        logger.info(f"Using model: {latest_model}")
        
        # Run trading
        run_trading() 