import os
import time
from datetime import datetime, timedelta
import schedule
import tensorflow as tf
import logging
import joblib
from src.trading.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.models.lstm_model import LSTMTradingModel
from src.trading.trading_agent import TradingAgent
from src.utils.config import (
    START_DATE, END_DATE, SYMBOL, TIMEFRAME,
    TRAINING_INTERVAL_DAYS, MODEL_DIR, FEATURE_COLUMNS
)
from set_env import set_mt5_env
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check and configure GPU availability"""
    logger.info("Checking GPU availability...")
    
    # Check TensorFlow version and build information
    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("Is built with CUDA: %s", tf.test.is_built_with_cuda())
    
    try:
        # Try to enable memory growth for all GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                
                # Test GPU with a simple operation
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                    logger.info("GPU test successful")
                
                # Get GPU details if available
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpus[0])
                    logger.info(f"GPU Details: {gpu_details}")
                except Exception as e:
                    logger.warning(f"Could not get GPU details: {e}")
                
                return True
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPU found, using CPU")
            # Print CPU information
            cpus = tf.config.experimental.list_physical_devices('CPU')
            logger.info("CPU Information:")
            logger.info("Num CPUs Available: %d", len(cpus))
            logger.info("Device Placement: %s", tf.config.get_visible_devices())
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
    
    return False

def train_model():
    """Train the LSTM model with historical data"""
    logger.info("Starting model training...")
    
    # Set environment variables
    set_mt5_env()
    
    # Initialize components
    mt5 = MT5Connector()
    data_processor = DataProcessor()
    model = LSTMTradingModel()
    
    try:
        # Get historical data
        logger.info("Fetching historical data...")
        df = mt5.get_historical_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df.empty:
            raise Exception("No historical data retrieved. Please check your MT5 connection and symbol settings.")
        
        logger.info(f"Retrieved {len(df)} data points")
        
        # Add technical indicators to data
        logger.info("Adding technical indicators...")
        df = data_processor.add_technical_indicators(df)
        
        # Set feature columns from config
        logger.info("Setting feature columns...")
        model.set_feature_columns(FEATURE_COLUMNS)
        
        # Prepare data for model
        logger.info("Preparing data for model...")
        X, y = data_processor.prepare_data(df)
        
        # Plot and save histogram of scaled training targets
        plt.figure(figsize=(8, 4))
        plt.hist(y, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Scaled Training Targets (y * 1000)')
        plt.xlabel('Scaled Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('scaled_training_targets_histogram.png')
        plt.close()
        logger.info('Saved scaled training targets histogram to scaled_training_targets_histogram.png')

        # Print and log class distribution for classification
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print('Class distribution:', class_dist)
        logger.info(f'Class distribution: {class_dist}')
        
        if len(X) == 0 or len(y) == 0:
            raise Exception("No data available after preparation. Please check your data processing settings.")
        
        logger.info(f"Prepared data shape - X: {X.shape}, y: {y.shape}")
        
        # Split data into train and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature scalers with feature information
        scaler_path = os.path.join(MODEL_DIR, 'scaler.save')
        save_data = {
            'scaler': data_processor.scaler,
            'is_fitted': True,
            'feature_columns': data_processor.feature_columns
        }
        joblib.dump(save_data, scaler_path)
        logger.info(f"Scaler saved to {scaler_path} with features: {data_processor.feature_columns}")
        
        # Log training metrics
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        mt5.disconnect()

def run_trading():
    """Run the trading agent"""
    logger.info("Starting trading cycle...")
    
    # Set environment variables
    set_mt5_env()
    
    # Initialize trading agent with latest model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        logger.warning("No trained model found. Please train the model first.")
        return
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    agent = TradingAgent(model_path=model_path)
    
    try:
        agent.run_trading_cycle()
    except Exception as e:
        logger.error(f"Error during trading: {e}")
    finally:
        agent.mt5.disconnect()

def main():
    """Main function to run the trading system"""
    logger.info("Initializing AI Trading System...")
    
    # Check GPU availability
    check_gpu()
    # Create necessary directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Set environment variables
    set_mt5_env()
    
    # Schedule training
    # schedule.every(TRAINING_INTERVAL_DAYS).days.at("00:00").do(train_model)
    
    # Run initial training
    # train_model()
    
    # Main loop
    while True:
        try:
            # Run scheduled tasks
            schedule.run_pending()
            
            # Run trading cycle
            run_trading()
            
            # Wait for next cycle
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Shutting down AI Trading System...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    main()
