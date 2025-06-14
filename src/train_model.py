import os
from datetime import datetime, timedelta
import logging
import numpy as np
from src.trading.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.models.lstm_model import LSTMTradingModel
from src.utils.config import SYMBOL, TIMEFRAME, MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_model_state(model, stage=""):
    """Verify model state and log details"""
    logger.info(f"\nModel state verification at {stage}:")
    logger.info(f"  Model object exists: {model is not None}")
    logger.info(f"  Model.model exists: {model.model is not None if model else False}")
    logger.info(f"  Feature columns set: {model.feature_columns is not None if model else False}")
    if model and model.feature_columns:
        logger.info(f"  Number of features: {len(model.feature_columns)}")
    logger.info(f"  Sequence length: {model.sequence_length if model else None}")
    if model and model.model:
        logger.info(f"  Model input shape: {model.model.input_shape}")
        logger.info(f"  Model output shape: {model.model.output_shape}")
        logger.info(f"  Model is compiled: {model.model.optimizer is not None}")

def train_model(symbol: str = SYMBOL, timeframe: str = TIMEFRAME):
    """Train the LSTM model"""
    try:
        # Initialize MT5 connection
        mt5 = MT5Connector()
        
        # Get historical data
        logger.info(f"Fetching historical data for {symbol}")
        df = mt5.get_historical_data(symbol, timeframe)
        if df.empty:
            raise ValueError("No historical data retrieved")
            
        logger.info(f"Retrieved {len(df)} data points")
        
        # Add technical indicators
        data_processor = DataProcessor()
        df = data_processor.add_technical_indicators(df)
        
        # Define feature columns
        feature_columns = [
            # Price-based features
            'returns',
            'volatility_ratio',
            'trend_strength',
            'price_momentum',
            
            # Trend indicators
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_diff',
            'adx', 'adx_pos', 'adx_neg',
            
            # Momentum indicators
            'rsi_7', 'rsi_14', 'rsi_21',
            'stoch_k', 'stoch_d',
            
            # Volatility indicators
            'bb_high', 'bb_low', 'bb_mid', 'bb_width',
            'atr',
            
            # Volume indicators
            'obv'
        ]
        
        # Initialize and train model
        model = LSTMTradingModel(sequence_length=60)
        model.set_feature_columns(feature_columns)
        
        # Prepare sequences
        X, y = data_processor.prepare_sequences(df[feature_columns], df['returns'].shift(-1))
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model.save_model(os.path.join(MODEL_DIR, f'lstm_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'))
        
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        mt5.disconnect()

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Run training
    try:
        model_path = train_model()
        logger.info(f"\nModel saved to: {model_path}")
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}")
        raise 