import os
from datetime import datetime

# Trading Configuration
SYMBOL = "XRPUSD"  # Changed to available symbol
TIMEFRAME = "M15"  # 1-hour timeframe
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Model Configuration
SEQUENCE_LENGTH = 60  # Number of time steps to look back
FEATURE_COLUMNS = [
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
TARGET = 'close'
BATCH_SIZE = 32  # Reduced batch size for better stability
EPOCHS = 500  # Reduced epochs since we have early stopping
LEARNING_RATE = 0.0001  # Reduced learning rate for more stable training
LSTM_UNITS = 64  # Reduced units to prevent overfitting
DROPOUT_RATE = 0.2  # Increased dropout for better regularization

# Training Configuration
TRAINING_INTERVAL_DAYS = 7  # Retrain every week
GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory
EARLY_STOPPING_PATIENCE = 15  # Increased patience for early stopping
VALIDATION_SPLIT = 0.2

# TensorBoard Configuration
TENSORBOARD_UPDATE_FREQ = 'epoch'
TENSORBOARD_PROFILE_BATCH = '500,520'
TENSORBOARD_HISTOGRAM_FREQ = 1
TENSORBOARD_WRITE_GRAPH = True
TENSORBOARD_WRITE_IMAGES = True

# Model Checkpoint Configuration
CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_SAVE_WEIGHTS_ONLY = False
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_MODE = 'min'

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
HISTORY_DIR = os.path.join(BASE_DIR, 'models', 'model_history')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'models', 'tensorboard_logs')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, HISTORY_DIR, TENSORBOARD_DIR]:
    os.makedirs(directory, exist_ok=True)
