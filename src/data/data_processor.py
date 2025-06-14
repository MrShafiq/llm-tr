import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..utils.config import FEATURE_COLUMNS, SEQUENCE_LENGTH, MODEL_DIR
import logging
import joblib
import os
from typing import Tuple
import ta

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, scaler_path=None):
        logger.info("Initializing DataProcessor...")
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Explicitly set feature range
        self.is_fitted = False
        self.scaler_path = scaler_path
        self.feature_columns = FEATURE_COLUMNS  # Use features from config
        self.sequence_length = SEQUENCE_LENGTH  # Use sequence length from config
        
        # Load scaler if path is provided
        if scaler_path and os.path.exists(scaler_path):
            try:
                logger.info(f"Attempting to load scaler from {scaler_path}")
                loaded_data = joblib.load(scaler_path)
                
                if isinstance(loaded_data, dict):
                    logger.info("Loading scaler from dictionary format")
                    self.scaler = loaded_data.get('scaler', MinMaxScaler(feature_range=(-1, 1)))
                    self.is_fitted = loaded_data.get('is_fitted', False)
                    if 'feature_columns' in loaded_data:
                        self.feature_columns = loaded_data['feature_columns']
                        logger.info(f"Loaded feature columns: {self.feature_columns}")
                else:
                    logger.info("Loading scaler directly")
                    self.scaler = loaded_data
                    self.is_fitted = True
                
                # Verify scaler is properly fitted for MinMaxScaler
                if hasattr(self.scaler, 'min_') and hasattr(self.scaler, 'scale_'):
                    logger.info("Scaler loaded and verified as fitted")
                    logger.info(f"Scaler statistics - Number of features: {len(self.scaler.min_)}")
                    logger.info(f"Feature mins: {self.scaler.min_}")
                    logger.info(f"Feature scales: {self.scaler.scale_}")
                else:
                    logger.warning("Loaded scaler is not properly fitted")
                    self.is_fitted = False
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
                    
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                self.is_fitted = False
                logger.warning("Using new unfitted scaler due to loading error")
        else:
            logger.info("No scaler path provided or file not found, using new unfitted scaler")
            
        logger.info(f"DataProcessor initialized - Scaler fitted: {self.is_fitted}")
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"Sequence length: {self.sequence_length}")

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        logger.info("Adding technical indicators...")
        
        # Log initial null counts
        logger.info("Initial null counts:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.info(f"{col}: {null_count} null values")
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate EMAs
        for period in [5, 10, 20, 50]:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # Calculate MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()  # Renamed from macd_hist to macd_diff
        
        # Calculate RSI for multiple periods
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # Calculate Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()  # Renamed from bb_upper
        df['bb_mid'] = bollinger.bollinger_mavg()    # Renamed from bb_middle
        df['bb_low'] = bollinger.bollinger_lband()   # Renamed from bb_lower
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Calculate ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Calculate ADX and its components
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()  # Positive Directional Indicator
        df['adx_neg'] = adx.adx_neg()  # Negative Directional Indicator
        
        # Calculate trend strength (using ADX)
        df['trend_strength'] = df['adx'] / 100.0  # Normalize to 0-1 range
        
        # Calculate volatility ratio
        df['volatility_ratio'] = df['atr'] / df['close']
        
        # Calculate price momentum (using ROC)
        df['price_momentum'] = ta.momentum.roc(df['close'], window=10)
        
        # Calculate OBV
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['tick_volume'])
        
        # Calculate market regime
        df['market_regime'] = self.detect_market_regime(df)
        
        # Fill NaN values with appropriate methods
        # For EMAs and other trend indicators, use forward fill
        trend_columns = [col for col in df.columns if 'ema_' in col or 'macd' in col or 'bb_' in col]
        df[trend_columns] = df[trend_columns].fillna(method='ffill')
        
        # For momentum indicators, use 0
        momentum_columns = [col for col in df.columns if 'rsi' in col or 'stoch' in col or 'momentum' in col]
        df[momentum_columns] = df[momentum_columns].fillna(0)
        
        # For volatility indicators, use forward fill
        volatility_columns = [col for col in df.columns if 'atr' in col or 'volatility' in col or 'adx' in col]
        df[volatility_columns] = df[volatility_columns].fillna(method='ffill')
        
        # For volume indicators, use 0
        volume_columns = [col for col in df.columns if 'obv' in col]
        df[volume_columns] = df[volume_columns].fillna(0)
        
        # For returns, use 0
        df['returns'] = df['returns'].fillna(0)
        
        # Log final null counts
        logger.warning("Null counts after filling:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"{col}: {null_count} null values")
        
        logger.info(f"Final data shape after adding indicators: {df.shape}")
        return df

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index (ADX)"""
        # Calculate True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = tr1.combine(tr2, max).combine(tr3, max)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed averages
        tr_smoothed = true_range.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr_smoothed
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr_smoothed
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (ATR)"""
        # Compute True Range (TR)
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = tr1.combine(tr2, max).combine(tr3, max)

        # Calculate ATR as rolling mean of TR
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_typical = typical_price.rolling(window=period).mean()
        deviation = typical_price - sma_typical
        mean_deviation = deviation.abs().rolling(window=period).mean()
        cci = (typical_price - sma_typical) / (0.015 * mean_deviation)
        return cci

    def detect_market_regime(self, df):
        """Detect market regime based on multiple indicators"""
        # Initialize regime column
        regime = pd.Series(index=df.index, dtype='str')
        
        # Calculate trend strength using ADX and EMAs
        trend_strength = df['trend_strength']  # Already normalized to 0-1 range
        
        # Calculate volatility state using ATR
        vol_ratio = df['volatility_ratio']
        
        # Calculate momentum state using RSI
        momentum = df['rsi_14']  # Using 14-period RSI
        
        # Determine regime with enhanced logic
        for i in range(len(df)):
            if pd.isna(df['adx'].iloc[i]) or pd.isna(trend_strength.iloc[i]):
                regime.iloc[i] = 'normal'
                continue
                
            if df['adx'].iloc[i] > 25:  # Strong trend threshold
                if trend_strength.iloc[i] > 0.7:  # Strong trend threshold
                    regime.iloc[i] = 'strong_trend'
                else:
                    regime.iloc[i] = 'trending'
            elif vol_ratio.iloc[i] > 0.02:  # High volatility threshold
                regime.iloc[i] = 'volatile'
            elif abs(momentum.iloc[i] - 50) < 10:  # RSI near middle
                regime.iloc[i] = 'ranging'
            elif momentum.iloc[i] > 70:  # Overbought condition
                regime.iloc[i] = 'overbought'
            elif momentum.iloc[i] < 30:  # Oversold condition
                regime.iloc[i] = 'oversold'
            else:
                regime.iloc[i] = 'normal'
        
        return regime

    def prepare_prediction_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction with proper scaling"""
        logger.info("Preparing prediction data...")
        
        # Verify data
        if data.empty:
            raise ValueError("Empty DataFrame provided for prediction")
            
        # Check for required features
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in prediction data: {missing_features}")
            
        # Check for null values
        null_counts = data[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in features:")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"{col}: {null_counts[col]} null values")
            # Handle nulls with forward fill first
            data[self.feature_columns] = data[self.feature_columns].fillna(method='ffill')
            # Then backward fill any remaining nulls
            data[self.feature_columns] = data[self.feature_columns].fillna(method='bfill')
            # Finally, fill any still remaining nulls with 0
            data[self.feature_columns] = data[self.feature_columns].fillna(0)
            
        # Ensure scaler is fitted
        if not self.is_fitted:
            logger.warning("Scaler not fitted, fitting with provided data")
            try:
                self.fit_scaler(data[self.feature_columns])
                logger.info("Scaler fitted successfully with prediction data")
                
                # Save the fitted scaler
                if self.scaler_path:
                    save_data = {
                        'scaler': self.scaler,
                        'is_fitted': True,
                        'feature_columns': self.feature_columns
                    }
                    joblib.dump(save_data, self.scaler_path)
                    logger.info(f"Fitted scaler saved to {self.scaler_path}")
            except Exception as e:
                logger.error(f"Failed to fit scaler: {str(e)}")
                raise RuntimeError("Could not fit scaler with prediction data")
        
        # Verify scaler is still fitted
        if not hasattr(self.scaler, 'min_') or not hasattr(self.scaler, 'scale_'):
            raise RuntimeError("Scaler is not properly fitted")
            
        # Scale the features
        try:
            scaled_data = self.scaler.transform(data[self.feature_columns])
            logger.info(f"Data scaled successfully. Shape: {scaled_data.shape}")
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise RuntimeError("Failed to scale prediction data")
        
        # Reshape for LSTM input (samples, time steps, features)
        try:
            reshaped_data = scaled_data.reshape(1, len(data), len(self.feature_columns))
            logger.info(f"Data reshaped for LSTM. Final shape: {reshaped_data.shape}")
            return reshaped_data
        except Exception as e:
            logger.error(f"Error reshaping data: {str(e)}")
            raise RuntimeError("Failed to reshape data for LSTM input")

    def fit_scaler(self, data: pd.DataFrame) -> None:
        """Fit the scaler with robust handling of edge cases"""
        logger.info("Fitting scaler with robust preprocessing...")
        
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Verify input data
        if data.empty:
            raise ValueError("Empty DataFrame provided for scaling")
            
        # Verify all required features are present
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
            
        # Check for null values
        null_counts = data[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in features:")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"{col}: {null_counts[col]} null values")
            # Handle nulls with forward fill first
            data[self.feature_columns] = data[self.feature_columns].fillna(method='ffill')
            # Then backward fill any remaining nulls
            data[self.feature_columns] = data[self.feature_columns].fillna(method='bfill')
            # Finally, fill any still remaining nulls with 0
            data[self.feature_columns] = data[self.feature_columns].fillna(0)
            
        # Log pre-scaling statistics
        logger.info("Pre-scaling statistics:")
        for col in self.feature_columns:
            stats = data[col].describe()
            logger.info(f"{col}:")
            logger.info(f"  Mean: {stats['mean']:.6f}")
            logger.info(f"  Std: {stats['std']:.6f}")
            logger.info(f"  Min: {stats['min']:.6f}")
            logger.info(f"  Max: {stats['max']:.6f}")
        
        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # For each column, handle extreme values before scaling
        for col in self.feature_columns:
            # Get robust statistics
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Clip extreme values
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Verify no infinite values
            if np.isinf(data[col]).any():
                logger.warning(f"Infinite values detected in {col} after clipping")
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill any remaining NaN values with median
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                logger.warning(f"Filled NaN values in {col} with median: {median_val:.6f}")
        
        # Verify data before scaling
        if data[self.feature_columns].isnull().any().any():
            raise ValueError("NaN values remain after preprocessing")
        if np.isinf(data[self.feature_columns].values).any():
            raise ValueError("Infinite values remain after preprocessing")
        
        # Initialize and fit the scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(data[self.feature_columns])
        
        # Verify scaler statistics
        logger.info("Scaler statistics after fitting:")
        for i, col in enumerate(self.feature_columns):
            scale = self.scaler.scale_[i]
            min_ = self.scaler.min_[i]
            logger.info(f"{col}:")
            logger.info(f"  Scale: {scale:.6f}")
            logger.info(f"  Min: {min_:.6f}")
        
        self.is_fitted = True
        logger.info("Scaler fitted successfully")
        
        # Save the fitted scaler
        if self.scaler_path:
            try:
                save_data = {
                    'scaler': self.scaler,
                    'is_fitted': True,
                    'feature_columns': self.feature_columns
                }
                joblib.dump(save_data, self.scaler_path)
                logger.info(f"Fitted scaler saved to {self.scaler_path}")
            except Exception as e:
                logger.error(f"Failed to save fitted scaler: {e}")
    
    def transform_features(self, data: pd.DataFrame) -> np.ndarray:
        """Transform features with robust error handling"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
            
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Verify input data
        if data.empty:
            raise ValueError("Empty DataFrame provided for transformation")
        if data.isnull().any().any():
            raise ValueError("DataFrame contains null values before transformation")
        
        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # For each column, handle extreme values before scaling
        for col in data.columns:
            # Get robust statistics
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Clip extreme values
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Fill any remaining NaN values with median
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                logger.warning(f"Filled NaN values in {col} with median: {median_val:.6f}")
        
        # Transform the data
        try:
            scaled_data = self.scaler.transform(data)
        except Exception as e:
            logger.error(f"Error during scaling: {str(e)}")
            logger.error("Data statistics before scaling:")
            for col in data.columns:
                stats = data[col].describe()
                logger.error(f"{col}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")
            raise
        
        # Verify transformed data
        if np.isnan(scaled_data).any():
            raise ValueError("NaN values detected after scaling")
        if np.isinf(scaled_data).any():
            raise ValueError("Infinite values detected after scaling")
        
        # Log post-scaling statistics
        logger.info("Post-scaling statistics:")
        for i, col in enumerate(data.columns):
            stats = {
                'mean': np.mean(scaled_data[:, i]),
                'std': np.std(scaled_data[:, i]),
                'min': np.min(scaled_data[:, i]),
                'max': np.max(scaled_data[:, i])
            }
            logger.info(f"{col} (scaled):")
            logger.info(f"  Mean: {stats['mean']:.6f}")
            logger.info(f"  Std: {stats['std']:.6f}")
            logger.info(f"  Min: {stats['min']:.6f}")
            logger.info(f"  Max: {stats['max']:.6f}")
        
        return scaled_data

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training with enhanced validation"""
        logger.info("Preparing data for model training...")
        
        # Verify input data
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Log initial data statistics
        logger.info("Initial data statistics:")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check for null values in input data
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning("Null values in input data:")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"{col}: {null_counts[col]} null values")
            # Forward fill null values first
            df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill')
            # Backward fill any remaining nulls
            df[self.feature_columns] = df[self.feature_columns].fillna(method='bfill')
            # Fill any still remaining nulls with 0
            df[self.feature_columns] = df[self.feature_columns].fillna(0)
            logger.info("Null values handled with forward fill, backward fill, and zero fill")
        
        # Calculate returns with proper handling
        df['returns'] = df['close'].pct_change()
        df['returns'] = df['returns'].fillna(0)  # Fill first row with 0
        
        # Handle extreme returns (winsorize at 5% to allow larger moves)
        returns_mean = df['returns'].mean()
        returns_std = df['returns'].std()
        lower_bound = returns_mean - 5 * returns_std  # Increased from 3 to 5
        upper_bound = returns_mean + 5 * returns_std
        df['returns'] = df['returns'].clip(lower=lower_bound, upper=upper_bound)
        
        # Log returns statistics
        logger.info("Returns statistics after winsorization:")
        logger.info(f"Mean: {df['returns'].mean():.6f}")
        logger.info(f"Std: {df['returns'].std():.6f}")
        logger.info(f"Min: {df['returns'].min():.6f}")
        logger.info(f"Max: {df['returns'].max():.6f}")
        
        # Feature-specific preprocessing
        for col in self.feature_columns:
            if col != 'returns':  # Skip returns as we already handled it
                # First ensure no null values
                if df[col].isnull().any():
                    logger.warning(f"Null values found in {col} before preprocessing")
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                if col == 'obv':
                    # For OBV, use percentage change instead of raw values
                    df[col] = df[col].pct_change()
                    df[col] = df[col].fillna(0)
                    # Clip extreme values
                    df[col] = df[col].clip(lower=-0.2, upper=0.2)  # Increased from 0.1 to 0.2
                elif col == 'tick_volume':
                    # For volume, use log transform
                    df[col] = np.log1p(df[col])
                elif col in ['spread']:
                    # For spread, use min-max scaling
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                elif col in ['rsi_7', 'rsi_14', 'rsi_21']:
                    # For RSI, normalize to 0-1 range
                    df[col] = df[col] / 100.0
                elif col in ['stoch_k', 'stoch_d']:
                    # For stochastic, normalize to 0-1 range
                    df[col] = df[col] / 100.0
                elif col in ['adx', 'adx_pos', 'adx_neg']:
                    # For ADX, normalize to 0-1 range
                    df[col] = df[col] / 100.0
                elif col in ['bb_width']:
                    # For BB width, clip extreme values
                    df[col] = df[col].clip(lower=-0.2, upper=0.2)  # Increased from 0.1 to 0.2
                elif col in ['volatility_ratio']:
                    # For volatility ratio, clip extreme values
                    df[col] = df[col].clip(lower=-0.02, upper=0.02)  # Increased from 0.01 to 0.02
                elif col in ['trend_strength']:
                    # For trend strength, ensure it's between -1 and 1
                    df[col] = df[col].clip(lower=-1, upper=1)
                elif col in ['price_momentum']:
                    # For price momentum, clip extreme values
                    df[col] = df[col].clip(lower=-0.02, upper=0.02)  # Increased from 0.01 to 0.02
                elif col in ['macd', 'macd_signal', 'macd_diff']:
                    # For MACD, normalize by price
                    df[col] = df[col] / df['close']
                elif col in ['ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
                           'bb_high', 'bb_low', 'bb_mid']:
                    # For price-based indicators, use percentage of current price
                    df[col] = (df[col] - df['close']) / df['close']
                elif col == 'atr':
                    # For ATR, normalize by price
                    df[col] = df[col] / df['close']
                
                # Verify no null values after preprocessing
                if df[col].isnull().any():
                    logger.warning(f"Null values found in {col} after preprocessing, filling with 0")
                    df[col] = df[col].fillna(0)
                
                # Winsorize remaining outliers with wider bounds
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 5 * std  # Increased from 3 to 5
                upper_bound = mean + 5 * std
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Processed {col} - New range: [{df[col].min():.6f}, {df[col].max():.6f}]")
        
        # Final null check
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values remain after preprocessing: {null_counts[null_counts > 0]}")
        
        # Scale features using the new transform_features method
        if not self.is_fitted:
            logger.info("Fitting scaler on training data...")
            self.fit_scaler(df[self.feature_columns])
        
        # Transform features
        scaled_features = self.transform_features(df[self.feature_columns])
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            # Get sequence of features
            sequence = scaled_features[i:(i + self.sequence_length)]
            # Get target (next period's return)
            target = df['returns'].iloc[i + self.sequence_length]
            
            # Skip sequences with extreme target values
            if abs(target) > 0.02:  # Increased from 0.01 to 0.02
                continue
                
            # Verify sequence and target
            if np.isnan(sequence).any() or np.isnan(target):
                logger.warning(f"Skipping sequence at index {i} due to NaN values")
                continue
            if np.isinf(sequence).any() or np.isinf(target):
                logger.warning(f"Skipping sequence at index {i} due to infinite values")
                continue
                
            X.append(sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Final validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid sequences created after preparation")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values in final prepared data")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Infinite values in final prepared data")
            
        # Log final prepared data statistics
        logger.info("Final prepared data statistics:")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"X - Mean: {np.mean(X):.6f}, Std: {np.std(X):.6f}")
        logger.info(f"y - Mean: {np.mean(y):.6f}, Std: {np.std(y):.6f}")
        logger.info(f"y - Min: {np.min(y):.6f}, Max: {np.max(y):.6f}")
        logger.info(f"Number of sequences: {len(X)}")
        
        return X, y

    def inverse_transform(self, scaled_return, current_price):
        """Convert scaled model output back to original price scale"""
        # No need to divide by 1000 since we're using linear activation
        predicted_price = current_price * (1 + scaled_return)
        return predicted_price

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training with improved null handling"""
        logger.info("Preparing sequences for training...")
        
        # Verify all required features exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Check for null values
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in features:")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"{col}: {null_counts[col]} null values")
            # Instead of raising an error, we'll handle nulls
            df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill').fillna(0)
        
        # Scale features
        if not self.is_fitted:
            self.fit_scaler(df[self.feature_columns])
        
        # Scale the features
        scaled_features = self.scaler.transform(df[self.feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            # Target is the next period's return
            y.append(df['returns'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
