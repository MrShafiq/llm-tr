import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.trading.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.models.lstm_model import LSTMTradingModel
from src.utils.config import SYMBOL, TIMEFRAME, MODEL_DIR
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, model_path=None):
        self.mt5 = MT5Connector()
        
        # Define the mapping of placeholder features to actual features
        self.feature_mapping = {
            # Price-based features
            'feature_0': 'returns',
            'feature_1': 'volatility_ratio',
            'feature_2': 'trend_strength',
            'feature_3': 'price_momentum',
            
            # Trend indicators
            'feature_4': 'ema_5',
            'feature_5': 'ema_10',
            'feature_6': 'ema_20',
            'feature_7': 'ema_50',
            'feature_8': 'macd',
            'feature_9': 'macd_signal',
            'feature_10': 'macd_diff',
            'feature_11': 'adx',
            'feature_12': 'adx_pos',
            'feature_13': 'adx_neg',
            
            # Momentum indicators
            'feature_14': 'rsi_7',
            'feature_15': 'rsi_14',
            'feature_16': 'rsi_21',
            'feature_17': 'stoch_k',
            'feature_18': 'stoch_d',
            
            # Volatility indicators
            'feature_19': 'bb_high',
            'feature_20': 'bb_low',
            'feature_21': 'bb_mid',
            'feature_22': 'bb_width',
            'feature_23': 'atr',
            
            # Volume indicators
            'feature_24': 'obv'
        }
        
        # Initialize model with the same sequence length used in training
        self.model = LSTMTradingModel(sequence_length=60)
        
        if model_path:
            # Load model first to get feature columns
            logger.info(f"Loading model from {model_path}")
            self.model.load_model(model_path)
            
            # Initialize DataProcessor with the correct scaler path
            scaler_path = model_path.replace('.h5', '_scaler.save')
            logger.info(f"Initializing DataProcessor with scaler path: {scaler_path}")
            self.data_processor = DataProcessor(scaler_path=scaler_path)
            
            # Map placeholder features to actual features
            if self.model.feature_columns and all(f.startswith('feature_') for f in self.model.feature_columns):
                actual_features = [self.feature_mapping[f] for f in self.model.feature_columns]
                self.model.feature_columns = actual_features
                self.data_processor.feature_columns = actual_features
                logger.info(f"Mapped placeholder features to actual features: {actual_features}")
            else:
                # If feature columns are not set or not placeholders, use the default training features
                default_features = list(self.feature_mapping.values())
                self.data_processor.feature_columns = default_features
                self.model.feature_columns = default_features
                logger.info(f"Using default features: {default_features}")
        else:
            # Initialize DataProcessor without scaler path for new model
            self.data_processor = DataProcessor()
            # Use default features
            default_features = list(self.feature_mapping.values())
            self.data_processor.feature_columns = default_features
            self.model.feature_columns = default_features
            logger.info(f"Using default features for new model: {default_features}")
            
        # Ensure scaler is fitted with historical data
        logger.info("Ensuring scaler is properly fitted...")
        try:
            # Get historical data and add technical indicators
            historical_data = self.get_historical_data(days=30)
            if historical_data.empty:
                raise ValueError("No historical data retrieved for scaler fitting")
                
            logger.info(f"Retrieved {len(historical_data)} historical data points for scaler fitting")
            
            # Add technical indicators
            historical_data = self.data_processor.add_technical_indicators(historical_data)
            logger.info(f"Added technical indicators, data shape: {historical_data.shape}")
            
            # Verify all required features are present
            missing_features = [col for col in self.model.feature_columns if col not in historical_data.columns]
            if missing_features:
                raise ValueError(f"Missing features in historical data: {missing_features}")
            
            # Log data shape and features before fitting
            logger.info(f"Historical data shape: {historical_data.shape}")
            logger.info(f"Features to fit scaler: {self.model.feature_columns}")
            logger.info(f"Feature columns in data: {historical_data.columns.tolist()}")
            
            # Check for null values
            null_counts = historical_data[self.model.feature_columns].isnull().sum()
            if null_counts.any():
                logger.warning("Null values found in features:")
                for col in null_counts[null_counts > 0].index:
                    logger.warning(f"{col}: {null_counts[col]} null values")
            
            # Fit scaler with complete feature set
            self.data_processor.fit_scaler(historical_data[self.model.feature_columns])
            logger.info("Scaler successfully fitted with historical data")
            
            # Verify scaler is fitted
            if not self.data_processor.is_fitted:
                raise RuntimeError("Scaler fitting failed - is_fitted flag is False")
                
            # Verify scaler attributes for MinMaxScaler
            if not hasattr(self.data_processor.scaler, 'min_') or not hasattr(self.data_processor.scaler, 'scale_'):
                raise RuntimeError("Scaler fitting failed - missing required MinMaxScaler attributes")
                
            logger.info("Scaler verification successful")
            logger.info(f"Number of features in scaler: {len(self.data_processor.scaler.min_)}")
            logger.info(f"Scaler min values: {self.data_processor.scaler.min_}")
            logger.info(f"Scaler scale values: {self.data_processor.scaler.scale_}")
            
            # Update model's scaler and fitted status
            self.model.scaler = self.data_processor.scaler
            self.model.is_fitted = True
            logger.info("Updated model's scaler and fitted status")
            
            # Save the fitted scaler
            if model_path:
                scaler_path = model_path.replace('.h5', '_scaler.save')
                save_data = {
                    'scaler': self.data_processor.scaler,
                    'is_fitted': True,
                    'feature_columns': self.model.feature_columns
                }
                joblib.dump(save_data, scaler_path)
                logger.info(f"Saved fitted scaler to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Failed to fit scaler during initialization: {str(e)}")
            raise RuntimeError(f"Could not initialize scaler - {str(e)}")
            
        logger.info("Backtester initialization completed successfully")
        logger.info(f"Model fitted status: {self.model.is_fitted}")
        logger.info(f"DataProcessor fitted status: {self.data_processor.is_fitted}")
        
        self.initial_balance = 30.0
        self.balance = self.initial_balance
        self.position = None
        self.position_size = 0.01
        self.entry_price = 0
        self.trades = []
        self.prediction_log = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.daily_loss_limit = 0.05
        self.daily_pnl = 0
        self.last_trade_date = None
        self.min_trades_per_day = 2
        self.trades_today = 0
        self.profit_target = 0.018
        self.min_prediction_threshold = 0.00002
        self.max_prediction_threshold = 0.00012
        self.trend_confirmation_periods = 2
        self.min_adx_threshold = 8.5
        self.max_volatility_threshold = 0.0045
        self.min_trend_strength = 0.065
        self.min_balance = 5.0
        self.max_position_size = 0.01
        self.min_position_size = 0.01
        self.max_risk_per_trade = 0.01
        self.pip_value = 0.1
        self.stop_loss_pips = 10
        self.take_profit_pips = 25
        self.prediction_scale_factor = 500.0
        self.min_rsi = 20
        self.max_rsi = 80
        self.min_bb_width = 0.001
        self.max_bb_width = 0.013
        self.min_prediction_confidence = 0.56
        self.prediction_history = []
        self.prediction_window = 5
        self.min_prediction_agreement = 0.56
        self.max_loss_per_trade = 0.025
        self.hard_stop_loss = 0.8
        self.max_daily_loss = 0.12

    def get_historical_data(self, days=30):
        """Get historical data for backtesting"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        return self.mt5.get_historical_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=start_date,
            end_date=end_date
        )
    
    def validate_prediction(self, current_price, predicted_price):
        """Validate prediction based on historical accuracy and agreement"""
        # Calculate prediction
        price_diff_percent = (predicted_price - current_price) / current_price
        
        # Add to prediction history
        self.prediction_history.append(price_diff_percent)
        if len(self.prediction_history) > self.prediction_window:
            self.prediction_history.pop(0)
            
        # Need enough predictions to validate
        if len(self.prediction_history) < self.prediction_window:
            # For initial predictions, require stronger signals
            if price_diff_percent < 0:
                # More lenient for SELL signals
                if abs(price_diff_percent) < self.min_prediction_threshold * 1.2:
                    return False
            else:
                # Stricter for BUY signals
                if abs(price_diff_percent) < self.min_prediction_threshold * 1.5:
                    return False
            return True
            
        # Calculate directional agreement with focus on recent predictions
        current_direction = np.sign(price_diff_percent)
        recent_predictions = self.prediction_history[-3:]  # Focus on last 3 predictions
        agreement_count = sum(1 for p in recent_predictions if np.sign(p) == current_direction)
        recent_agreement = agreement_count / len(recent_predictions)
        
        # Calculate overall directional agreement
        all_agreement_count = sum(1 for p in self.prediction_history if np.sign(p) == current_direction)
        overall_agreement = all_agreement_count / len(self.prediction_history)
        
        # Calculate magnitude agreement with direction-specific bounds
        if price_diff_percent < 0:
            magnitude_threshold = abs(price_diff_percent) * 0.45  # More lenient for SELL
        else:
            magnitude_threshold = abs(price_diff_percent) * 0.35  # Stricter for BUY
        magnitude_agreement = sum(1 for p in self.prediction_history 
                                if abs(p - price_diff_percent) <= magnitude_threshold) / len(self.prediction_history)
        
        # Combined agreement score with direction-specific weights
        if price_diff_percent < 0:
            # More weight on recent predictions for SELL signals
            combined_agreement = (recent_agreement * 0.6 + overall_agreement * 0.2 + magnitude_agreement * 0.2)
        else:
            # More balanced weights for BUY signals
            combined_agreement = (recent_agreement * 0.5 + overall_agreement * 0.3 + magnitude_agreement * 0.2)
        
        # Log prediction validation with more detail
        logger.info(f"Prediction validation:")
        logger.info(f"  Current prediction: {price_diff_percent:.12%}")
        logger.info(f"  Direction: {'Positive' if price_diff_percent > 0 else 'Negative'}")
        logger.info(f"  Recent agreement: {recent_agreement:.2%}")
        logger.info(f"  Overall agreement: {overall_agreement:.2%}")
        logger.info(f"  Magnitude agreement: {magnitude_agreement:.2%}")
        logger.info(f"  Combined agreement: {combined_agreement:.2%}")
        logger.info(f"  Prediction history: {[f'{p:.12%}' for p in self.prediction_history]}")
        
        # Direction-specific validation thresholds
        min_agreement = self.min_prediction_agreement
        if price_diff_percent < 0:
            # More lenient for SELL signals
            min_agreement *= 0.9  # Reduced from 0.95
        return combined_agreement >= min_agreement

    def calculate_position_size(self, price, current_data):
        """Calculate position size based on dynamic risk management"""
        # Check minimum balance
        if self.balance < self.min_balance:
            logger.info(f"Balance too low (${self.balance:.2f}), skipping trade")
            return 0
            
        # Calculate maximum risk amount based on both percentage and absolute value
        max_risk_amount = min(
            self.balance * self.max_risk_per_trade,  # Percentage-based risk
            self.max_loss_per_trade,  # Absolute risk limit
            self.balance * self.max_daily_loss  # Daily loss limit
        )
        
        # Calculate position size based on stop loss
        risk_per_pip = max_risk_amount / self.stop_loss_pips
        position_size = risk_per_pip / self.pip_value
        
        # Convert to lots (1 lot = 1000 units)
        position_size = round(position_size * 1000, 2)  # Convert to units first
        position_size = round(position_size / 1000, 2)  # Then to lots
        
        # Apply position size limits
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))
        
        # Additional safety check
        if position_size <= 0:
            logger.warning(f"Invalid position size calculated: {position_size}, using minimum size")
            position_size = self.min_position_size
            
        # Verify the maximum possible loss
        max_possible_loss = position_size * 1000 * self.stop_loss_pips * self.pip_value
        if max_possible_loss > self.max_loss_per_trade:
            # Recalculate position size based on maximum loss
            position_size = (self.max_loss_per_trade / (self.stop_loss_pips * self.pip_value)) / 1000
            position_size = round(position_size, 2)
            # Ensure we don't go below minimum position size
            position_size = max(self.min_position_size, position_size)
            logger.info(f"Position size adjusted to respect maximum loss limit: {position_size:.2f} lots")
            
        # Final verification of maximum loss
        max_possible_loss = position_size * 1000 * self.stop_loss_pips * self.pip_value
        if max_possible_loss > self.max_loss_per_trade:
            logger.warning(f"Maximum loss limit exceeded, reducing position size")
            position_size = self.min_position_size
            max_possible_loss = position_size * 1000 * self.stop_loss_pips * self.pip_value
            
        # Log position sizing details
        logger.info(f"Position sizing details:")
        logger.info(f"  Balance: ${self.balance:.2f}")
        logger.info(f"  Max risk amount: ${max_risk_amount:.2f}")
        logger.info(f"  Risk per pip: ${risk_per_pip:.4f}")
        logger.info(f"  Calculated position size: {position_size:.2f} lots")
        logger.info(f"  Maximum allowed position: {self.max_position_size:.2f} lots")
        logger.info(f"  Minimum position size: {self.min_position_size:.2f} lots")
        logger.info(f"  Maximum possible loss: ${max_possible_loss:.2f}")
        logger.info(f"  Stop loss pips: {self.stop_loss_pips}")
        logger.info(f"  Take profit pips: {self.take_profit_pips}")
        if hasattr(self, 'consecutive_losses'):
            logger.info(f"  Consecutive losses: {self.consecutive_losses}")
        logger.info(f"  Trades today: {self.trades_today}")
        
        return position_size
    
    def calculate_pnl(self, entry_price, current_price, position_type, position_size):
        """Calculate PnL for a position"""
        if position_type == "BUY":
            pnl = (current_price - entry_price) * position_size * 1000
        else:  # SELL
            pnl = (entry_price - current_price) * position_size * 1000
        return round(pnl, 2)
    
    def execute_trade(self, predicted_price, current_price, timestamp):
        """Execute a trade based on prediction with enhanced risk management"""
        try:
            # Check minimum balance first
            if self.balance < self.min_balance:
                logger.info(f"Balance too low (${self.balance:.2f}), skipping trade")
                return
                
            historical_data = self.get_historical_data(days=5)
            historical_data = self.data_processor.add_technical_indicators(historical_data)
            current_data = historical_data.iloc[-1]
            
            # Calculate raw and scaled price differences with more precision
            raw_price_diff = (predicted_price - current_price) / current_price
            price_diff_percent = raw_price_diff * self.prediction_scale_factor
            
            # Log prediction details with more precision
            logger.info(f"Prediction details:")
            logger.info(f"  Raw price difference: {raw_price_diff:.12%}")
            logger.info(f"  Scaled price difference: {price_diff_percent:.12%}")
            logger.info(f"  Prediction threshold: {self.min_prediction_threshold:.12%}")
            logger.info(f"  Scale factor: {self.prediction_scale_factor}")
            
            # Calculate prediction confidence with more granularity
            if len(self.prediction_history) >= self.prediction_window:
                recent_predictions = self.prediction_history[-self.prediction_window:]
                prediction_direction = np.sign(raw_price_diff)
                agreement_count = sum(1 for p in recent_predictions if np.sign(p) == prediction_direction)
                prediction_confidence = agreement_count / len(recent_predictions)
                logger.info(f"  Prediction confidence: {prediction_confidence:.2%}")
                logger.info(f"  Recent predictions: {[f'{p:.12%}' for p in recent_predictions]}")
                
                # Calculate prediction stability
                prediction_std = np.std(recent_predictions)
                prediction_mean = np.mean(recent_predictions)
                prediction_stability = 1 - (prediction_std / (abs(prediction_mean) + 1e-10))
                logger.info(f"  Prediction stability: {prediction_stability:.2%}")
                
                # Validate prediction with more lenient conditions
                if not self.validate_prediction(current_price, predicted_price):
                    logger.info("Prediction validation failed, skipping trade")
                    return
            else:
                # For the first few predictions, use a more lenient approach
                prediction_confidence = 0.6  # Increased from 0.5
                prediction_stability = 0.6   # Increased from 0.5
                logger.info(f"  Prediction confidence: {prediction_confidence:.2%} (building history)")
                logger.info(f"  Prediction stability: {prediction_stability:.2%} (building history)")
                # Don't skip trade just because of insufficient history
                # Instead, we'll use the default confidence values
            
            # Market condition checks with direction-specific thresholds
            market_score = 0
            max_score = 5
            
            # ADX check with direction-specific threshold
            min_adx = self.min_adx_threshold
            if price_diff_percent < 0:
                min_adx *= 0.9  # More lenient for SELL signals
            if current_data['adx'] >= min_adx:
                market_score += 1
                logger.info(f"ADX acceptable ({current_data['adx']:.2f})")
            else:
                logger.info(f"ADX low ({current_data['adx']:.2f})")
            
            # Trend strength check with direction-specific threshold
            min_trend = self.min_trend_strength
            if price_diff_percent < 0:
                min_trend *= 0.9  # More lenient for SELL signals
            if abs(current_data['trend_strength']) >= min_trend:
                market_score += 1
                logger.info(f"Trend strength acceptable ({current_data['trend_strength']:.2f})")
            else:
                logger.info(f"Trend strength weak ({current_data['trend_strength']:.2f})")
            
            # Volatility check with direction-specific threshold
            max_vol = self.max_volatility_threshold
            if price_diff_percent < 0:
                max_vol *= 1.1  # More lenient for SELL signals
            if current_data['volatility_ratio'] <= max_vol:
                market_score += 1
                logger.info(f"Volatility acceptable ({current_data['volatility_ratio']:.4f})")
            else:
                logger.info(f"Volatility high ({current_data['volatility_ratio']:.4f})")
            
            # BB width check with direction-specific range
            min_bb = self.min_bb_width
            max_bb = self.max_bb_width
            if price_diff_percent < 0:
                min_bb *= 0.9  # More lenient for SELL signals
                max_bb *= 1.1
            if min_bb <= current_data['bb_width'] <= max_bb:
                market_score += 1
                logger.info(f"BB width acceptable ({current_data['bb_width']:.4f})")
            else:
                logger.info(f"BB width out of range ({current_data['bb_width']:.4f})")
            
            # RSI check with direction-specific range
            min_rsi = self.min_rsi
            max_rsi = self.max_rsi
            if price_diff_percent < 0:
                min_rsi = int(min_rsi * 0.9)  # More lenient for SELL signals
                max_rsi = int(max_rsi * 1.1)
            if min_rsi <= current_data['rsi_14'] <= max_rsi:
                market_score += 1
                logger.info(f"RSI acceptable ({current_data['rsi_14']:.2f})")
            else:
                logger.info(f"RSI out of range ({current_data['rsi_14']:.2f})")
            
            logger.info(f"Market score: {market_score}/{max_score}")
            
            # Direction-specific market score requirements
            min_required_score = 3
            if price_diff_percent < 0:
                min_required_score = 2  # More lenient for SELL signals
            if market_score < min_required_score:
                logger.info("Insufficient market conditions, skipping trade")
                return

            # Calculate position size with balanced risk
            self.position_size = self.calculate_position_size(current_price, current_data)
            
            # Skip trade if position size is too small
            if self.position_size < self.min_position_size:
                logger.info(f"Position size too small ({self.position_size:.2f} lots), skipping trade")
                return
                
            # Dynamic threshold based on market conditions, confidence, and stability
            base_threshold = self.min_prediction_threshold
            
            # Adjust threshold based on market score, confidence, and stability with balanced adjustments
            if market_score >= 4 and prediction_confidence > 0.6 and prediction_stability > 0.6:
                # Balanced reduction for strong conditions
                if price_diff_percent < 0:
                    base_threshold *= 0.45  # Slightly more aggressive for SELL signals
                else:
                    base_threshold *= 0.5
                logger.info("Significantly reducing threshold due to strong market conditions")
            elif market_score >= 3 and prediction_confidence > 0.55 and prediction_stability > 0.55:
                # Moderate reduction for good conditions
                if price_diff_percent < 0:
                    base_threshold *= 0.65  # Slightly more aggressive for SELL signals
                else:
                    base_threshold *= 0.7
                logger.info("Moderately reducing threshold due to good market conditions")
            else:
                base_threshold *= 1.1  # Small increase for weak conditions
                logger.info("Slightly increasing threshold due to weak conditions")
                
            logger.info(f"Adjusted threshold: {base_threshold:.12%}")
            
            # Additional trend confirmation with balanced conditions
            trend_confirmed = True
            for i in range(1, self.trend_confirmation_periods + 1):
                if i < len(historical_data):
                    prev_data = historical_data.iloc[-i-1]
                    # Use 65% of trend strength threshold for confirmation
                    trend_threshold = self.min_trend_strength * 0.65
                    if (price_diff_percent > 0 and prev_data['trend_strength'] < trend_threshold) or \
                       (price_diff_percent < 0 and prev_data['trend_strength'] > -trend_threshold):
                        trend_confirmed = False
                        break
            
            if not trend_confirmed:
                logger.info("Trend not confirmed, skipping trade")
                return
            
            # Execute trade if price difference exceeds threshold
            if abs(price_diff_percent) > base_threshold:
                if price_diff_percent > 0:
                    logger.info(f"Long entry triggered - Price diff: {price_diff_percent:.12%} > {base_threshold:.12%}")
                    self.position = {
                        'type': 'BUY',
                        'entry_price': current_price,
                        'entry_time': timestamp,
                        'predicted_return': price_diff_percent,
                        'bars_held': 0,
                        'size': self.position_size,
                        'threshold': base_threshold,
                        'market_conditions': {
                            'rsi_14': current_data['rsi_14'],
                            'adx': current_data['adx'],
                            'trend_strength': current_data['trend_strength'],
                            'volatility_ratio': current_data['volatility_ratio'],
                            'bb_width': current_data['bb_width'],
                            'prediction_confidence': prediction_confidence,
                            'prediction_stability': prediction_stability,
                            'market_score': market_score,
                            'raw_prediction': raw_price_diff,
                            'scaled_prediction': price_diff_percent
                        }
                    }
                elif price_diff_percent < 0:
                    logger.info(f"Short entry triggered - Price diff: {price_diff_percent:.12%} < {-base_threshold:.12%}")
                    self.position = {
                        'type': 'SELL',
                        'entry_price': current_price,
                        'entry_time': timestamp,
                        'predicted_return': price_diff_percent,
                        'bars_held': 0,
                        'size': self.position_size,
                        'threshold': base_threshold,
                        'market_conditions': {
                            'rsi_14': current_data['rsi_14'],
                            'adx': current_data['adx'],
                            'trend_strength': current_data['trend_strength'],
                            'volatility_ratio': current_data['volatility_ratio'],
                            'bb_width': current_data['bb_width'],
                            'prediction_confidence': prediction_confidence,
                            'prediction_stability': prediction_stability,
                            'market_score': market_score,
                            'raw_prediction': raw_price_diff,
                            'scaled_prediction': price_diff_percent
                        }
                    }
            else:
                logger.info(f"No entry triggered - Price diff: {price_diff_percent:.12%} between {-base_threshold:.12%} and {base_threshold:.12%}")
                    
        except Exception as e:
            logger.error(f"Failed to process technical indicators in execute_trade: {str(e)}")
            return
    
    def check_exit_conditions(self, prediction, current_price, timestamp):
        """Check if we should exit the current position with enhanced risk management"""
        if self.position is None:
            return
            
        # Check minimum balance
        if self.balance < self.min_balance:
            logger.info(f"Balance too low (${self.balance:.2f}), closing position")
            self.close_position(current_price, timestamp, "balance_too_low")
            return
            
        # Calculate current PnL
        current_pnl = self.calculate_pnl(
            self.position['entry_price'],
            current_price,
            self.position['type'],
            self.position['size']
        )
        
        # Check hard stop loss
        if self.balance + current_pnl <= self.initial_balance * (1 - self.hard_stop_loss):
            logger.info(f"Hard stop loss triggered at {current_price:.2f}")
            self.close_position(current_price, timestamp, "hard_stop_loss")
            return
            
        # Check if trade would result in negative balance
        if self.balance + current_pnl < 0:
            logger.info("Trade would result in negative balance, closing position")
            self.close_position(current_price, timestamp, "negative_balance_prevention")
            return
            
        # Update daily PnL
        self.daily_pnl = current_pnl
        
        # Check daily loss limit
        if self.daily_pnl < -self.balance * self.max_daily_loss:
            logger.info(f"Daily loss limit reached (${self.daily_pnl:.2f}), closing position")
            self.close_position(current_price, timestamp, "daily_loss_limit")
            return
            
        # Calculate stop loss and take profit prices in pips
        if self.position['type'] == "BUY":
            stop_loss_price = self.position['entry_price'] - (self.stop_loss_pips * self.pip_value)
            take_profit_price = self.position['entry_price'] + (self.take_profit_pips * self.pip_value)
            
            # Log current price relative to stop loss and take profit
            logger.info(f"Current price: {current_price:.2f}, Stop loss: {stop_loss_price:.2f}, Take profit: {take_profit_price:.2f}")
            logger.info(f"Distance to stop loss: {((current_price - stop_loss_price) / self.pip_value):.1f} pips")
            logger.info(f"Distance to take profit: {((take_profit_price - current_price) / self.pip_value):.1f} pips")
            
            if current_price <= stop_loss_price:
                self.close_position(current_price, timestamp, "stop_loss")
            elif current_price >= take_profit_price:
                self.close_position(current_price, timestamp, "take_profit")
                
        elif self.position['type'] == "SELL":
            stop_loss_price = self.position['entry_price'] + (self.stop_loss_pips * self.pip_value)
            take_profit_price = self.position['entry_price'] - (self.take_profit_pips * self.pip_value)
            
            # Log current price relative to stop loss and take profit
            logger.info(f"Current price: {current_price:.2f}, Stop loss: {stop_loss_price:.2f}, Take profit: {take_profit_price:.2f}")
            logger.info(f"Distance to stop loss: {((stop_loss_price - current_price) / self.pip_value):.1f} pips")
            logger.info(f"Distance to take profit: {((current_price - take_profit_price) / self.pip_value):.1f} pips")
            
            if current_price >= stop_loss_price:
                self.close_position(current_price, timestamp, "stop_loss")
            elif current_price <= take_profit_price:
                self.close_position(current_price, timestamp, "take_profit")
                
        # Increment bars held
        self.position['bars_held'] += 1
    
    def close_position(self, current_price, timestamp, exit_reason):
        """Helper method to close positions and update statistics"""
        if self.position is None:
            return
            
        # Calculate PnL
        pnl = self.calculate_pnl(
            self.position['entry_price'],
            current_price,
            self.position['type'],
            self.position['size']
        )
        
        # Update balance and daily PnL
        new_balance = self.balance + pnl
        
        # Safety check to prevent negative balance
        if new_balance < 0:
            logger.warning(f"Trade would result in negative balance (${new_balance:.2f}), adjusting PnL")
            pnl = -self.balance  # Limit loss to current balance
            new_balance = 0
            
        # Additional safety check for hard stop loss
        if exit_reason == "hard_stop_loss" and new_balance < self.initial_balance * (1 - self.hard_stop_loss):
            pnl = -(self.initial_balance * self.hard_stop_loss)
            new_balance = self.initial_balance * (1 - self.hard_stop_loss)
            logger.info(f"Hard stop loss enforced, limiting loss to ${abs(pnl):.2f}")
            
        # Additional safety check for daily loss limit
        if exit_reason == "daily_loss_limit" and new_balance < self.balance * (1 - self.max_daily_loss):
            pnl = -(self.balance * self.max_daily_loss)
            new_balance = self.balance * (1 - self.max_daily_loss)
            logger.info(f"Daily loss limit enforced, limiting loss to ${abs(pnl):.2f}")
            
        self.balance = new_balance
        self.daily_pnl += pnl
        self.trades_today += 1
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Log trade details
        logger.info(f"Trade closed:")
        logger.info(f"  Type: {self.position['type']}")
        logger.info(f"  Entry: {self.position['entry_price']:.2f}")
        logger.info(f"  Exit: {current_price:.2f}")
        logger.info(f"  Size: {self.position['size']:.2f} lots")
        logger.info(f"  PnL: ${pnl:.2f}")
        logger.info(f"  New Balance: ${self.balance:.2f}")
        logger.info(f"  Daily PnL: ${self.daily_pnl:.2f}")
        logger.info(f"  Trades today: {self.trades_today}")
        logger.info(f"  Consecutive losses: {self.consecutive_losses}")
        logger.info(f"  Exit reason: {exit_reason}")
        
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'position': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': current_price,
            'size': self.position['size'],
            'pnl': pnl,
            'exit_reason': exit_reason,
            'bars_held': self.position['bars_held'],
            'market_conditions': self.position['market_conditions'],
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'balance': self.balance
        })
        
        logger.info(f"{self.position['type']} {exit_reason} at {current_price}, PnL: ${pnl:.2f}")
        self.position = None
    
    def run_backtest(self, days=30):
        """Run backtest simulation"""
        try:
            # Get historical data with extra buffer for technical indicators
            df = self.get_historical_data(days + 10)  # Add 10 days buffer for technical indicators
            if df.empty:
                raise Exception("No historical data retrieved")
            
            logger.info(f"Running backtest on {len(df)} data points")
            logger.info(f"Data range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            logger.info(f"Data columns: {df.columns.tolist()}")
            
            # Add technical indicators to the entire dataset first
            df = self.data_processor.add_technical_indicators(df)
            
            # Verify all required features are present
            missing_features = [col for col in self.model.feature_columns if col not in df.columns]
            if missing_features:
                logger.error(f"Missing features in data: {missing_features}")
                logger.error(f"Available features: {df.columns.tolist()}")
                logger.error(f"Model features: {self.model.feature_columns}")
                raise ValueError(f"Missing features in data: {missing_features}")
            
            # Log feature information
            logger.info(f"Using features: {self.model.feature_columns}")
            logger.info(f"Number of features: {len(self.model.feature_columns)}")
            
            # Remove rows with NaN values (from technical indicator calculation)
            df = df.dropna()
            
            if len(df) < self.data_processor.sequence_length:
                raise Exception(f"Not enough data after technical indicator calculation. Required: {self.data_processor.sequence_length}, Available: {len(df)}")
            
            logger.info(f"Data points after technical indicators: {len(df)}")
            
            # Prepare data for prediction
            for i in range(self.data_processor.sequence_length, len(df)):
                # Get historical window for prediction
                historical_window = df.iloc[i-self.data_processor.sequence_length:i]
                current_price = df.iloc[i]['close']
                timestamp = df.iloc[i]['time']
                
                # Log the historical window
                logger.info(f"\nProcessing timestamp: {timestamp}")
                logger.info(f"Historical window shape: {historical_window.shape}")
                
                try:
                    # Ensure features are in the same order as during training
                    feature_data = historical_window[self.model.feature_columns].copy()
                    
                    # Create a DataFrame with proper feature names
                    feature_df = pd.DataFrame(
                        feature_data.values,
                        columns=self.model.feature_columns,
                        index=feature_data.index
                    )
                    
                    # Prepare data for prediction using model's feature columns
                    X = self.data_processor.prepare_prediction_data(feature_df)
                    logger.info(f"Prepared data shape: {X.shape}")
                    logger.info(f"Feature names: {feature_df.columns.tolist()}")
                    
                    # Make prediction (regression: next-period return, scaled)
                    predicted_return = self.model.predict(X)[0][0]
                    predicted_return = predicted_return / 1000  # Scale back to original
                    predicted_price = current_price * (1 + predicted_return)
                    
                    # Calculate price difference percentage
                    price_diff_percent = (predicted_price - current_price) / current_price
                    logger.info(f"Price difference: {price_diff_percent:.4%}")
                    
                    # Increment bars_held for open position
                    if self.position is not None:
                        self.position['bars_held'] += 1
                    
                    # Check exit conditions first
                    self.check_exit_conditions(predicted_price, current_price, timestamp)
                    
                    # Then check for new entries
                    if self.position is None:
                        self.execute_trade(predicted_price, current_price, timestamp)
                        
                    self.prediction_log.append(predicted_return)
                    
                except Exception as e:
                    logger.error(f"Error processing data point: {e}")
                    logger.error(f"Historical window data:\n{historical_window.tail().to_string()}")
                    continue
            
            # Calculate performance metrics
            self.calculate_performance_metrics()
            
            if self.prediction_log:
                logger.info(f"Prediction stats: min={np.min(self.prediction_log)}, max={np.max(self.prediction_log)}, mean={np.mean(self.prediction_log)}, median={np.median(self.prediction_log)}")
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise  # Re-raise the exception to see the full traceback
        finally:
            self.mt5.disconnect()
    
    def calculate_performance_metrics(self):
        """Calculate and log performance metrics"""
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate BUY and SELL trade metrics
        buy_trades = trades_df[trades_df['position'] == 'BUY']
        sell_trades = trades_df[trades_df['position'] == 'SELL']
        
        buy_trades_count = len(buy_trades)
        sell_trades_count = len(sell_trades)
        
        buy_winning = len(buy_trades[buy_trades['pnl'] > 0])
        sell_winning = len(sell_trades[sell_trades['pnl'] > 0])
        
        buy_win_rate = buy_winning / buy_trades_count if buy_trades_count > 0 else 0
        sell_win_rate = sell_winning / sell_trades_count if sell_trades_count > 0 else 0
        
        buy_avg_pnl = buy_trades['pnl'].mean() if buy_trades_count > 0 else 0
        sell_avg_pnl = sell_trades['pnl'].mean() if sell_trades_count > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Log results
        logger.info("\nBacktest Results:")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance: ${self.balance:,.2f}")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Total Return: {(self.balance/self.initial_balance - 1)*100:.2f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Average Win: ${avg_win:,.2f}")
        logger.info(f"Average Loss: ${avg_loss:,.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Log BUY and SELL trade statistics
        logger.info("\nTrade Direction Analysis:")
        logger.info(f"BUY Trades:")
        logger.info(f"  Count: {buy_trades_count}")
        logger.info(f"  Win Rate: {buy_win_rate*100:.2f}%")
        logger.info(f"  Average PnL: ${buy_avg_pnl:,.2f}")
        logger.info(f"  Winning Trades: {buy_winning}")
        logger.info(f"  Losing Trades: {buy_trades_count - buy_winning}")
        
        logger.info(f"\nSELL Trades:")
        logger.info(f"  Count: {sell_trades_count}")
        logger.info(f"  Win Rate: {sell_win_rate*100:.2f}%")
        logger.info(f"  Average PnL: ${sell_avg_pnl:,.2f}")
        logger.info(f"  Winning Trades: {sell_winning}")
        logger.info(f"  Losing Trades: {sell_trades_count - sell_winning}")
        
        # Save results to CSV
        if os.path.exists('backtest_results.csv'):
            os.remove('backtest_results.csv')
        trades_df.to_csv('backtest_results.csv', index=False)
        logger.info("\nDetailed trade results saved to backtest_results.csv")

    def prepare_prediction_data(self, data):
        """Prepare data for prediction with proper feature names"""
        try:
            # Ensure we have a DataFrame with proper feature names
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=self.model.feature_columns)
            
            # Verify all required features are present
            missing_features = [col for col in self.model.feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features in data: {missing_features}")
            
            # Select features in the same order as during training
            feature_data = data[self.model.feature_columns].copy()
            
            # Reshape for LSTM input (samples, time steps, features)
            X = feature_data.values.reshape(1, feature_data.shape[0], feature_data.shape[1])
            
            # Scale the data
            if self.data_processor.scaler is not None:
                # Reshape for scaling
                X_reshaped = X.reshape(-1, X.shape[-1])
                # Scale
                X_scaled = self.data_processor.scaler.transform(X_reshaped)
                # Reshape back
                X = X_scaled.reshape(X.shape)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in prepare_prediction_data: {str(e)}")
            raise

def main():
    """Main function to run backtest"""
    # Find latest model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        logger.error("No trained model found. Please train the model first.")
        return
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    # Run backtest
    backtester = Backtester(model_path=model_path)
    backtester.run_backtest(days=2)  # Test last 30 days

if __name__ == "__main__":
    main()
