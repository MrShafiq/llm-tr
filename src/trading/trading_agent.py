import numpy as np
from datetime import datetime, timedelta
from ..utils.config import SYMBOL, TIMEFRAME
from .mt5_connector import MT5Connector
from ..models.lstm_model import LSTMTradingModel
from ..data.data_processor import DataProcessor
import joblib
import os
from ..utils.config import MODEL_DIR
import logging
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, model_path=None):
        self.mt5 = MT5Connector()
        self.model = LSTMTradingModel()
        
        # Initialize DataProcessor with the correct scaler path
        scaler_path = os.path.join(MODEL_DIR, 'scaler.save')
        self.data_processor = DataProcessor(scaler_path=scaler_path)
        
        if model_path:
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        self.position = None
        self.last_prediction = None
        self.last_prediction_time = None
        
        # If scaler is not fitted, fit it with historical data
        if not self.data_processor.is_fitted:
            logger.info("Scaler not fitted. Fitting with historical data...")
            historical_data = self.get_market_data(lookback_periods=1000)  # Get more data for better fitting
            self.data_processor.fit_scaler(historical_data)

    def get_market_data(self, lookback_periods=100):
        """Get recent market data for prediction"""
        end_date = datetime.now()
        # Use at least 24 hours of data for better prediction
        start_date = end_date - timedelta(hours=max(24, lookback_periods))
        return self.mt5.get_historical_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=start_date,
            end_date=end_date
        )

    def make_prediction(self):
        """Make a price prediction using the LSTM model"""
        # Get recent market data
        df = self.get_market_data()
        
        # Add technical indicators
        df = self.data_processor.add_technical_indicators(df)
        
        # Use only the last 60 time steps for prediction
        df = df.tail(60)
        
        # Prepare data for prediction
        X = self.data_processor.prepare_prediction_data(df)
        
        # Make prediction
        prediction = self.model.predict(X)
        
        # Store the scaled return for later use
        scaled_return = prediction[-1][0]
        self.last_prediction = scaled_return
        
        # Convert scaled return to predicted price using current price
        current_price = df.iloc[-1]['close']
        predicted_price = self.data_processor.inverse_transform(scaled_return, current_price)
        self.last_prediction_time = datetime.now()
        
        return predicted_price

    def calculate_position_size(self, price, current_data):
        """Calculate position size based on dynamic risk management (same as backtest)"""
        base_risk = 0.005  # 0.5% base risk
        regime = current_data['market_regime']
        if regime == 'strong_trend':
            risk_multiplier = 1.5
        elif regime == 'trending':
            risk_multiplier = 1.2
        elif regime == 'volatile':
            risk_multiplier = 0.5
        elif regime == 'ranging':
            risk_multiplier = 0.3
        else:
            risk_multiplier = 1.0
        trend_strength = current_data['trend_strength']
        if trend_strength > 2.0:
            risk_multiplier *= 1.2
        elif trend_strength < 0.5:
            risk_multiplier *= 0.8
        risk_per_trade = base_risk * risk_multiplier
        account_info = self.mt5.get_account_info()
        risk_amount = account_info['balance'] * risk_per_trade
        stop_loss_pips = 20
        pip_value = 0.1
        position_size = risk_amount / (stop_loss_pips * pip_value)
        position_size = round(position_size / 1000, 2)
        max_position = min(2.0, 5.0 / max(1e-6, current_data['volatility_ratio']))
        min_size = 0.01
        return max(min_size, min(position_size, max_position))

    def execute_trade(self, prediction, current_price):
        """Execute trading decision based on prediction"""
        try:
            if self.position is not None:
                logger.info("Already in a position, skipping trade execution")
                return
            
            # Calculate position size
            historical_data = self.get_market_data(lookback_periods=100)
            processed_data = self.data_processor.add_technical_indicators(historical_data)  # Add technical indicators
            current_data = processed_data.iloc[-1]
            position_size = self.calculate_position_size(current_price, current_data)
            
            if position_size <= 0:
                logger.warning("Invalid position size calculated, skipping trade")
                return
            
            # Determine trade direction with tighter thresholds
            price_diff_percent = (prediction - current_price) / current_price
            
            # Add trend confirmation
            historical_data = self.get_market_data(lookback_periods=20)
            sma_20 = historical_data['close'].rolling(window=20).mean().iloc[-1]
            trend = 'up' if current_price > sma_20 else 'down'

            if price_diff_percent > 0.0002 and trend == 'up':  # Removed invalid 'strong_trend' check, only use 'up' for BUY signals
                logger.info(f"Placing BUY order: prediction={prediction}, current={current_price}, diff={price_diff_percent:.4%}")
                # Calculate stop loss and take profit
                stop_loss = current_price * 0.997  # 0.3% stop loss
                take_profit = current_price * 1.006  # 0.6% take profit
                
                result = self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="BUY",
                    volume=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    entry_price = current_price
                    self.position = {
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'size': position_size,
                        'bars_held': 0
                    }
                    logger.info("BUY order executed successfully")
                else:
                    logger.error("Failed to execute BUY order")
                
            elif price_diff_percent < -0.001 and trend in ['down']:  # 0.1% threshold using valid trend states
                logger.info(f"Placing SELL order: prediction={prediction}, current={current_price}, diff={price_diff_percent:.4%}")
                # Calculate stop loss and take profit
                stop_loss = current_price * 1.003  # 0.3% stop loss
                take_profit = current_price * 0.994  # 0.6% take profit
                
                result = self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="SELL",
                    volume=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    entry_price = current_price
                    self.position = {
                        'type': 'SHORT',  # Changed from 'SELL' to 'SHORT'
                        'entry_price': entry_price,
                        'size': position_size,
                        'bars_held': 0
                    }
                    logger.info("SELL order executed successfully")
                else:
                    logger.error("Failed to execute SELL order")
            else:
                logger.info(f"No trade signal: prediction={prediction}, current={current_price}, diff={price_diff_percent:.4%}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.position = None  # Reset position on error

    def check_exit_conditions(self, current_data, current_price, timestamp):
        """Check if we should exit the current position (synchronized with backtest logic)"""
        if self.position is None:
            return
        # Require a minimum hold time of 5 bars
        min_bars_held = 5
        if self.position['bars_held'] < min_bars_held:
            return
        # Prepare data for prediction
        X = self.data_processor.prepare_prediction_data(current_data)
        predicted_return = self.model.predict(X)[0][0]
        predicted_price = current_price * (1 + predicted_return)
        # ATR-based stop loss/take profit
        atr = current_data['atr']
        if self.position['type'] == "LONG":
            if current_price <= self.position['entry_price'] - (atr * 3.0):
                logger.info(f"BUY stop loss hit at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="SELL",
                    volume=self.position['size']
                )
                self.position = None
            elif current_price >= self.position['entry_price'] + (atr * 3.0):
                logger.info(f"BUY take profit hit at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="SELL",
                    volume=self.position['size']
                )
                self.position = None
            elif (current_price < current_data['sma_20'] or
                  current_data['macd'] < current_data['macd_signal'] or
                  current_data['stoch_k'] < current_data['stoch_d'] or
                  current_price > current_data['bb_upper'] or
                  current_data['market_regime'] == 'volatile'):
                logger.info(f"BUY exit signal at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="SELL",
                    volume=self.position['size']
                )
                self.position = None
        elif self.position['type'] == "SHORT":
            if current_price >= self.position['entry_price'] + (atr * 3.0):
                logger.info(f"SELL stop loss hit at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="BUY",
                    volume=self.position['size']
                )
                self.position = None
            elif current_price <= self.position['entry_price'] - (atr * 3.0):
                logger.info(f"SELL take profit hit at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="BUY",
                    volume=self.position['size']
                )
                self.position = None
            elif (current_price > current_data['sma_20'] or
                  current_data['macd'] > current_data['macd_signal'] or
                  current_data['stoch_k'] > current_data['stoch_d'] or
                  current_price < current_data['bb_lower'] or
                  current_data['market_regime'] == 'volatile'):
                logger.info(f"SELL exit signal at {current_price}")
                self.mt5.place_order(
                    symbol=SYMBOL,
                    order_type="BUY",
                    volume=self.position['size']
                )
                self.position = None

    def run_trading_cycle(self):
        """Run one complete trading cycle (synchronized with backtest logic)"""
        try:
            # Get current market data
            current_data = self.get_market_data(lookback_periods=100)
            current_data = self.data_processor.add_technical_indicators(current_data)
            latest = current_data.iloc[-1]
            current_price = latest['close']
            timestamp = latest['time'] if 'time' in latest else datetime.now()

            # Track bars_held for open position
            if self.position is not None:
                self.position['bars_held'] += 1

            # Check exit conditions for existing position
            self.check_exit_conditions(current_data, current_price, timestamp)

            # If no position, look for new entry
            if self.position is None:
                predicted_price = self.make_prediction()
                self.execute_trade(predicted_price, current_price)
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.mt5.disconnect()
