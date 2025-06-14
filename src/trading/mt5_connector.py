import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import logging
from ..utils.config import SYMBOL, TIMEFRAME, START_DATE, END_DATE

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.connected = False
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

    def connect(self):
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                raise Exception(f"Failed to initialize MT5: {mt5.last_error()}")
            
            # Check if AutoTrading is enabled
            if not mt5.terminal_info().trade_allowed:
                raise Exception(
                    "AutoTrading is disabled in MT5. Please enable it by:\n"
                    "1. Opening MT5 terminal\n"
                    "2. Click 'Tools' -> 'Options'\n"
                    "3. Go to 'Expert Advisors' tab\n"
                    "4. Check 'Allow Automated Trading'\n"
                    "5. Click 'OK' and restart MT5"
                )
            
            self.connected = True
            logger.info("Connected to MT5 terminal")
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            raise

    def disconnect(self):
        """Close MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_historical_data(self, symbol=SYMBOL, timeframe=TIMEFRAME, 
                          start_date=START_DATE, end_date=END_DATE):
        """
        Fetch historical data from MT5
        """
        if not self.connected:
            self.connect()

        # Convert timeframe string to MT5 timeframe
        mt5_timeframe = self.timeframe_map.get(timeframe)
        if mt5_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Convert dates to UTC
        timezone = pytz.timezone("UTC")
        start_date = timezone.localize(start_date)
        end_date = timezone.localize(end_date)

        logger.info(f"Fetching historical data for {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Start date: {start_date}")
        logger.info(f"End date: {end_date}")

        # Verify symbol exists and is enabled
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            # Try with trailing dot
            symbol_with_dot = f"{symbol}."
            symbol_info = mt5.symbol_info(symbol_with_dot)
            if symbol_info is not None:
                symbol = symbol_with_dot
                logger.info(f"Using symbol with trailing dot: {symbol}")
            else:
                error = mt5.last_error()
                logger.error(f"Symbol {symbol} not found. Error: {error}")
                raise Exception(f"Symbol {symbol} not found. Please check the symbol name.")

        if not symbol_info.visible:
            logger.info(f"Enabling symbol {symbol} for trading")
            if not mt5.symbol_select(symbol, True):
                error = mt5.last_error()
                logger.error(f"Failed to enable {symbol} for trading: {error}")
                raise Exception(f"Failed to enable {symbol} for trading: {error}")

        # Get symbol properties
        logger.info(f"Symbol properties for {symbol}:")
        logger.info(f"  Bid: {symbol_info.bid}")
        logger.info(f"  Ask: {symbol_info.ask}")
        logger.info(f"  Volume Min: {symbol_info.volume_min}")
        logger.info(f"  Volume Max: {symbol_info.volume_max}")
        logger.info(f"  Trade Mode: {symbol_info.trade_mode}")
        logger.info(f"  Contract Size: {symbol_info.trade_contract_size}")

        # Fetch rates with error checking
        try:
            # Try with a smaller date range first
            test_end_date = min(end_date, datetime.now(timezone))
            test_start_date = test_end_date - timedelta(days=7)  # Try last 7 days first
            
            logger.info(f"Testing with smaller date range: {test_start_date} to {test_end_date}")
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, test_start_date, test_end_date)
            
            if rates is None:
                error = mt5.last_error()
                logger.error(f"Failed to get test historical data: {error}")
                raise Exception(f"Failed to get test historical data: {error}")
            
            # If test succeeds, try full range
            logger.info("Test successful, fetching full date range")
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None:
                error = mt5.last_error()
                logger.error(f"Failed to get historical data: {error}")
                raise Exception(f"Failed to get historical data: {error}")
                
        except Exception as e:
            logger.error(f"Error fetching rates: {str(e)}")
            raise

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        if df.empty:
            raise Exception(f"No historical data found for {symbol} between {start_date} and {end_date}")
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"Retrieved {len(df)} data points")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df

    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            self.connect()
        
        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            raise Exception(f"Failed to get account info: {error}")
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }

    def place_order(self, symbol, order_type, volume, price=None, stop_loss=None, take_profit=None):
        """Place a market order"""
        try:
            if not self.connected:
                self.connect()
            
            # Get current price if not provided
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    raise Exception(f"Failed to get price for {symbol}")
                price = tick.ask if order_type == "BUY" else tick.bid
            
            # Prepare the trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
            
            # Send the trading request
            result = mt5.order_send(request)
            
            if result is None:
                raise Exception("Order placement failed - no response from MT5")
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order placement failed - {result.comment}")
            
            logger.info(f"Order placed successfully: {order_type} {volume} lots at {price}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise 