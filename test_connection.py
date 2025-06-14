import MetaTrader5 as mt5
import os
from set_env import set_mt5_env
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_mt5_connection():
    """Test MT5 connection and list available symbols"""
    try:
        # Set environment variables
        set_mt5_env()
        
        # Get credentials
        login = os.getenv('MT5_LOGIN')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        path = os.getenv('MT5_PATH')
        
        logger.info("Connecting to MT5...")
        logger.info(f"Login: {login}")
        logger.info(f"Server: {server}")
        logger.info(f"Path: {path}")
        
        # Initialize MT5
        if not mt5.initialize(
            path=path,
            login=int(login),
            password=password,
            server=server
        ):
            error = mt5.last_error()
            logger.error(f"Failed to initialize MT5: {error}")
            return
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.error("Failed to get terminal info")
            return
        
        logger.info("\nTerminal Info:")
        logger.info(f"Connected: {terminal_info.connected}")
        logger.info(f"Trade Allowed: {terminal_info.trade_allowed}")
        logger.info(f"Terminal Path: {terminal_info.path}")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return
        
        logger.info("\nAccount Info:")
        logger.info(f"Balance: {account_info.balance}")
        logger.info(f"Equity: {account_info.equity}")
        logger.info(f"Margin: {account_info.margin}")
        logger.info(f"Free Margin: {account_info.margin_free}")
        logger.info(f"Leverage: {account_info.leverage}")
        logger.info(f"Currency: {account_info.currency}")
        logger.info(f"Company: {account_info.company}")
        
        # Get available symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error("Failed to get symbols list")
            return
        
        logger.info("\nAvailable Symbols:")
        for symbol in symbols:
            logger.info(f"- {symbol.name}")
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol.name)
            if symbol_info is not None:
                logger.info(f"  Bid: {symbol_info.bid}")
                logger.info(f"  Ask: {symbol_info.ask}")
                logger.info(f"  Volume Min: {symbol_info.volume_min}")
                logger.info(f"  Volume Max: {symbol_info.volume_max}")
                logger.info(f"  Trade Mode: {symbol_info.trade_mode}")
                logger.info(f"  Trade Contract Size: {symbol_info.trade_contract_size}")
                logger.info("  ---")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

if __name__ == "__main__":
    test_mt5_connection() 