# LLM-Trade: AI-Powered Trading System

An automated trading system that uses LSTM (Long Short-Term Memory) neural networks to predict market movements and execute trades through MetaTrader 5.

## Features

- LSTM-based price prediction model
- Real-time market data processing with technical indicators
- Dynamic position sizing based on market regime
- Risk management with stop-loss and take-profit
- Automated trading through MetaTrader 5
- GPU acceleration support
- Comprehensive logging and monitoring

## Requirements

- Python 3.8+
- MetaTrader 5
- TensorFlow 2.x
- pandas
- numpy
- ta (Technical Analysis library)
- scikit-learn
- matplotlib
- schedule

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-trade.git
cd llm-trade
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up MetaTrader 5:
   - Install MetaTrader 5
   - Create a `.env` file with your MT5 credentials:
     ```
     MT5_LOGIN=your_login
     MT5_PASSWORD=your_password
     MT5_SERVER=your_server
     ```

## Project Structure

```
llm-trade/
├── src/
│   ├── data/
│   │   └── data_processor.py    # Data processing and technical indicators
│   ├── models/
│   │   └── lstm_model.py        # LSTM model implementation
│   ├── trading/
│   │   ├── mt5_connector.py     # MetaTrader 5 connection handling
│   │   └── trading_agent.py     # Trading logic and execution
│   └── utils/
│       └── config.py            # Configuration settings
├── main.py                      # Main application entry point
├── train_model.py               # Model training script
├── backtest.py                  # Backtesting functionality
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Usage

1. Train the model:
```bash
python train_model.py
```

2. Run the trading system:
```bash
python main.py
```

## Configuration

Key parameters can be configured in `src/utils/config.py`:
- Trading symbol
- Timeframe
- Feature columns
- Model parameters
- Training settings

## Risk Warning

This trading system is for educational purposes only. Trading financial markets carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 