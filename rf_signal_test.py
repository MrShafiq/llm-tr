import pandas as pd
import numpy as np
from src.data.data_processor import DataProcessor
from src.trading.mt5_connector import MT5Connector
from src.utils.config import SYMBOL, TIMEFRAME, START_DATE, END_DATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info('Fetching historical data...')
    mt5 = MT5Connector()
    df = mt5.get_historical_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE
    )
    if df.empty:
        logger.error('No historical data retrieved.')
        return
    logger.info(f'Retrieved {len(df)} data points')

    # Add technical indicators
    data_processor = DataProcessor()
    df = data_processor.add_technical_indicators(df)

    # Prepare features and labels for classification
    X = df[data_processor.feature_columns].iloc[60:-4].values  # skip initial NaNs, adjust for 4-bar lookahead
    # Binary label: 1 if close in 4 bars < current close by more than 0.2%, else 0
    threshold = -0.002  # -0.2%
    future_return = (df['close'].shift(-4).iloc[60:-4] - df['close'].iloc[60:-4]) / df['close'].iloc[60:-4]
    y = (future_return < threshold).astype(int).values

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train RandomForest with class_weight balanced
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info(f'RandomForest accuracy: {acc:.4f}')
    logger.info('Classification report:')
    print(classification_report(y_test, y_pred))

    # Feature importances
    importances = clf.feature_importances_
    feature_names = data_processor.feature_columns
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print('Top 10 feature importances:')
    print(imp_df.head(10))

if __name__ == '__main__':
    main() 