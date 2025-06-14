import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LambdaCallback
import numpy as np
import os
import logging
from datetime import datetime
from ..utils.config import (
    SEQUENCE_LENGTH, FEATURE_COLUMNS, TARGET, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, LSTM_UNITS, DROPOUT_RATE, MODEL_DIR,
    GPU_MEMORY_FRACTION, EARLY_STOPPING_PATIENCE,
    TENSORBOARD_UPDATE_FREQ, TENSORBOARD_PROFILE_BATCH,
    TENSORBOARD_HISTOGRAM_FREQ, TENSORBOARD_WRITE_GRAPH,
    TENSORBOARD_WRITE_IMAGES, TENSORBOARD_DIR,
    CHECKPOINT_SAVE_BEST_ONLY, CHECKPOINT_SAVE_WEIGHTS_ONLY,
    CHECKPOINT_MONITOR, CHECKPOINT_MODE
)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from typing import Tuple, List, Dict
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

logger = logging.getLogger(__name__)

class LSTMTradingModel:
    def __init__(self, sequence_length: int = 60):
        """Initialize the LSTM trading model with improved architecture"""
        logger.info("\nInitializing LSTMTradingModel:")
        logger.info(f"  Sequence length: {sequence_length}")
        
        self.sequence_length = sequence_length
        self.model = None
        self.feature_columns = None
        self.input_dim = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False
        
        # Log initial state
        logger.info("Initial model state:")
        logger.info(f"  Model object: {self.model}")
        logger.info(f"  Feature columns: {self.feature_columns}")
        logger.info(f"  Input dimension: {self.input_dim}")
        
        # Configure GPU memory growth
        try:
            gpus = tf.config.list_physical_devices('GPU')
            logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {str(e)}")
        
        logger.info(f"Model initialized: {self}")

    def __str__(self) -> str:
        """String representation of model state"""
        return (
            f"LSTMTradingModel(\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  model_built={self.model is not None},\n"
            f"  feature_columns={self.feature_columns},\n"
            f"  input_dim={self.input_dim}\n"
            f")"
        )

    def set_feature_columns(self, feature_columns: List[str]) -> None:
        """Set feature columns and build model"""
        logger.info("\nSetting feature columns:")
        logger.info(f"Current model state: {self}")
        
        if not feature_columns:
            raise ValueError("Feature columns cannot be empty")
            
        logger.info(f"  Number of features: {len(feature_columns)}")
        logger.info(f"  Features: {feature_columns}")
        
        self.feature_columns = feature_columns
        self.input_dim = len(feature_columns)
        
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False
        
        # Build model immediately after setting features
        logger.info("\nBuilding model after setting features...")
        self.build_model()
        
        logger.info(f"Model after setting features: {self}")

    def fit_scaler(self, data: pd.DataFrame):
        """Fit the scaler with the provided data"""
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Fit scaler only on feature columns
        self.scaler.fit(data[self.feature_columns])
        self.is_fitted = True
        logger.info("Scaler fitted successfully")

    def build_model(self) -> None:
        """Build the LSTM model with improved architecture"""
        logger.info("\nBuilding model:")
        logger.info(f"Current model state: {self}")
        
        if not self.feature_columns:
            raise ValueError("Feature columns must be set before building model")
            
        logger.info(f"  Input dimension: {self.input_dim}")
        logger.info(f"  Sequence length: {self.sequence_length}")
        
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.input_dim))
            logger.info(f"  Input shape: {inputs.shape}")
            
            # First LSTM layer with residual connection
            x = LSTM(128, return_sequences=True, 
                    kernel_regularizer=l2(0.0001))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Residual connection
            residual = x
            
            # Second LSTM layer
            x = LSTM(128, return_sequences=False,
                    kernel_regularizer=l2(0.0001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Dense layers
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Output layer (linear activation for unbounded predictions)
            outputs = Dense(1, activation='linear')(x)
            
            # Create model
            self.model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model with custom loss
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss=self.custom_loss,
                metrics=['mse', 'mae']
            )
            
            # Log model summary
            logger.info("\nModel architecture:")
            self.model.summary(print_fn=logger.info)
            
            # Verify model is compiled
            if not self.model.optimizer:
                raise ValueError("Model was not compiled successfully")
                
            logger.info("\nModel compilation successful")
            logger.info(f"  Optimizer: {self.model.optimizer.__class__.__name__}")
            logger.info(f"  Learning rate: {self.model.optimizer.learning_rate.numpy()}")
            logger.info(f"  Loss function: {self.model.loss.__name__ if hasattr(self.model.loss, '__name__') else str(self.model.loss)}")
            logger.info(f"Final model state: {self}")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            logger.error(f"Model state at error: {self}")
            self.model = None
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the model with improved monitoring"""
        logger.info("\nStarting training:")
        logger.info(f"Current model state: {self}")
        
        if self.model is None:
            raise ValueError("Model must be built before training. Call build_model() or set_feature_columns() first")
            
        logger.info(f"  Training data shape: {X_train.shape}")
        logger.info(f"  Validation data shape: {X_val.shape}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Verify data shapes
        expected_shape = (self.sequence_length, self.input_dim)
        actual_shape = X_train.shape[1:]
        if actual_shape != expected_shape:
            raise ValueError(
                f"Training data shape mismatch.\n"
                f"Expected: (..., {expected_shape})\n"
                f"Got: {X_train.shape}\n"
                f"Model state: {self}"
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(MODEL_DIR, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Log training results
            logger.info("\nTraining completed:")
            logger.info(f"  Final training loss: {history.history['loss'][-1]:.6f}")
            logger.info(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
            logger.info(f"  Best validation loss: {min(history.history['val_loss']):.6f}")
            logger.info(f"  Epochs trained: {len(history.history['loss'])}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training with improved null handling"""
        logger.info("Preparing sequences for training...")
        
        # Verify all required features exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Check for null values and handle them
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in features:")
            for col in null_counts[null_counts > 0].index:
                logger.warning(f"{col}: {null_counts[col]} null values")
            
            # Handle nulls with forward fill first
            df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill')
            # Then backward fill any remaining nulls
            df[self.feature_columns] = df[self.feature_columns].fillna(method='bfill')
            # Finally, fill any still remaining nulls with 0
            df[self.feature_columns] = df[self.feature_columns].fillna(0)
            logger.info("Null values handled with forward fill, backward fill, and zero fill")
            
            # Verify no nulls remain
            remaining_nulls = df[self.feature_columns].isnull().sum()
            if remaining_nulls.any():
                raise ValueError(f"Null values remain after handling: {remaining_nulls[remaining_nulls > 0]}")
        
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
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared sequences shape: X: {X.shape}, y: {y.shape}")
        logger.info(f"X mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
        logger.info(f"y mean: {np.mean(y):.4f}, std: {np.std(y):.4f}")
        
        return X, y

    def save_model(self, model_path: str):
        """Save model and scaler"""
        if not self.model:
            raise ValueError("No model to save")
            
        # Save model
        self.model.save(model_path)
        
        # Save scaler using joblib for consistency
        scaler_path = model_path.replace('.h5', '_scaler.save')
        save_data = {
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns
        }
        joblib.dump(save_data, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path: str):
        """Load model and scaler"""
        # Load model
        self.model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': self.custom_loss})
        
        # Load scaler using joblib
        scaler_path = model_path.replace('.h5', '_scaler.save')
        try:
            loaded_data = joblib.load(scaler_path)
            if isinstance(loaded_data, dict):
                self.scaler = loaded_data.get('scaler', MinMaxScaler(feature_range=(-1, 1)))
                self.is_fitted = loaded_data.get('is_fitted', False)
                if 'feature_columns' in loaded_data:
                    self.feature_columns = loaded_data['feature_columns']
                    logger.info(f"Loaded feature columns: {self.feature_columns}")
            else:
                self.scaler = loaded_data
                self.is_fitted = True
                
            # Verify scaler is properly fitted
            if not hasattr(self.scaler, 'min_') or not hasattr(self.scaler, 'scale_'):
                logger.warning("Loaded scaler is not properly fitted")
                self.is_fitted = False
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                logger.info("Scaler loaded and verified as fitted")
                logger.info(f"Number of features in scaler: {len(self.scaler.min_)}")
                self.is_fitted = True  # Explicitly set is_fitted to True if scaler has required attributes
                
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.is_fitted = False
            logger.warning("Using new unfitted scaler due to loading error")
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
        logger.info(f"Model fitted status: {self.is_fitted}")
        
        # Set feature columns and input dimension from model
        self.input_dim = self.model.input_shape[-1]
        # Note: We need to set feature columns separately as they're not stored in the model
        if self.feature_columns is None:
            self.feature_columns = [f'feature_{i}' for i in range(self.model.input_shape[2])]
            logger.warning(f"Feature columns not set, using placeholder names: {len(self.feature_columns)} features")

    def custom_loss(self, y_true, y_pred):
        """Custom loss function as a class method for model loading"""
        # MSE loss
        mse_loss = K.mean(K.square(y_true - y_pred))
        
        # Directional accuracy loss
        direction_true = K.sign(y_true)
        direction_pred = K.sign(y_pred)
        direction_loss = K.mean(K.binary_crossentropy(
            (direction_true + 1) / 2,  # Convert to 0/1
            (direction_pred + 1) / 2
        ))
        
        # Combine losses with more weight on direction
        return mse_loss + 2.0 * direction_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with proper scaling"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale input features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Make prediction
        return self.model.predict(X_scaled, verbose=0)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise Exception("Model not trained yet")
        results = self.model.evaluate(X_test, y_test)
        metrics = ['loss', 'mae', 'mse', 'mape']
        for metric, value in zip(metrics, results):
            logger.info(f"Model evaluation - {metric}: {value:.4f}")
        return results
