import os
import numpy as np
import tensorflow as tf
import scoring_analysis.scoring_predictor as sp
import scoring_analysis.anomalies as anomalies


DATA_INCREASE = 200
EPOCHS = 10
LEARNING_RATE = 1e-3


def get_model(model_path: str, data: tuple[np.ndarray, np.ndarray]) -> tf.keras.Sequential:
  if os.path.exists(model_path):
    return sp.load_model(model_path)
  x, y = sp.repeat_data(*data, DATA_INCREASE)
  model, mse = sp.build_model(x, y, model_path, EPOCHS, LEARNING_RATE)
  print('\nmse:', mse)
  return model


def flatten_data(data: np.ndarray) -> np.ndarray:
  rejected = anomalies.reject_outliers(data, m=2)
  flattened = anomalies.detect_anomalies_aging(rejected, threshold=1, aging_factor=1e-2)
  return flattened
