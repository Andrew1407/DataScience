import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataset as dt


def create_model(dataset: pd.DataFrame, model_path: str, target: str) -> tuple[tf.keras.Sequential, float]:
  x = dt.get_date_features(dataset)
  y = dataset[target].values.reshape(-1, 1)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  _, inputs_num = x.shape
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(inputs_num,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
  ])
  model.compile(optimizer='adam', loss='mse')
  history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
  mse = model.evaluate(x_test, y_test)
  model.save(filepath=model_path)
  return model, mse


def load_model(model_path: str) -> tf.keras.Sequential:
  return tf.keras.models.load_model(model_path)


def predict(model: tf.keras.Sequential, dates: np.ndarray) -> pd.DataFrame:
  table = pd.DataFrame(index=dates)
  dt.split_by_featurtes(table)
  inputs = dt.get_date_features(table)
  table['predicted'] = model.predict(inputs).flatten()
  return table
