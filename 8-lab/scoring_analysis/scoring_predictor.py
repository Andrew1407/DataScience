import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def build_model(x: np.ndarray, y: np.ndarray, model_path: str, epochs: int = 10, learning_rate: float = 1e-3) -> tuple[tf.keras.Sequential, float]:
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  _, inputs_num = x.shape
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(inputs_num,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test))
  mse = model.evaluate(x_test, y_test)
  model.save(filepath=model_path)
  return model, mse


def load_model(model_path: str) -> tf.keras.Sequential:
  return tf.keras.models.load_model(model_path)


def repeat_data(x, y, repeat: int = 1):
  repeat_data = lambda a: np.repeat(a, repeat, axis=0)
  return repeat_data(x), repeat_data(y)


def predict(model: tf.keras.Sequential, data: np.ndarray, validation_criteria: float) -> tuple[np.ndarray, np.ndarray]:
  predicted: np.ndarray = model.predict(data).flatten()
  predicted[predicted < 0] = 0
  valid = predicted > validation_criteria
  return predicted, valid.astype(np.int32)


def predict_single(model: tf.keras.Sequential, data: np.ndarray, validation_criteria: float) -> tuple[float, bool]:
  nested = np.expand_dims(data, axis=0)
  predicted, valid = predict(model, nested, validation_criteria)
  return predicted[0], valid[0]
