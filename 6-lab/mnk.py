from typing import Iterable
import math
import numpy as np


def extrapolate(sample: Iterable[float], predict: float) -> tuple[np.ndarray, np.ndarray]:
  num_samples = len(sample)
  num_predictions = math.ceil(num_samples * predict)
  extrapolated_values = np.zeros((num_samples + num_predictions, 1))
  input_values = np.array(sample).reshape(-1, 1)
  input_matrix = np.column_stack((np.ones(num_samples), np.arange(num_samples), np.arange(num_samples)**2))

  # Matrix computations for linear regression
  input_transpose = input_matrix.T
  inner_matrix = input_transpose.dot(input_matrix)
  inverse_matrix = np.linalg.inv(inner_matrix)
  inverse_and_transpose = inverse_matrix.dot(input_transpose)
  coefficients = inverse_and_transpose.dot(input_values)

  # Compute the extrapolated values
  extrapolated_values[:, 0] = coefficients[0, 0] + coefficients[1, 0] * np.arange(num_samples + num_predictions) + \
    (coefficients[2, 0] * np.arange(num_samples + num_predictions) ** 2)
  return extrapolated_values, coefficients


def show_regression_model(coefficients: np.ndarray):
  print('Regression model:')
  print('y(t) =', coefficients[0, 0], '+', coefficients[1, 0], '*t', '+', coefficients[2, 0], '*t^2')
