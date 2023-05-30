from typing import Iterable
import math
import numpy as np


Result = tuple[np.ndarray, np.ndarray]
Analyzed = tuple[Result, Result]

def extrapolate(sample: Iterable[float|int], predict: float) -> Result:
  num_samples = len(sample)
  num_predictions = math.ceil(num_samples * predict)
  extrapolated = np.zeros((num_samples + num_predictions, 1))
  input_values = np.array(sample).reshape(-1, 1)
  input_matrix = np.column_stack((np.ones(num_samples), np.arange(num_samples), np.arange(num_samples) ** 2))

  # Matrix computations for linear regression
  input_transpose = input_matrix.T
  inner_matrix = input_transpose.dot(input_matrix)
  inverse_matrix = np.linalg.inv(inner_matrix)
  inverse_and_transpose = inverse_matrix.dot(input_transpose)
  coefficients = inverse_and_transpose.dot(input_values)

  # Compute the extrapolated values
  extrapolated[:, 0] = coefficients[0, 0] + coefficients[1, 0] * np.arange(num_samples + num_predictions) + \
    (coefficients[2, 0] * np.arange(num_samples + num_predictions) ** 2)
  return extrapolated.flatten(), coefficients


def flatten(sample: Iterable[float|int], flat: bool = True) -> Result:
  iter = len(sample)
  Yin = np.zeros((iter, 1))
  F = np.ones((iter, 5))
  for i in range(iter):  # формування структури вхідних матриць МНК
    Yin[i, 0] = float(sample[i])  # формування матриці вхідних даних
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)
    F[i, 3] = float(i * i * i)
    F[i, 4] = float(i * i * i * i)
  FT=F.T
  FFT = FT.dot(F)
  FFTI=np.linalg.inv(FFT)
  FFTIFT=FFTI.dot(FT)
  C=FFTIFT.dot(Yin)
  Yout=F.dot(C)
  return Yout.flatten() if flat else Yout, C


def analyze(sample: Iterable[float|int], predict: float) -> Analyzed:
  return flatten(sample, flat=True), extrapolate(sample, predict)


def show_regression_model(coefficients: np.ndarray):
  print('Regression model:')
  print(f'y(t) = {coefficients[0, 0]} + {coefficients[1, 0]} * t + {coefficients[2, 0]} * t ^ 2')
