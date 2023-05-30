import numpy as np
import value_generators as vg


def kalman_filter(data: np.ndarray, order: int = 1) -> np.ndarray:
  Q = 1e-5  # process variance
  R = 0.1**2  # estimate of measurement variance
  xhat = np.zeros(len(data))  # a posteri estimate of x
  # xhat = data.copy()  # a posteri estimate of x
  P = np.zeros(len(data))  # a posteri error estimate
  xhatminus = np.zeros(len(data))  # a priori estimate of x
  Pminus = np.zeros(len(data))  # a priori error estimate
  K = np.zeros(len(data))  # gain or blending factor
  for k in range(order, len(data)):
    # time update
    xhatminus[k] = np.mean(data[k-order:k])
    Pminus[k] = P[k-1] + Q
    # measurement update
    K[k] = Pminus[k] / (Pminus[k] + R)
    xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]
  return xhat


def reject_outliers(data: np.ndarray, m: int = 2) -> np.ndarray:
  data = data.copy()
  mean = np.mean(data)
  std = np.std(data)
  
  # Identify the outliers based on the criterion
  mask = np.abs(data - mean) >= m * std
  
  # Replace the outliers with the mean of the non-outlier values
  # data[mask] = np.mean(data[~mask])
  flattened = kalman_filter(data, order=2)
  data[mask] = flattened[mask]
  
  return data


def detect_anomalies_aging(data: np.ndarray, threshold: int = 3, aging_factor: float = 0.01) -> np.ndarray:
  info = np.zeros_like(data)
  output = np.zeros_like(data)
  alpha = 1 - aging_factor
  info[0] = np.abs(data[0])

  for i in range(1, len(data)):
    info[i] = alpha * info[i - 1] + (1 - alpha) * np.abs(data[i])
    deviation = np.abs(data[i] - output[i - 1])
    if deviation > threshold * info[i]:
      output[i] = alpha * output[i - 1] + (1 - alpha) * data[i]
    else:
      output[i] = data[i]

  return output


def show_stats_kalman(values: np.ndarray, title: str, apply_kalman: bool = False):
  print(title + ':')
  kalman = lambda data: kalman_filter(data, order=2)
  values = kalman(values) if apply_kalman else values
  stats = values - kalman(values)
  vg.print_value_params(stats)
  vg.plot_values(stats, title)


def analyze_purified_vales(array_size: int):
  normal, square = vg.mk_initial_data(array_size)
  composition = normal + square

  print()
  show_stats_kalman(composition, apply_kalman=True, title='Kalman set & normal distribution noise')

  abnormal_percentage = .1
  quality_coef = 5
  abnormal_size = int(len(composition) * abnormal_percentage)

  ab = vg.mk_abnormal_set(square, composition, abnormal_size, quality_coef)
  title = 'Square trend & normal distribution with abnormal error'
  print()
  vg.plot_sales_dynamics(square, ab, title)
  show_stats_kalman(composition, apply_kalman=True, title=title)

  flattened = detect_anomalies_aging(reject_outliers(ab, m=3))
  title = 'Square trend & normal distribution cleaned with 2-nd order Kalman'
  print()
  vg.plot_sales_dynamics(square, flattened, title)
  show_stats_kalman(flattened, apply_kalman=False, title=title)
  print()
  show_stats_kalman(flattened, apply_kalman=True, title=title + ' (Kalman applied)')
