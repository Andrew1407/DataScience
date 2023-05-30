import math
import numpy as np
import matplotlib.pyplot as plt


LabeledArrays = dict[str, np.ndarray]

mk_normal_dist = lambda n: np.random.normal(loc=0, scale=5, size=n)
mk_exponential_dist = lambda n: np.random.exponential(scale=3, size=n)

mk_square_trend = lambda n: np.array([pow(i, 2) for i in range(n)])
mk_constant_trend = lambda n, constant: np.ones(n) * constant

def calc_metrics(distribution: np.ndarray) -> tuple[float, float, float]:
  median = np.median(distribution)
  variance = np.var(distribution)
  dispersion = np.sqrt(variance)
  return median, variance, dispersion


def print_value_params(values: np.ndarray):
  median, variance, dispersion = calc_metrics(values)
  print(f'{values =}\n{median = }\n{variance = }\n{dispersion = }')


def plot_values(values: np.ndarray, title: str):
  plt.hist(values, bins=20, facecolor="blue", alpha=0.5)
  plt.title(title)
  plt.show()


def plot_sales_dynamics(trend: np.ndarray, composition: np.ndarray, title: str):
  plt.plot(composition, label='composition')
  plt.plot(trend, label='trend')
  plt.ylabel('Sales dynamics')
  plt.legend()
  plt.title(title)
  plt.show()


def plot_static_metrics(values: LabeledArrays):
  for label, value in values.items():
    plt.hist(value, bins=20, alpha=0.5, label=label)  
  plt.legend()
  plt.title(', '.join(values.keys()))
  plt.show()


def show_statistics(values: LabeledArrays):
  model = None
  trend = None

  for k, v in values.items():
    if 'distribution' in k: model = k, v
    elif 'trend' in k: trend = k, v
  if not (model and trend): return

  model_label, model_values = model
  trend_label, trend_values = trend
  composition_offset = model_values + trend_values

  print(model_label, 'values:')
  print_value_params(model_values)
  plot_values(model_values, model_label)

  composition_label = f'{model_label} values with {trend_label} offset'
  print(f'\n{composition_label}:')
  print_value_params(composition_offset)
  plot_sales_dynamics(trend_values, composition_offset, composition_label)

  plot_static_metrics({
    model_label: model_values,
    trend_label: trend_values,
    'offset': composition_offset,
    'controlled value': composition_offset - trend_values,
  })


def mk_distribution_sets(array_size: int) -> tuple[LabeledArrays]:
  normal_label = 'Normal distribution'
  normal = mk_normal_dist(array_size)

  exponential_label = 'Exponential distribution'
  offset = 0
  exponential = mk_exponential_dist(array_size) + offset
  return (
    { normal_label: normal },
    { exponential_label: exponential },
  )


def mk_trend_sets(array_size: int) -> tuple[LabeledArrays]:
  square_label = 'Square trend'
  trend_scale = 5e-3
  square_trend = mk_square_trend(array_size) * trend_scale

  constant = .5
  constant_label = f'Constant ({constant}) trend'
  constant_trend = mk_constant_trend(array_size, constant)

  return (
    { square_label: square_trend },
    { constant_label: constant_trend },
  )


def mk_initial_data(array_size: int) -> tuple[np.ndarray, np.ndarray]:
  normal, _ = mk_distribution_sets(array_size)
  square, _ = mk_trend_sets(array_size)
  data_to_analyze = normal | square
  show_statistics(data_to_analyze)
  return tuple(tuple(s.values())[0] for s in (normal, square))


def mk_abnormal_set(values_set: np.ndarray, composition: np.ndarray, abnormal_size: int, quality_coef: float) -> np.ndarray:
  random_errors = np.random.normal(0, quality_coef * 5, abnormal_size)  # аномальна випадкова похибка з нормальним законом
  values_count = len(values_set)
  abnormal = composition.copy()
  for i in range(abnormal_size):
    k = math.ceil(np.random.randint(1, values_count))
    abnormal[k] = values_set[k] + random_errors[i]        # аномальні вимірів з рівномірно розподіленими номерами
  return abnormal 
