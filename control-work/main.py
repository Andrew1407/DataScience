import numpy as np
import matplotlib.pyplot as plt


ARRAY_SIZE = 10_000

LabeledArrays = dict[str, np.ndarray]

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


def mk_distribution_set() -> LabeledArrays:
  normal_label = 'Normal distribution'
  normal = np.random.normal(loc=0, scale=1, size=ARRAY_SIZE)
  return { normal_label: normal }


def mk_trend_set() -> LabeledArrays:
  periodic_label = 'Periodic trend (cos)'
  step = .1
  x_stretch = .02
  y_stretch = 1.5
  trend_lin = np.linspace(0, step * (ARRAY_SIZE - 1), ARRAY_SIZE) * x_stretch
  periodic_trend = np.cos(trend_lin) * y_stretch
  return { periodic_label: periodic_trend }


if __name__ == '__main__':
  normal_distribution = mk_distribution_set()
  periodic_trend = mk_trend_set()
  show_statistics(normal_distribution | periodic_trend)
