from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset as ds


def extend_dataframe(real: pd.DataFrame, extrapolated: np.ndarray) -> pd.DataFrame:
  num_real = len(real)
  time_step = real.index[1] - real.index[0]
  indices_to_add = np.array([real.index[-1] + i * time_step for i in range(len(extrapolated) - num_real)])
  indices_to_add += time_step
  target = 'extrapolated'
  extended = pd.concat((real, pd.DataFrame(index=indices_to_add)))
  extended[target] = extrapolated
  return extended[[target]]


def get_metrics(data: Sequence[float]) -> tuple[float, float, float]:
  metrics = ds.calc_metrics(data)
  median, variance, dispersion = metrics
  print(f'{median = }')
  print(f'{variance = }')
  print(f'{dispersion = }')
  return metrics


def show_metrics_extended(collection: Sequence, pivot: int, label: str):
  print(f'\nMertics for the {label} (full):')
  get_metrics(collection)

  print(f'\nMertics for the {label} (original):')
  get_metrics(collection[:pivot])

  print(f'\nMertics for the {label} (predicted):')
  get_metrics(collection[pivot:])


def show_mnk_extrapolation(dataset: pd.DataFrame, extrapolation: pd.DataFrame, target: str):
  plt.ylabel('pessure')
  plt.xlabel('date')
  plt.plot(dataset.index, dataset[target], label='origimal sample', color='black')
  plt.plot(extrapolation.index, extrapolation['extrapolated'], label='extrapolated', color='red')
  plt.plot(extrapolation.index, [extrapolation['extrapolated'].mean()] * len(extrapolation.index), label='mean', color='yellowgreen')
  plt.xticks(rotation=45)
  plt.title('MNK extrapolation')
  plt.legend()
  plt.show()


def show_rolling_std(data: pd.Series, index: pd.Index, title: str, label: str):
  rolling = data.rolling(window=10)
  plt.ylabel('pessure')
  plt.xlabel('date')
  plt.plot(index, rolling.std(), label=label)
  plt.xticks(rotation=45)
  plt.title(title)
  plt.legend()
  plt.show()


def show_nn_predictions(dataset: pd.DataFrame, predicted: pd.DataFrame, target: str, full: bool = True):
  label = 'predicted mean'
  title = 'NN prediction'
  if not full:
    predicted = predicted[len(dataset):]
    label += ' (further)'
    title += ' (further)'
  predictions = predicted['predicted']

  plt.ylabel('pessure')
  plt.xlabel('date')
  plt.plot(dataset.index, dataset[target], label='original sample')
  plt.plot(predicted.index, predictions, label='predicted')
  plt.plot(predicted.index, [predictions.mean()] * len(predictions), label=label)
  if full:
    plt.plot(dataset.index, [dataset[target].mean()] * len(dataset.index), label='original mean', color='black')
  else:
    plt.plot(dataset.index, [dataset[target].mean()] * len(dataset.index), label='original mean')
  plt.xticks(rotation=45)
  plt.title(title)
  plt.legend()
  plt.show()
