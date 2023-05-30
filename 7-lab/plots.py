from typing import Collection, Callable
import pandas as pd
import matplotlib.pyplot as plt


def multiple(data: dict[str, Collection], title: str, y: str = None, legend: bool=True):
  for label in data: plt.plot(data[label], label=label)
  if y: plt.ylabel(y)
  if legend: plt.legend()
  plt.title(title)
  plt.show()


def hist(data: Collection, title: str):
  plt.title(title)
  plt.hist(data, bins=20, facecolor="blue", alpha=0.5)
  plt.show()


def segments(dataset: pd.DataFrame, column: str, target: str, formatter: Callable=None) -> dict:
  title = f'Segmentation - "{column}"'
  fields = set(dataset[column])
  segments = dict()
  print('Segments:', fields)

  for field in fields:
    values = dataset[dataset[column] == field][target]
    segments[field] = formatter(values) if callable(formatter) else values
    plt.plot(values.index, values, label=field)
  plt.ylabel('cost')
  plt.title(title)
  plt.legend()
  plt.show()

  return segments


def segment_forms(data: pd.Series, title: str):
  data.plot(kind='bar', color='b')
  plt.title(title)
  plt.show()

  data.plot()
  plt.title(title)
  plt.show()

  hist(data, title)


def pair(data: dict[str, Collection], hist_values: Collection, title: str):
  multiple(data, title=title, y='cost', legend=True)
  hist(hist_values, title=title + ' (distribution hist)')


def pair_metrics(values: pd.Series, title: str, rolling_window: int):
  mk_series = lambda x: pd.Series(data=x, index=values.index)
  rolling = values.rolling(window=rolling_window)
  pair(title=title, hist_values=values, data={
    'original sample': values,
    'rolling mean': mk_series(rolling.mean()),
    'rolling std': mk_series(rolling.std()),
    'mean': mk_series([values.mean()] * len(values)),
  })


def pair_mnk(values: pd.Series, flattened: Collection, extrapolated: Collection, loggers: tuple[Callable, Callable]=(None, None)):
  fl_logger, ex_logger = loggers
  
  if callable(fl_logger): fl_logger()
  pair(title='mnk flattening', hist_values=flattened, data={
    'original sample': values,
    'flattened': flattened,
  })

  if callable(ex_logger): ex_logger()
  pair(title='mnk extrapolation', hist_values=extrapolated, data={
    'original sample': values,
    'extrapolated': extrapolated,
  })
