from typing import Sequence, Callable
import pandas as pd
import mnk
import dataset as ds
import plots


def get_metrics(data: Sequence[float]) -> tuple[float, float, float]:
  metrics = ds.calc_metrics(data)
  median, variance, dispersion = metrics
  print(f'{median = }')
  print(f'{variance = }')
  print(f'{dispersion = }')
  return metrics


def make_metrics_loggers(analyzed: mnk.Analyzed) -> tuple[Callable, Callable]:
  flattening, extrapolation = analyzed
  def mk_logger(option: str, params: mnk.Result) -> callable:
    def logger():
      flattened, coefs = params
      print(f'\nMNK {option} regression model:')
      mnk.show_regression_model(coefs)
      print(f'\nMNK {option} metrics:')
      get_metrics(flattened)
    return logger
  return mk_logger('flattening', flattening), mk_logger('extrapolation', extrapolation)


def show_full_dataset(dataset: pd.DataFrame, target: str):
  print(f'Full dataset (tagtet column: "{target}"):')
  print(dataset)
  values = dataset[target]
  print('\nFull dataset metrics:')
  get_metrics(values)
  plots.pair_metrics(values, title='Total cost per date', rolling_window=2)


def show_full_mnk_analysys(dataset: pd.DataFrame, target: str, predict: float):
  values = dataset[target]
  analyzed = mnk.analyze(values, predict)
  loggers = make_metrics_loggers(analyzed)
  (flattened, _), (extrapolated, _) = analyzed
  flattened = ds.extend_series(values, flattened)
  extrapolated = ds.extend_series(values, extrapolated)
  plots.pair_mnk(values, flattened, extrapolated, loggers)


def show_segment_generalization(dataset: pd.DataFrame, target: str, column: str, factor: str, predict: float):
  title = f'Segmentation - "{column}" by factor: {factor}'
  print(f'\nGeneralization {title}:')
  segments = plots.segments(dataset, column, target, formatter=lambda x: getattr(x, factor)())
  series = pd.Series(segments).sort_values()
  print(f'{series = }')
  plots.segment_forms(series, title)
  analyzed = mnk.analyze(series.values, predict)
  loggers = make_metrics_loggers(analyzed)
  (flattened, _), (extrapolated, _) = analyzed
  plots.pair_mnk(series.values, flattened, extrapolated, loggers)


def show_segmentation(dataset: pd.DataFrame, target: str, column: str, predict: float):
  title = f'Segmentation - "{column}"'
  print(f'\nParticle {title}:')
  segments = plots.segments(dataset, column, target)
  for field in segments:
    segment = segments[field]
    print(f'\nSegment "{field}":')
    print('Original sample metrics:')
    get_metrics(segment)
    plots.pair_metrics(segment, title=f'Segment - "{field}"', rolling_window=2)
    analyzed = mnk.analyze(segment.values, predict)
    loggers = make_metrics_loggers(analyzed)
    (flattened, _), (extrapolated, _) = analyzed
    plots.pair_mnk(segment.values, flattened, extrapolated, loggers)
