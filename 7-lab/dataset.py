import numpy as np
import pandas as pd


def get_dataset(filename: str) -> pd.DataFrame:
  index_column = 'OrderDate'
  dataset = pd.read_excel(filename, parse_dates=[index_column], index_col=index_column)
  return dataset.sort_index()


def calc_metrics(data: np.ndarray) -> tuple[float, float, float]:
  median = np.median(data)
  variance = np.var(data)
  dispersion = np.sqrt(variance)
  return median, variance, dispersion


def extend_series(data: pd.Series, extrapolated: np.ndarray) -> pd.Series:
  new_count = len(extrapolated) - len(data)
  last_date = data.index[-1]
  date_step = last_date - data.index[-2]
  new_dates = np.array([data.index[-1] + i * date_step for i in range(new_count)])
  new_series = pd.Series(data=extrapolated[len(data):], index=new_dates)
  old_series = data.copy()
  old_series[:] = extrapolated[:len(data)]
  return pd.concat((old_series, new_series))
