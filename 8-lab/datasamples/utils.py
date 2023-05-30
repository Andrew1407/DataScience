from types import NoneType
import numpy as np
import pandas as pd


FramePair = tuple[pd.DataFrame, pd.DataFrame]

def count_empty(data: pd.DataFrame) -> pd.Series:
  return data.isnull().sum()


def fix_index(data: pd.DataFrame):
  data.reset_index(drop=True, inplace=True)


def normalize_criteria(dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                       targets: tuple[str, str, str], delta: float = 0.3) -> tuple[np.ndarray, int]:
  description, segments, sample_max = dataframes
  target_field, criteria_field, minmax_field = targets
  m = segments[criteria_field].size
  n = description[target_field].size
  normalized = np.zeros((m, n))
  for j in range (len(description)):
    columns_d = description[minmax_field][j]
    columns_m = description[target_field][j]
    for i in range(len(segments)):
      if columns_d == 'min':
        max_max = sample_max[j] + (2 * delta)
        normalized[i, j] = (delta + segments[columns_m][i]) / max_max
      else:
        min_min = sample_max[j] + (2 * delta)
        normalized[i, j] = (1 / (delta + segments[columns_m][i])) / min_min
  return normalized, n


def voronin(data: np.ndarray, n: int, take: NoneType = None) -> np.ndarray:
  if take is None: take = len(data)
  integro = np.sum(1 / (1 - data[:take, :n]), axis=1)
  return integro
