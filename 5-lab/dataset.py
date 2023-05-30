import os.path
from dataclasses import dataclass
import requests
import numpy as np
import pandas as pd


@dataclass
class Dataset:
  country_region: str
  date_barrier: pd.Timestamp

  data: pd.DataFrame
  train: pd.DataFrame
  test: pd.DataFrame

  x_train: pd.DataFrame
  y_train: pd.DataFrame
  x_test: pd.DataFrame
  y_test: pd.DataFrame


def make_dataset(data: pd.DataFrame, date_split: str, country_region: str) -> Dataset:
  barrier = pd.to_datetime(date_split, format='%Y-%m-%d')
  train = data[data.index <= barrier]
  test = data[data.index > barrier]
  features = ['month', 'year', 'dayofyear', 'dayofweek', 'quarter']
  return Dataset(country_region=country_region, date_barrier=barrier,
                 data=data,
                 train=train,
                 test=test,
                 x_train = train[features],
                 y_train = train['new_cases'],
                 x_test = test[features],
                 y_test = test['new_cases'])


def fetch_dataset(url: str, path: str) -> pd.DataFrame:
  if not os.path.exists(path):
    response = requests.get(url)
    with open(path, 'w') as f: f.write(response.text)
  return pd.read_csv(path)


def format_dataset(data: pd.DataFrame, country_region: str) -> pd.DataFrame:
  formatted = data[data['location'] == country_region]
  formatted = formatted[['date', 'total_cases', 'new_cases']]
  # formatted.fillna(0, inplace=True)
  formatted = formatted.dropna()
  formatted.index = pd.to_datetime(formatted['date'], format='%Y-%m-%d')
  del formatted['date']

  formatted['month'] = formatted.index.month
  formatted['year'] = formatted.index.year
  formatted['dayofyear'] = formatted.index.dayofyear
  formatted['dayofweek'] = formatted.index.dayofweek
  formatted['quarter'] = formatted.index.quarter

  return formatted


def calc_metrics(data: np.ndarray) -> tuple[float, float, float]:
  median = np.median(data)
  variance = np.var(data)
  dispersion = np.sqrt(variance)
  return median, variance, dispersion
