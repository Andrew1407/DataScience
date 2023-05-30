import os.path
import requests
import numpy as np
import pandas as pd
from zipfile import ZipFile
import io


def fetch_dataset(url: str, filename: str) -> pd.DataFrame:
  if not os.path.exists(filename):
    response = requests.get(url)
    with ZipFile(io.BytesIO(response.content)) as archive:
      archive.extract(filename)
  return pd.read_csv(filename)


def format_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
  formatted = pd.DataFrame()
  formatted['ap'] = dataset['p (mbar)']
  formatted['vp'] = dataset['VPdef (mbar)']
  formatted.index = pd.to_datetime(dataset['Date Time'], format='%d.%m.%Y %H:%M:%S')
  return formatted


def split_by_featurtes(dataset: pd.DataFrame):
  dataset['month'] = dataset.index.month
  dataset['day_of_month'] = dataset.index.day
  dataset['day_of_week'] = dataset.index.dayofweek
  dataset['hour'] = dataset.index.hour
  dataset['minute'] = dataset.index.minute
  dataset['day_of_year'] = dataset.index.dayofyear


def get_dataset(url: str, filename: str) -> pd.DataFrame:
  dataset = format_dataset(fetch_dataset(url, filename))
  split_by_featurtes(dataset)
  return dataset


def get_date_features(dataset: pd.DataFrame) -> np.ndarray:
  features = ['month', 'day_of_month', 'day_of_week', 'hour', 'minute', 'day_of_year']
  return dataset[features].values


def calc_metrics(data: np.ndarray) -> tuple[float, float, float]:
  median = np.median(data)
  variance = np.var(data)
  dispersion = np.sqrt(variance)
  return median, variance, dispersion
