from typing import Callable
import numpy as np
import pandas as pd
from datasamples.utils import count_empty, FramePair


def get_datasets_logger(fn: Callable) -> Callable[..., FramePair]:
  ResultType = FramePair
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    data_sample, data_description = result
    print('data_sample:\n', data_sample)
    print('\ndata_sample coliumns:', data_sample.columns)
    print('\nempty spaces count:\n', count_empty(data_sample))
    print('\ndata_description:\n', data_description)
    return result
  return wrapper


def make_segment_data_logger(fn: Callable) -> Callable[..., pd.DataFrame]:
  ResultType = pd.DataFrame
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    print('\nsegment_data:\n', result)
    return result
  return wrapper


def find_matches_logger(fn: Callable) -> Callable[..., tuple[pd.Series, list[int], int]]:
  ResultType = tuple[pd.Series, list[int], int]
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    found_fields, match_indices, matches_count = result
    print(f'\nfound match indices ({matches_count}):\n', match_indices)
    print(f'\nfound sample matches ({matches_count}):\n', found_fields)
    return result
  return wrapper


def clean_empty_logger(fn: Callable) -> Callable[..., FramePair]:
  ResultType = FramePair
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    found_samples, description_cleaned = result
    print(f'\nempty fields for all table:\n', count_empty(found_samples))
    print(f'\ndescription_cleaned:\n', description_cleaned)
    return result
  return wrapper


def drop_cleaaned_logger(fn: Callable) -> Callable[..., pd.DataFrame]:
  ResultType = pd.DataFrame
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    print(f'\nempty fields for segments (sample_cleaned):\n', count_empty(result))
    print(f'\nsample_cleaned:\n', result)
    return result
  return wrapper


def get_minmax_dataset_logger(fn: Callable) -> Callable[..., FramePair]:
  ResultType = FramePair
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    description_minmax, segment_description_minmax = result
    print('\nsegment_minmax:\n:', segment_description_minmax)
    return result
  return wrapper


def minmax_criteria_filter_logger(fn: Callable) -> Callable[..., tuple[pd.Series, pd.DataFrame]]:
  ResultType = tuple[pd.Series, pd.DataFrame]
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    minmax_cols, segment_sample_minmax = result
    print('\nminmax_cols:', minmax_cols.values)
    print('\nsegment_sample_minmax:\n', segment_sample_minmax)
    return result
  return wrapper


def split_minmax_logger(fn: Callable) -> Callable[..., FramePair]:
  ResultType = FramePair
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    segment_sample_min, segment_sample_max = result
    print('\nsegment_sample_min:\n', segment_sample_min)
    print('\nsegment_sample_max:\n', segment_sample_max)
    return result
  return wrapper


def analyze_scoring_map_logger(fn: Callable) -> Callable[..., tuple[np.ndarray, np.ndarray, int]]:
  ResultType = tuple[np.ndarray, np.ndarray, int]
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    normalized, integro, take = result
    print('\nnormalized:', normalized)
    print('\nintegro:', integro)
    return result
  return wrapper


def parse_age_logger(fn: Callable) -> Callable[..., pd.DataFrame]:
  ResultType = pd.DataFrame
  def wrapper(*args, **kvars) -> ResultType:
    result: ResultType = fn(*args, **kvars)
    print('\nage field formatted frame:\n', result)
    return result
  return wrapper
