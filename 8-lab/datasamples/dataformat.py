import numpy as np
import pandas as pd
import datasamples.loggers as loggers
from datasamples.utils import fix_index, FramePair, voronin, normalize_criteria


@loggers.get_datasets_logger
def get_datasets(sample_path: str, description_path: str) -> FramePair:
  data_sample = pd.read_excel(sample_path, parse_dates=['birth_date'])
  data_description = pd.read_excel(description_path)
  return data_sample, data_description


@loggers.make_segment_data_logger
def make_segment_data(data_description: pd.DataFrame) -> pd.DataFrame:
  target = 'Place_of_definition'
  necessary_fields = (
    'Указывает заемщик',
    'параметры связанные с выданным продуктом',
  )
  fields_mask = data_description[target].isin((necessary_fields))
  segment_data = data_description[fields_mask]
  fix_index(segment_data)
  return segment_data


@loggers.find_matches_logger
def find_matches(segment_data: pd.DataFrame, data_sample: pd.DataFrame, target: str) -> tuple[pd.Series, list[int], int]:
  matches_count = segment_data[target].map(lambda x: x in data_sample.columns).sum()
  found_fields = segment_data[segment_data[target].isin(data_sample.columns)]
  match_indices = found_fields.index
  fix_index(found_fields)
  return found_fields, match_indices, matches_count


@loggers.clean_empty_logger
def clean_empty(found_fields: pd.DataFrame, data_sample: pd.DataFrame, target: str) -> FramePair:
  fields_for_samples = found_fields[target]
  found_samples = data_sample[fields_for_samples]
  fix_index(found_samples)
  fields_to_filter = (
    'fact_addr_start_date',
    'position_id',
    'employment_date',
    'has_prior_employment',
    'prior_employment_start_date',
    'prior_employment_end_date',
    'income_frequency_other',
  )
  description_cleaned = found_fields[~fields_for_samples.isin(fields_to_filter)]
  fix_index(description_cleaned)
  return found_samples, description_cleaned


@loggers.drop_cleaaned_logger
def drop_cleaaned(found_samples: pd.DataFrame) -> pd.DataFrame:
  columns_to_drop = [
    'fact_addr_start_date',
    'position_id',
    'employment_date',
    'has_prior_employment',
    'prior_employment_start_date',
    'prior_employment_end_date',
    'income_frequency_other',
  ]
  sample_cleaned = found_samples.drop(columns=columns_to_drop)
  fix_index(sample_cleaned)
  return sample_cleaned


@loggers.get_minmax_dataset_logger
def get_minmax_dataset(sample_path: str, minmax_target: str) -> FramePair:
  description_minmax = pd.read_excel(sample_path)
  segment_description_minmax = description_minmax[description_minmax[minmax_target].isin(('min', 'max'))]
  fix_index(segment_description_minmax)
  return description_minmax, segment_description_minmax


@loggers.minmax_criteria_filter_logger
def minmax_criteria_filter(segment_description_minmax: pd.DataFrame, sample_cleaned: pd.DataFrame, target: str) -> tuple[pd.Series, pd.DataFrame]:
  minmax_cols = segment_description_minmax[target]
  segment_sample_minmax = sample_cleaned[minmax_cols]
  return minmax_cols, segment_sample_minmax


@loggers.split_minmax_logger
def split_minmax(segment_sample_minmax: pd.DataFrame, minmax_cols: pd.Series) -> FramePair:
  segment_sample_min = segment_sample_minmax[minmax_cols].min()
  segment_sample_max = segment_sample_minmax[minmax_cols].max()
  return segment_sample_min, segment_sample_max


@loggers.analyze_scoring_map_logger
def analyze_scoring_map(dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                        targets: tuple[str, str, str], delta: float = 0.3, take: int = None) -> tuple[np.ndarray, np.ndarray, int]:
  normalized, n = normalize_criteria(dataframes, targets, delta)
  if take is None: take = len(normalized)
  integro = voronin(normalized, n, take)
  return normalized, integro, take


@loggers.parse_age_logger
def parse_age(data: pd.DataFrame, date_field: str, age_field: str) -> pd.DataFrame:
  formatted = data.copy()
  formatted[date_field] = pd.to_datetime(formatted[date_field])
  current_date = pd.Timestamp.now()
  formatted[age_field] = (current_date - formatted[date_field]) // pd.Timedelta(days=365)
  del formatted[date_field]
  return formatted
