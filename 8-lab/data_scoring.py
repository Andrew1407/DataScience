import os
import pandas as pd
import numpy as np
import datasamples.dataformat as df
import plots


TARGET_FIELD = 'Field_in_data'

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

DATASAMPLE_PATH = f'{INPUT_DIR}/Pr15_sample_data.xlsx'
DESCRIPTION_PATH = f'{INPUT_DIR}/Pr15_data_description.xlsx'
DESCRIPTION_MINMAX_PATH = f'{INPUT_DIR}/d_segment_data_description_cleaning_minimax.xlsx'

SAMPLE_DESCRIPTION_CLEANED_PATH = f'{OUTPUT_DIR}/d_segment_data_description_cleaning.xlsx'
SEGMENT_CLEANED_PATH = f'{OUTPUT_DIR}/d_segment_sample_cleaning.xlsx'
SEGMENT_DESCRIPTION_MINMAX_PATH = f'{OUTPUT_DIR}/d_segment_data_description_minimax.xlsx'
SEGMENT_SAMPLE_MINMAX_PATH = f'{OUTPUT_DIR}/d_segment_sample_minimax.xlsx'

INTEGRO_SCORE_PATH = f'{OUTPUT_DIR}/Integro_Scor.txt'
NORMALIZED_PATH = f'{OUTPUT_DIR}/d_segment_sample_minimax_Normal.txt'

VALIDATION_CRITERIA = 1000


def setup():
  if not os.path.exists(INPUT_DIR):
    os.mkdir(INPUT_DIR)
  if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def prepare_data() -> pd.DataFrame:
  data_sample, data_description = df.get_datasets(DATASAMPLE_PATH, DESCRIPTION_PATH)
  segment_data = df.make_segment_data(data_description)
  found_fields, match_indices, matches_count = df.find_matches(segment_data, data_sample, TARGET_FIELD)
  found_samples, description_cleaned = df.clean_empty(found_fields, data_sample, TARGET_FIELD)
  sample_cleaned = df.drop_cleaaned(found_samples)

  description_cleaned.to_excel(SAMPLE_DESCRIPTION_CLEANED_PATH)
  sample_cleaned.to_excel(SEGMENT_CLEANED_PATH)

  return sample_cleaned


def parse_scoring_map(sample_cleaned: pd.DataFrame) -> tuple[np.ndarray, int]:
  minmax_target = 'Minimax'
  _, segment_description_minmax = df.get_minmax_dataset(DESCRIPTION_MINMAX_PATH, minmax_target)
  minmax_cols, segment_sample_minmax = df.minmax_criteria_filter(segment_description_minmax, sample_cleaned, TARGET_FIELD)
  segment_sample_min, segment_sample_max = df.split_minmax(segment_sample_minmax, minmax_cols)
  criteria_field = 'loan_amount'
  normalized, integro, take = df.analyze_scoring_map(
    dataframes=(segment_description_minmax, segment_sample_minmax, segment_sample_max),
    targets=(TARGET_FIELD, criteria_field, minmax_target),
    delta=.3,
    # take=150,
  )

  segment_description_minmax.to_excel(SEGMENT_DESCRIPTION_MINMAX_PATH)
  segment_sample_minmax.to_excel(SEGMENT_SAMPLE_MINMAX_PATH)
  np.savetxt(NORMALIZED_PATH, normalized)
  np.savetxt(INTEGRO_SCORE_PATH, integro)
  plots.show_integro_score(dict(integro=integro), VALIDATION_CRITERIA, title='Integro score')

  return integro, take
