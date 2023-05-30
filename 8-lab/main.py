import numpy as np
import pandas as pd 
import datasamples.dataformat as df
import data_scoring as ds
import data_predictions as dp


def calc_scoring() -> tuple[pd.DataFrame, np.ndarray]:
  ds.setup()
  sample_cleaned = ds.prepare_data()
  integro, take = ds.parse_scoring_map(sample_cleaned)
  return sample_cleaned[:take], integro


def make_predictions(sample_cleaned: pd.DataFrame, integro: np.ndarray):
  integro = dp.fltten_anomalies(integro)
  dataset = df.parse_age(sample_cleaned, date_field='birth_date', age_field='age')
  integro = dp.handle_fraud(dataset, integro)
  dp.predict_score(dataset, integro)


if __name__ == '__main__':
  sample_cleaned, integro = calc_scoring()
  make_predictions(sample_cleaned, integro)
