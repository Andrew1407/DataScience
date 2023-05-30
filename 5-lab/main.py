from typing import Sequence
import dataset as ds
import time_series_forecasting as tsf
import plots


URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
DATASET_PATH = 'stats.csv'

# COUNTRY = 'Ukraine'
# COUNTRY = 'United States'
# COUNTRY = 'Afghanistan'
COUNTRY = 'United Kingdom'

DATE_SPLIT = '2021-10-01'
# DATE_SPLIT = '2021-12-01'
# DATE_SPLIT = '2022-03-01'


def get_dataset() -> ds.Dataset:
  dataset_full = ds.fetch_dataset(URL, DATASET_PATH)
  dataset_formatted = ds.format_dataset(dataset_full, COUNTRY)
  print(f'{dataset_full=}')
  print(f'{dataset_formatted=}')
  return ds.make_dataset(dataset_formatted, DATE_SPLIT, COUNTRY)


def get_metrics(data: Sequence[float]) -> tuple[float, float, float]:
  metrics = ds.calc_metrics(data)
  median, variance, dispersion = metrics
  print(f'{median = }')
  print(f'{variance = }')
  print(f'{dispersion = }')
  return metrics


if __name__ == '__main__':
  dt = get_dataset()
  
  print('\nFull dataset metrics for new_cases:')
  dt_metics = get_metrics(dt.data['new_cases'])
  plots.show_data_plot(dt)

  print('\nTrain dataset metrics for new_cases:')
  train_metics = get_metrics(dt.train['new_cases'])

  print('\nTest dataset metrics for new_cases:')
  test_metics = get_metrics(dt.test['new_cases'])
  plots.show_data_split(dt)

  reg_model, predictions = tsf.make_predictions_xgb(dt)
  predicted = dt.test.copy()
  del predicted['new_cases']
  predicted['predictions'] = predictions

  print('\nTest dataset metrics for new_cases:')
  predictions_metics = get_metrics(predictions)

  plots.show_data_predictions(dt, predicted)
  plots.show_importance_table(reg_model)
