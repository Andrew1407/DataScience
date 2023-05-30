import pandas as pd
import os.path
import mnk
import nn
import dataset as ds
import utils


URL = 'https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip'
FILENAME = 'jena_climate_2009_2016.csv'
MODELNAME = 'atmospheric_pressure_predictor'
TARGET_COLUMN = 'ap'


def generate_extrapolation(dataset: pd.DataFrame, target: str) -> pd.DataFrame:
  print()
  extrapolated, regression_coefs = mnk.extrapolate(sample=dataset[target], predict=.5)
  mnk.show_regression_model(regression_coefs)
  extrapolation = utils.extend_dataframe(dataset, extrapolated)
  utils.show_metrics_extended(collection=extrapolation['extrapolated'], pivot=len(dataset), label='mnk extrapolation')
  utils.show_mnk_extrapolation(dataset, extrapolation, target)
  utils.show_rolling_std(dataset[target], dataset.index, label='rolling std', title='Rolling std')
  return extrapolation


def generate_predictions(dataset: pd.DataFrame, target: str, model_name: str, inputs: pd.DataFrame) -> pd.DataFrame:
  model = None
  if os.path.exists(model_name):
    model = nn.load_model(model_name)
  else:
    model, mse = nn.create_model(dataset, model_name, target)
    print(f'{mse = }')
  model.summary()
  input_features = inputs.index
  predicted = nn.predict(model, input_features)
  predictions = predicted['predicted']
  utils.show_metrics_extended(collection=predictions, pivot=len(dataset), label='nn prediction')
  utils.show_nn_predictions(dataset, predicted, target, full=True)
  utils.show_nn_predictions(dataset, predicted, target, full=False)
  utils.show_rolling_std(predictions, predicted.index, label='predicted rolling std', title='Predicted rolling std')
  return predicted


if __name__ == '__main__':
  dataset = ds.get_dataset(URL, FILENAME)
  print(f'\n{dataset = }')
  print('\nMertics for the fulll original dataset:')
  utils.get_metrics(dataset[TARGET_COLUMN])
  extrapolated = generate_extrapolation(dataset, TARGET_COLUMN)
  generate_predictions(dataset, TARGET_COLUMN, MODELNAME, extrapolated)
