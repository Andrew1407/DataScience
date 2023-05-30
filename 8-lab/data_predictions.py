import numpy as np
import pandas as pd 
import data_scoring as ds
import scoring_analysis.scoring_predictor as sp
import scoring_analysis.validation as sav
import scoring_analysis.tools as sa
import plots


MODEL_PATH = 'scoring_predictor_model'
FRAUD_BARRIER = 2


def fltten_anomalies(integro: np.ndarray) -> np.ndarray:
  flatened = sa.flatten_data(integro)
  print('\nintegro flattened:', flatened)
  plots.show_integro_score(validation_criteria=ds.VALIDATION_CRITERIA, title='Integro score anomalies flttened comparison', data={
    'original data sample': integro,
    'flatened': flatened,
  })
  plots.show_integro_score(dict(flatened=flatened), ds.VALIDATION_CRITERIA, title='Integro score anomalies flttened')
  return flatened


def handle_fraud(dataset: pd.DataFrame, integro: np.ndarray) -> np.ndarray:
  contradictions, valid = sav.validate_table(dataset, FRAUD_BARRIER)
  frauds_mask = ~valid.astype(np.bool8)
  frauds = dataset[frauds_mask].index
  integro_filtered = integro.copy()
  integro_filtered[frauds_mask] = 0
  print('\ncontradictions:', contradictions)
  print(f'\nvalidation table (no fraud - {valid.sum()} elements):', valid)
  print('\nfraunds indices:', frauds)
  plots.show_integro_score(validation_criteria=ds.VALIDATION_CRITERIA, title='Integro score fraud filtered', data={
    'original data sample': integro,
    'fraud filtered': integro_filtered,
  })
  return integro_filtered


def predict_score(dataset: pd.DataFrame, integro: np.ndarray):
  x, y = dataset.values, integro.reshape(-1, 1)
  model = sa.get_model(MODEL_PATH, data=(x, y))
  model.summary()
  predicted, valid = sp.predict(model, x, ds.VALIDATION_CRITERIA)
  print('\npredicted integro:', predicted)
  print('\npredicted binary (valid):', valid)
  plots.show_integro_score(validation_criteria=ds.VALIDATION_CRITERIA, title='Integro score prediction', data={
    'fraud filtered sample': integro,
    'predicted': predicted,
  })
