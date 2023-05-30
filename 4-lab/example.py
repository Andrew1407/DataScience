import pandas as pd
import numpy as np
from voronin import voronin


CRITERIA_COUNT = 9
SAMPLES_COUNT = 9
CONTAINER_SHAPE = CRITERIA_COUNT, SAMPLES_COUNT

def parse_table(path: str) -> tuple[np.ndarray, np.ndarray]:
  criteria = np.zeros(shape=CONTAINER_SHAPE)
  table = pd.read_excel(path)
  print(f'{table = }')
  min_max_mask = table['Критерій'].to_numpy() == 'мін'
  for index, _ in np.ndenumerate(criteria):
    criteria_index, sample_index = index
    label = f'Товар {SAMPLES_COUNT - sample_index}'
    table_criteria = table[label][criteria_index].replace(',', '.')
    criteria[index] = float(table_criteria)
  return criteria, min_max_mask


def make_criteria_coefs() -> tuple[np.ndarray, np.ndarray]:
  coefs = np.ones(SAMPLES_COUNT)
  coefs /= np.sum(coefs) + 1
  min_criteria_mask = np.ones(CRITERIA_COUNT).astype(bool)
  min_criteria_mask[5] = False
  return coefs, min_criteria_mask


def analyze_criteria():
  criteria, min_max_mask = parse_table(path='tables/Pr1.xls')
  coefs, min_criteria_mask = make_criteria_coefs()
  integral_convolition, optimal = voronin(criteria, min_max_mask, coefs)
  print(f'\n{integral_convolition = }')
  print(f'\nThe most optimal item is {optimal} - "Товар {SAMPLES_COUNT - optimal}".')
