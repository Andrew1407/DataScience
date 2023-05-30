from types import NoneType
from typing import Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def multiple(data: dict[str, Sequence], labels: tuple[str|NoneType, str|NoneType], title: str):
  for label in data: plt.plot(data[label], label=label)
  x, y = labels
  if x is not None: plt.xlabel(x)
  if y is not None: plt.ylabel(y)
  if len(data) > 1: plt.legend()
  plt.title(title)
  plt.show()


def show_integro_score(data: dict[str, Sequence], validation_criteria: float, title: str):
  extended_data = data.copy()
  criteria_len = max(tuple(len(x) for x in data.values()))
  criteria_label = f'validation criteria ({validation_criteria})'
  extended_data[criteria_label] = np.full(criteria_len, validation_criteria)
  labels = (None, 'criteria estimation')
  multiple(extended_data, labels, title)
