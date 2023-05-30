import pandas as pd
import numpy as np
from dataclasses import dataclass
from voronin import voronin


@dataclass
class RouteMetrics:
  table: pd.DataFrame
  route_types: np.ndarray
  metric_types: np.ndarray
  coefs: np.ndarray
  criteria_mask: np.ndarray
  criteria: np.ndarray

  @staticmethod
  def of(path: str):
    table = pd.read_excel(path)
    labels_to_drop = ['Коефіцієнти переваги критеріїв', 'Критерій', 0]
    [coefs_label, criteria_label, types_label] = labels_to_drop
    criteria_only_table = table.drop(columns=labels_to_drop)
    return RouteMetrics(
      table=table,
      route_types=criteria_only_table.columns.to_numpy(),
      metric_types=table[types_label].to_numpy(),
      coefs=table[coefs_label].to_numpy(),
      criteria_mask=table[criteria_label].to_numpy() == 'min',
      criteria=criteria_only_table.to_numpy()
    )


def analyze_criteria():
  route_metrics = RouteMetrics.of(path='tables/routes.xls')
  integral_convolition, optimal = voronin(route_metrics.criteria, route_metrics.criteria_mask, route_metrics.coefs)
  print(f'{route_metrics.table = }')
  print(f'\n{integral_convolition = }')
  print(f'\nThe most way item is {optimal} - "{route_metrics.route_types[optimal]}".')
