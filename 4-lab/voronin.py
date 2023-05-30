import numpy as np


def voronin(criteria: np.ndarray, min_criteria_mask: np.ndarray, coefs: np.ndarray) -> tuple[np.ndarray, int]:
  criteria_sums = criteria.sum(axis=1).reshape((-1, 1))
  criteria_metrics = np.zeros(criteria.shape)
  criteria_metrics[min_criteria_mask] = criteria[min_criteria_mask] / criteria_sums[min_criteria_mask]
  criteria_metrics[~min_criteria_mask] = (criteria[~min_criteria_mask] * criteria_sums[~min_criteria_mask]) ** -1
  integal_chunks_mtx = coefs * (1 - np.transpose(criteria_metrics)) ** -1
  integral_convolition = integal_chunks_mtx.sum(axis=1)
  optimal = np.argmin(integral_convolition)
  return integral_convolition, optimal
