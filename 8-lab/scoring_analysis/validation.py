import numpy as np
import pandas as pd


def fraud_check(data: pd.DataFrame, i: int) -> int:
  contradictions = 0
  field_of = lambda f: data[f][i]
  age = field_of('age')
  contradictions += age < 20
  contradictions += (age - 20) > field_of('seniority_years')
  ids = 'organization_type_id', 'organization_branch_id', 'income_frequency_id', 'income_source_id'
  for id in ids: contradictions += field_of(id) > 100
  contradictions += field_of('monthly_income') < field_of('monthly_expenses')
  contradictions += field_of('other_loans_active') > 2
  contradictions += field_of('other_loans_about_current') < field_of('other_loans_about_monthly')
  contradictions += field_of('product_amount_from') > field_of('product_amount_to')
  contradictions += field_of('has_immovables') > 0 and age < 20
  return contradictions


def validate_table(data: pd.DataFrame, fraud_barrier: int = 2) -> tuple[np.ndarray, np.ndarray]:
  data_len = len(data)
  contradictions = np.zeros(data_len, dtype=np.int32)
  for i in range(data_len):
    contradictions[i] = fraud_check(data, i)
  valid = (contradictions <= fraud_barrier).astype(np.int32)
  return contradictions, valid
