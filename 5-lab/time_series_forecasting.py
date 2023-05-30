import numpy as np
import xgboost as xgb
from dataset import Dataset


def make_predictions_xgb(dataset: Dataset) -> tuple[xgb.XGBRegressor, np.ndarray]:
  dt = dataset
  reg_model = xgb.XGBRegressor(n_estimators=10_000, early_stopping_rounds=500, learning_rate=0.001)
  reg_model.fit(dt.x_train, dt.y_train,
                eval_set=[(dt.x_train, dt.y_train), (dt.x_test, dt.y_test)],
                verbose=100)  
  return reg_model, reg_model.predict(dt.x_test)
