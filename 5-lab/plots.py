import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from dataset import Dataset


def show_data_plot(dataset: Dataset):
  data = dataset.data
  rolling = data['new_cases'].rolling(window=10)
  sns.set()
  plt.ylabel('new_cases')
  plt.xlabel('date')
  plt.xticks(rotation=45)
  plt.plot(data.index, data['new_cases'], label='cases')
  plt.plot(data.index, rolling.mean(), label='rolling mean')
  plt.plot(data.index, rolling.std(), label='rolling std')
  plt.plot(data.index, [data['new_cases'].mean()] * len(data.index), label='mean')
  plt.title('COVID new cases in ' + dataset.country_region)
  plt.legend()
  plt.show()


def show_data_split(dataset: Dataset):
  train = dataset.train
  test = dataset.test
  sns.set()
  plt.plot(train.index, train['new_cases'], color = 'black', label = 'training')
  plt.plot(test.index, test['new_cases'], color = 'red', label = 'testing')
  plt.plot(train.index, [train['new_cases'].mean()] * len(train.index), label='train mean', color='orange')
  plt.plot(test.index, [test['new_cases'].mean()] * len(test.index), label='test mean', color='blue')
  plt.ylabel('new_cases')
  plt.xlabel('date')
  plt.xticks(rotation=45)
  plt.title('Train/Test split for COVID new cases')
  plt.legend()
  plt.show()


def show_data_predictions(dataset: Dataset, predictions: pd.DataFrame):
  train = dataset.train
  test = dataset.test
  rolling_pred = predictions['predictions'].rolling(window=10)
  sns.set()
  plt.plot(train.index, train['new_cases'], color = 'black', label = 'training')
  plt.plot(test.index, test['new_cases'], color = 'red', label = 'testing')
  plt.plot(train.index, [train['new_cases'].mean()] * len(train.index), label='train mean', color='orange')

  plt.plot(predictions.index, predictions['predictions'], color = 'green', label = 'predicted')
  plt.plot(predictions.index, rolling_pred.mean(), label='predicted rolling mean', color='purple')
  plt.plot(predictions.index, rolling_pred.std(), label='predicted rolling std', color='yellowgreen')
  plt.plot(predictions.index, [predictions['predictions'].mean()] * len(predictions.index), label='predicted mean', color='blue')
  
  plt.ylabel('new_cases')
  plt.xlabel('date')
  plt.xticks(rotation=45)
  plt.title('Train/Test split & prediction for COVID Data')
  plt.legend()
  plt.show()


def show_importance_table(model: xgb.XGBRegressor):
  fi = pd.DataFrame(data=model.feature_importances_, index=model.feature_names_in_, columns=['importance'])
  fi.sort_values('importance').plot(kind='barh', title='Feature importances')
  plt.show()
