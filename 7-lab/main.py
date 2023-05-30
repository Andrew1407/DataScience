import utils
import dataset as ds


FILENAME = 'Data_Set_3.xls'

TARGET_COLUMN = 'Total'
GENERALIZATION_COLUMN = 'Item'
SEGMENTATION_COLUMN = 'Region'

GENERALIZATION_FACTOR = 'sum'     # can be: sum, mean, min, max
PREDICTION_PRECENTAGE = .3


if __name__ == '__main__':
  dataset = ds.get_dataset(FILENAME)
  utils.show_full_dataset(dataset, TARGET_COLUMN)
  utils.show_full_mnk_analysys(dataset, TARGET_COLUMN, PREDICTION_PRECENTAGE)
  utils.show_segment_generalization(dataset, TARGET_COLUMN, GENERALIZATION_COLUMN, GENERALIZATION_FACTOR, PREDICTION_PRECENTAGE)
  utils.show_segmentation(dataset, TARGET_COLUMN,  SEGMENTATION_COLUMN, PREDICTION_PRECENTAGE)
