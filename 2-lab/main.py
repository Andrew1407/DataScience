import value_generators as vg
import sample_purification as sp
import numpy as np


ARRAY_SIZE = 10_000

def mk_initial_data() -> tuple[np.ndarray, np.ndarray]:
  normal, _ = vg.mk_distribution_sets(ARRAY_SIZE)
  square, _ = vg.mk_trend_sets(ARRAY_SIZE)
  data_to_analyze = normal | square
  vg.show_statistics(data_to_analyze)
  return tuple(tuple(s.values())[0] for s in (normal, square))


if __name__ == '__main__':
  normal, square = mk_initial_data()
  composition = normal + square

  print()
  sp.show_stats_mnk(composition, apply_mnk=True, title='MNK set & normal distrubution noise')

  abnormal_percentage = .1
  quality_coef = 3
  abnormal_size = int(len(composition) * abnormal_percentage)

  ab = sp.mk_abnormal_set(square, composition, abnormal_size, quality_coef)
  title = 'Square trend & normal distribultion with abnormal error'
  print()
  vg.plot_sales_dynamics(square, ab, title)
  sp.show_stats_mnk(composition, apply_mnk=True, title=title)

  # Q_MNK, n_Wind = 5, 10
  Q_MNK, n_Wind = sp.learn_detection_params(square, composition, abnormal_percentage, quality_coef)
  print(f'{Q_MNK = } {n_Wind = }')
  slided = sp.sliding_window_AV_detect_mnk(ab, Q_MNK, n_Wind)
  title = 'Square trend & normal distribultion cleaned with mnk'
  print()
  vg.plot_sales_dynamics(square, slided, title)
  sp.show_stats_mnk(slided, apply_mnk=False, title=title)
  print()
  sp.show_stats_mnk(slided, apply_mnk=True, title=title + ' (mnk applied)')
