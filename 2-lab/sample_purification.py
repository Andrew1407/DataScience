import value_generators as vg
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from sklearn.metrics import confusion_matrix


def mnk_flattening(values_set: np.ndarray, flat: bool = True) -> np.ndarray:
  iter = len(values_set)
  Yin = np.zeros((iter, 1))
  F = np.ones((iter, 3))
  for i in range(iter):  # формування структури вхідних матриць МНК
    Yin[i, 0] = float(values_set[i])  # формування матриці вхідних даних
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)
  FT=F.T
  FFT = FT.dot(F)
  FFTI=np.linalg.inv(FFT)
  FFTIFT=FFTI.dot(FT)
  C=FFTIFT.dot(Yin)
  Yout=F.dot(C)
  return Yout.flatten() if flat else Yout


def mk_abnormal_set(values_set: np.ndarray, composition: np.ndarray, abnormal_size: int, quality_coef: float) -> np.ndarray:
  random_errors = np.random.normal(0, quality_coef * 5, abnormal_size)  # аномальна випадкова похибка з нормальним законом
  values_count = len(values_set)
  abnormal = composition.copy()
  for i in range(abnormal_size):
    k = math.ceil(np.random.randint(1, values_count))
    abnormal[k] = values_set[k] + random_errors[i]        # аномальні вимірів з рівномірно розподіленими номерами
  return abnormal


def show_stats_mnk(values: np.ndarray, title: str, apply_mnk: bool = False):
  print(title + ':')
  values = mnk_flattening(values) if apply_mnk else values
  stats = values - mnk_flattening(values)
  vg.print_value_params(stats)
  vg.plot_values(stats, title)


def mnk_detect(values: np.ndarray) -> float:
    iter = len(values)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
      Yin[i, 0] = float(values[i])  # формування матриці вхідних даних
      F[i, 1] = float(i)
      F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    return C[1,0]


def sliding_window_AV_detect_mnk(values: np.ndarray, q_mnk: float, n_wind: float) -> np.ndarray:
  slided = values.copy()
  # ---- параметри циклів ----
  iter = len(slided)
  j_Wind=math.ceil(iter-n_wind)+1
  S0_Wind=np.zeros(n_wind)
  # -------- еталон  ---------
  Speed_ethadone = mnk_detect(slided)
  Yout_S0 = mnk_flattening(slided, flat=False)
  # ---- ковзне вікно ---------
  for j in range(j_Wind):
    for i in range(n_wind):
      l=(j+i)
      S0_Wind[i] = slided[l]
# - Стат хар ковзного вікна --
    dS = np.var(S0_Wind)
    scvS = math.sqrt(dS)
# --- детекція та заміна АВ --
    Speed_ethadone_1 = abs(Speed_ethadone * math.sqrt(iter))
    # Speed_1 = abs(Speed / (Q*scvS))
    Speed_1 = abs(q_mnk * Speed_ethadone * math.sqrt(n_wind) * scvS)
    # print('Speed_ethadone=', Speed_ethadone_1)
    # print('Speed_1=', Speed_1)
    if Speed_1  > Speed_ethadone_1:
      # детектор виявлення АВ
      # print('S0[l] !!!=', S0[l])
      slided[l]=Yout_S0[l,0]
  return slided


def learn_detection_params(values: np.ndarray, composition: np.ndarray, abnormal_percentage: float, quality_coef: float) -> tuple[int, int]:
  """
  Learn the optimal values of Q_MNK and n_Wind for detecting abnormal values in the input statistical sample.

  Args:
      composition: The input statistical sample to analyze.
      abnormal_percentage: The percentage of abnormal values in the sample.
      quality_coef: The quality coefficient for detecting abnormal values.

  Returns:
      A tuple containing the optimal values of Q_MNK and n_Wind.
  """
  sizes = np.arange(5, 15, 1)  # possible sizes of the sliding window
  q_mnks = np.arange(5, 15, 1)  # possible values of the quality coefficient
  best_params = (0, 0)
  best_score = 0

  def compute_score(n_Wind, Q_MNK):
    ab = mk_abnormal_set(values, composition, int(len(composition) * abnormal_percentage), quality_coef)
    detected = sliding_window_AV_detect_mnk(ab, Q_MNK, n_Wind)
    cm = confusion_matrix(ab < np.mean(ab), detected < np.mean(detected))
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    return tpr, (Q_MNK, n_Wind)

  workers = multiprocessing.cpu_count()
  with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(compute_score, n_Wind, Q_MNK) for n_Wind in sizes for Q_MNK in q_mnks]
    for future in as_completed(futures):
      score, params = future.result()
      if score > best_score:
        best_score = score
        best_params = params
  return best_params
