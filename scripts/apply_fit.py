import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

filename = 'vers_1'

def load_csv(file_path):
  """Load CSV file into a DataFrame."""
  return pd.read_csv(file_path)


def load_model(filename):
  """ loaf pickle """
  with open(filename, 'rb') as handle:
      model_p3 = pickle.load(handle)
  return model_p3

def apply_calibration(x, model_p3):
  """
  apply degree 3 poly
  """
  poly3_coeff = model_p3
  calibrated_data = np.polyval(poly3_coeff, x)
  return calibrated_data

def plot_comparison(time_data, ft_data, force_estimates):

  std_ft = np.std(ft_data)
  plt.figure(figsize=(10, 6))

  
  plt.plot(time_data, ft_data, label='Ground Truth FT', color='blue', linewidth=2)
  plt.plot(time_data, force_estimates, label='Force Estimates', color='red', linestyle='--', linewidth=2)
  
  plt.fill_between(time_data, ft_data - std_ft, ft_data + std_ft, color='blue', alpha=0.2, label='1 Std Dev of FT')

  plt.xlabel('Time')
  plt.ylabel('Force (V)')
  plt.title('Ground truth FT and force estimates over time')
  plt.legend(loc='best')
  plt.savefig("compare_time_force.png")
  
  plt.show()

def evaluate_model(test_fsr_data, test_ft_data, model_p3, time_data):
  """
  apply calibrationa nd find mse
  """
  force_estimates = apply_calibration(test_fsr_data, model_p3)
  
  mse = mean_squared_error(test_ft_data, force_estimates)
  print(f"Mean Squared Error (MSE): {mse:.2f}")

  # pd.set_option('display.max_rows', None)  # all rows
  # combined_df = pd.DataFrame({'ground': test_ft_data, 'pred': force_estimates})
  # print(combined_df)

  plot_comparison(time_data, test_ft_data, force_estimates)



def main():
  filename = 'vers_1'

  model_filename = 'poly3_model_' + filename + '.pickle'
  model_p3 = load_model(model_filename)

  data = pd.read_csv(filename + '_aligned.csv')
  time_data = data['time']
  fsr_data = data['fsr_data']
  ft_data = data['ft_data']

  evaluate_model(fsr_data, ft_data, model_p3, time_data)


if __name__ == '__main__':
  main()
