import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import torch


filename = 'flat_iter_1'

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
    

def apply_mlp(x_raw, model_path, norm_path):
    """
    Load MLP model and apply it to new data.
    """
    norm = np.load(norm_path)
    x_mean, x_std = norm['x_mean'], norm['x_std']
    y_mean, y_std = norm['y_mean'], norm['y_std']

    x_norm = (x_raw - x_mean) / x_std
    x_tensor = torch.tensor(x_norm.to_numpy().reshape(-1, 1), dtype=torch.float32)

    model = TwoLayerNet(1, 20, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(x_tensor).numpy()
        y_pred = y_pred * y_std + y_mean 

    return y_pred

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

def plot_comparison(time_data, ft_data, mlp_estimates, poly3_estimates):
    """
    plots poly3 and mlp
    """
    std_ft = np.std(ft_data)
    plt.figure(figsize=(10, 6))

    plt.plot(time_data, ft_data, label='Ground Truth FT', color='blue', linewidth=2)
    plt.plot(time_data, mlp_estimates, label='MLP Estimates', color='red', linestyle='--', linewidth=2)
    plt.plot(time_data, poly3_estimates, label='Poly3 Estimates', color='green', linestyle=':', linewidth=2)
    
    plt.fill_between(time_data, ft_data - std_ft, ft_data + std_ft, color='blue', alpha=0.2, label='1 Std Dev of FT')

    plt.xlabel('Time')
    plt.ylabel('Force (V)')
    plt.title('Ground truth FT and force estimates over time')
    plt.legend(loc='best')
    plt.savefig(filename + "_compare_time_force.png")
    plt.show()

def evaluate_pol3(test_fsr_data, test_ft_data, model_p3):
  """
  apply calibrationa nd find mse
  """
  force_estimates = apply_calibration(test_fsr_data, model_p3)
  
  mse = mean_squared_error(test_ft_data, force_estimates)
  print(f"Mean Squared Error (MSE): {mse:.2f}")

  # pd.set_option('display.max_rows', None)  # all rows
  # combined_df = pd.DataFrame({'ground': test_ft_data, 'pred': force_estimates})
  # print(combined_df)
  return force_estimates



def main():
  # filename = 'vers_2'

  model_filename = 'poly3_model_' + filename + '.pickle'
  model_p3 = load_model(model_filename)

  data = pd.read_csv(filename + '_aligned.csv')
  time_data = data['time']
  fsr_data = data['fsr_data']
  ft_data = data['ft_data']

  print("POLY3")
  poly3_force_estimates = evaluate_pol3(fsr_data, ft_data, model_p3)

  print("MLP")
  mlp_model_path = f"mlp_model_{filename}.pt"
  norm_path = f"mlp_norm_{filename}.npz"
  mlp_force_estimates = apply_mlp(fsr_data, mlp_model_path, norm_path)

  mse = mean_squared_error(ft_data, mlp_force_estimates)
  print(f"MLP Mean Squared Error (MSE): {mse:.2f}")

  plot_comparison(time_data, ft_data, mlp_force_estimates, poly3_force_estimates)


if __name__ == '__main__':
  main()
