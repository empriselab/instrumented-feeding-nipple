from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
import scipy.linalg
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim


filename = 'flat_iter_1'

csv_names = '_flat'

def load_csv(file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def find_max_peak(time, signal):
    """Find the maximum peak in the signal and return its time, and value."""
    peaks, _ = find_peaks(signal)
    
    if len(peaks) > 0:
        signal_np = np.asarray(signal)
        time_np = np.asarray(time)

        max_peak_idx = peaks[signal_np[peaks].argmax()]
        max_peak_time = time_np[max_peak_idx]
        max_peak_value = signal_np[max_peak_idx]
        print(f"Max peak detected at time: {max_peak_time}, Value: {max_peak_value}")
        return max_peak_time, max_peak_value
    else:
        print("No peak detected.")
        return None, None
    

def find_first_peak(time, signal):
    """Find the first local peak and return its time and value."""
    peaks, _ = find_peaks(signal, height=signal.median(), prominence=50)
    
    if len(peaks) > 0:
        signal_np = np.asarray(signal)
        time_np = np.asarray(time)
        
        first_peak_idx = peaks[0]
        first_peak_time = time_np[first_peak_idx]
        first_peak_value = signal_np[first_peak_idx]
        print(f"First peak detected at time: {first_peak_time}, Value: {first_peak_value}")
        return first_peak_time, first_peak_value
    else:
        print("No local peak detected.")
        return None, None

def preprocess_align(df1, df2, time_col, signal_col,  align_mode='max'):
    df1['data'] = abs(df1['data'])
    min_index = df2['data'].idxmin()
    min_time = df2['time'].iloc[min_index]
    min_value = df2['data'].iloc[min_index]
    threshold = 0
    baseline_start_index = df2['data'].iloc[min_index:].diff().abs() <= threshold
    baseline_start_index = baseline_start_index.idxmax()
    baseline_value = df2['data'].iloc[baseline_start_index:].median()
    df2['data'] -= baseline_value 
    df2['data'] = df2['data'].clip(lower=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(df1[time_col], df1[signal_col], label='FT')
    plt.plot(df2[time_col], df2[signal_col], label='FSR')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('Original Signals')
    plt.show()


    if align_mode == 'max':
        peak_func = find_max_peak
    elif align_mode == 'first':
        peak_func = find_first_peak
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    t1_peak, y1_peak = peak_func(df1[time_col], df1[signal_col])
    t2_peak, y2_peak = peak_func(df2[time_col], df2[signal_col])
    
    if t1_peak is None or t2_peak is None:
        print("Could not detect peaks in one or both signals.")
        return df1, df2
    
    shift = t1_peak - t2_peak
    df2[time_col] += shift
    # factor = y2_peak / y1_peak
    # df1[signal_col] *= factor
    # factor = y1_peak / y2_peak  
    # df2[signal_col] *= factor 

    min_ft_time = df1[time_col].min()
    max_ft_time = df1[time_col].max()
    df2 = df2[(df2[time_col] >= min_ft_time) & (df2[time_col] <= max_ft_time)]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(df1[time_col], df1[signal_col], label='FT')
    plt.plot(df2[time_col], df2[signal_col], label='FSR (Aligned)')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('Aligned Signals')

    t1_new_peak, y1_new_peak = peak_func(df1[time_col], df1[signal_col])
    t2_new_peak, y2_new_peak = peak_func(df2[time_col], df2[signal_col])
    
    if t1_new_peak is not None:
        plt.plot(t1_new_peak, y1_new_peak, 'ro', label='FT First Peak')
    if t2_new_peak is not None:
        plt.plot(t2_new_peak, y2_new_peak, 'ro', label='FSR First Peak')
    
    plt.tight_layout()
    plt.show()
    
    return df1, df2


def convert_timestamps(ft_data, fsr_data):
    """Convert one of the time columns to match the other based on mean time value."""
    ft_max = ft_data['time'].max()
    fsr_max = fsr_data['time'].max()

    print(f"before conversion:")
    print(f"FT time range: {ft_data['time'].min()} to {ft_data['time'].max()}")
    print(f"FSR time range: {fsr_data['time'].min()} to {fsr_data['time'].max()}")

    if fsr_max > 10 * ft_max:
        print("Converting FSR time from microeconds to seconds.")
        fsr_data['time'] = fsr_data['time'] / 1000000.0
    elif ft_max > 10 * fsr_max:
        print("Converting FT time from microseconds to seconds.")
        ft_data['time'] = ft_data['time'] / 1000000.0
    else:
        print("No time conversion needed.")

    print(f"After conversion:")
    print(f"FT time range: {ft_data['time'].min()} to {ft_data['time'].max()}")
    print(f"FSR time range: {fsr_data['time'].min()} to {fsr_data['time'].max()}")
    
    return ft_data, fsr_data


def load_baseline(align_mode='max'):
    '''
    finalize and save the fully interpolated and aligned dfs
    '''
    ft_data = load_csv('FT' + csv_names + '.csv')
    fsr_data = load_csv('FSR' + csv_names + '.csv')
    fsr_data.rename(columns={'Timestamp': 'time'}, inplace=True)
    fsr_data.rename(columns={'FSR 1': 'data'}, inplace=True)
    ft_data.rename(columns={'ft_time': 'time'}, inplace=True)

    ft_data, fsr_data = convert_timestamps(ft_data, fsr_data)

    print(f"FT time range: {ft_data['time'].min()} to {ft_data['time'].max()}")
    print(f"FSR time range: {fsr_data['time'].min()} to {fsr_data['time'].max()}")

    ft_data.rename(columns={'ft': 'data'}, inplace=True)
    ft_data, fsr_data = preprocess_align(ft_data, fsr_data, 'time', 'data', align_mode=align_mode)

    ft_interp = np.interp(fsr_data['time'], ft_data['time'], ft_data['data'])
    fsr = fsr_data['data'] 

    interpolated_df = pd.DataFrame({'time': fsr_data['time'], 'ft_data': ft_interp, 'fsr_data': fsr})
    interpolated_df.to_csv(f"{filename}_aligned.csv", index=False)

    return ft_interp, fsr

def calibration_curve(ft, fsr):
    '''
    applies linear, log, polynomial degree 3 and 5 and saves to png. 
    currently save the polynomial degree 5 to pickle file to then use in apply_fit.py
    '''
    def linear_regression(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = scipy.linalg.lstsq(A, y)[0]
        print("linear_regression")
        print(f"Slope: {m}, Intercept: {c}")
        return m, c

    def log_func(x, a, b, k):
        return a * np.log((b * abs(x)) + 1e-5) + k

    def log_linear_regression(x, y):
        coeff, _ = curve_fit(log_func, x, y + 1e-5, maxfev = 10000)
        print(f"log_linear_regression coefficients: {coeff}")
        return coeff

    def poly3_regression(x, y):
        coeff = np.polyfit(x, y, 3)
        print(f"Poly3 coefficients: {coeff}")
        return coeff

    def poly5_regression(x, y):
        coeff = np.polyfit(x, y, 5)
        print(f"Poly5 coefficients: {coeff}")
        return coeff

    
    m, c = linear_regression(fsr, ft)
    # a, b, k = log_linear_regression(fsr, ft)
    poly3_coeff = poly3_regression(fsr, ft)
    poly5_coeff = poly5_regression(fsr, ft)

    # a0, a1, a2, a3 = poly3_coeff
    # model_p3 = {'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3}
    with open("poly3_model_" + filename + ".pickle", 'wb') as handle:
        # pickle.dump(model_p3, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(poly3_coeff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    linear_fit = m * fsr + c
    # log_fit = log_func(fsr, a, b, k)
    poly3_fit = np.polyval(poly3_coeff, fsr)
    poly5_fit = np.polyval(poly5_coeff, fsr)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[0].plot(fsr, linear_fit, label='Linear Fit', color='red')
    axs[0].set_title('Linear Fit')
    axs[0].set_xlabel('FSR Signal')
    axs[0].set_ylabel('FT Signal')
    axs[0].legend()


    axs[1].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[1].plot(fsr, poly3_fit, label='Poly3 Fit', color='blue')
    axs[1].set_title('Poly3 Fit')
    axs[1].set_xlabel('FSR Signal')
    axs[1].set_ylabel('FT Signal')
    axs[1].legend()

    df = pd.DataFrame({
    'FSR Signal': fsr,
    'Actual FT': ft,
    'Predicted FT': poly3_fit
    })
    pd.set_option('display.max_rows', None) 

    plt.savefig(filename + "_diff_fit_models.png")
    plt.tight_layout()
    plt.show()

    return




# credit to my ORIE 3741 professor Soroosh Shafiee for the two layer NN layout
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        """
        
        super(TwoLayerNet, self).__init__()
        
        self.net = nn. Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
        
    def forward(self, x):
        """ Forward pass of the network """
        return self.net(x)


def mlp(x_preprocessed, y_preprocessed):

    x_mean = x_preprocessed.mean()
    x_std = x_preprocessed.std()
    x_preprocessed = (x_preprocessed - x_mean) / x_std
    x_raw = x_preprocessed * x_std + x_mean

    y_mean = y_preprocessed.mean()
    y_std = y_preprocessed.std()
    y_preprocessed = (y_preprocessed - y_mean) / y_std
    y_raw = y_preprocessed * y_std + y_mean


    x_train = torch.tensor(x_preprocessed.to_numpy().reshape(-1, 1), dtype=torch.float32)
    y_train = torch.tensor(y_preprocessed.reshape(-1, 1), dtype=torch.float32)

    model = TwoLayerNet(input_size=1, hidden_size=20, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 500
    for epoch in range(epochs):
        model.train()
        
        # forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train).numpy()
        y_pred = y_pred * y_std + y_mean  
    
    plt.figure(figsize=(8, 5))

    plt.scatter(x_raw, y_raw, label='measured data', alpha=0.6)
    plt.plot(x_raw, y_pred, color='red', label='mlp fit')
    plt.xlabel("fsr reading (raw)")
    plt.ylabel("force (N)")
    
    plt.title("calibration using shallow mlp")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), f"mlp_model_{filename}.pt")
    np.savez(f"mlp_norm_{filename}.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


    
def main():
    ft_data, fsr_data = load_baseline(align_mode='first')

    calibration_curve(ft_data, fsr_data)

    mlp(fsr_data, ft_data)

    

if __name__ == '__main__':
    main()