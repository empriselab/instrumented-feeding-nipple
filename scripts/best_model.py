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



filename = 'vers_1'

def load_csv(file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def find_max_peak(time, signal):
    """Find the maximum peak in the signal and return its time, and value."""
    peaks, _ = find_peaks(signal)
    
    if len(peaks) > 0:
        max_peak_idx = peaks[signal[peaks].argmax()]
        max_peak_time = time[max_peak_idx]
        max_peak_value = signal[max_peak_idx]
        print(f"Max peak detected at time: {max_peak_time}, Value: {max_peak_value}")
        return max_peak_time, max_peak_value
    else:
        print("No peak detected.")
        return None, None

def preprocess_align(df1, df2, time_col, signal_col):
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

    t1_peak, y1_peak = find_max_peak(df1[time_col], df1[signal_col])
    t2_peak, y2_peak = find_max_peak(df2[time_col], df2[signal_col])
    
    if t1_peak is None or t2_peak is None:
        print("Could not detect peaks in one or both signals.")
        return df1, df2
    
    shift = t1_peak - t2_peak
    df2[time_col] += shift
    factor = y2_peak / y1_peak
    df1[signal_col] *= factor

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

    t1_new_peak, y1_new_peak = find_max_peak(df1[time_col], df1[signal_col])
    t2_new_peak, y2_new_peak = find_max_peak(df2[time_col], df2[signal_col])
    
    if t1_new_peak is not None:
        plt.plot(t1_new_peak, y1_new_peak, 'ro', label='FT First Peak')
    if t2_new_peak is not None:
        plt.plot(t2_new_peak, y2_new_peak, 'ro', label='FSR First Peak')
    
    plt.tight_layout()
    plt.show()
    
    return df1, df2

def load_baseline():
    '''
    finalize and save the fully interpolated and aligned dfs
    '''
    ft_data = load_csv('FT_test_data.csv')
    fsr_data = load_csv('FSR_test_data.csv')
    fsr_data.rename(columns={'Timestamp': 'time'}, inplace=True)
    fsr_data.rename(columns={'FSR 1 (mV)': 'data'}, inplace=True)
    ft_data.rename(columns={'ft_time': 'time'}, inplace=True)
    ft_data.rename(columns={'ft': 'data'}, inplace=True)
    ft_data, fsr_data = preprocess_align(ft_data, fsr_data, 'time', 'data')

    ft_interp = np.interp(fsr_data['time'], ft_data['time'], ft_data['data'])
    fsr = fsr_data['data'] 
    # ft_data.to_csv(f"{filename}_ft_aligned.csv", index=False)
    # fsr_data.to_csv(f"{filename}_fsr_aligned.csv", index=False)
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
    a, b, k = log_linear_regression(fsr, ft)
    poly3_coeff = poly3_regression(fsr, ft)
    poly5_coeff = poly5_regression(fsr, ft)

    # a0, a1, a2, a3 = poly3_coeff
    # model_p3 = {'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3}
    with open("poly3_model_" + filename + ".pickle", 'wb') as handle:
        # pickle.dump(model_p3, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(poly3_coeff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    linear_fit = m * fsr + c
    log_fit = log_func(fsr, a, b, k)
    poly3_fit = np.polyval(poly3_coeff, fsr)
    poly5_fit = np.polyval(poly5_coeff, fsr)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[0, 0].plot(fsr, linear_fit, label='Linear Fit', color='red')
    axs[0, 0].set_title('Linear Fit')
    axs[0, 0].set_xlabel('FSR Signal')
    axs[0, 0].set_ylabel('FT Signal')
    axs[0, 0].legend()

    axs[0, 1].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[0, 1].plot(fsr, log_fit, label='Log Fit', color='green')
    axs[0, 1].set_title('Log Fit')
    axs[0, 1].set_xlabel('FSR Signal')
    axs[0, 1].set_ylabel('FT Signal')
    axs[0, 1].legend()

    axs[1, 0].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[1, 0].plot(fsr, poly3_fit, label='Poly3 Fit', color='blue')
    axs[1, 0].set_title('Poly3 Fit')
    axs[1, 0].set_xlabel('FSR Signal')
    axs[1, 0].set_ylabel('FT Signal')
    axs[1, 0].legend()

    df = pd.DataFrame({
    'FSR Signal': fsr,
    'Actual FT': ft,
    'Predicted FT': poly3_fit
    })
    pd.set_option('display.max_rows', None) 
    # Print the entire DataFrame (all rows)
    print('help', df)

    axs[1, 1].scatter(fsr, ft, label='Raw Data', alpha=0.7, marker='.')
    axs[1, 1].plot(fsr, poly5_fit, label='Poly5 Fit', color='purple')
    axs[1, 1].set_title('Poly5 Fit')
    axs[1, 1].set_xlabel('FSR Signal')
    axs[1, 1].set_ylabel('FT Signal')
    axs[1, 1].legend()

    plt.savefig("diff_fit_models.png")
    plt.tight_layout()
    plt.show()

    print()
    return

def main():
    ft_data, fsr_data = load_baseline()
    # print(ft_data, fsr_data)
    calibration_curve(ft_data, fsr_data)

if __name__ == '__main__':
    main()
