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


def load_csv(file_path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def find_max_peak(time, signal):
    """Find the maximum peak in the signal and return its time, and value."""
    peaks, _ = find_peaks(signal)
    
    if len(peaks) > 0:
        # Find the peak with the highest value
        max_peak_idx = peaks[signal[peaks].argmax()]
        max_peak_time = time[max_peak_idx]
        max_peak_value = signal[max_peak_idx]
        
        print(f"Max peak detected at time: {max_peak_time}, Value: {max_peak_value}")
        return max_peak_time, max_peak_value
    else:
        print("No peak detected.")
        return None, None
        
        

def preprocess_align(df1, df2, time_col, signal_col):
    """Align df2 to df1 based on the first detected peak. Plots before and after changing"""
    
    # aligning baseline to 0
    df1['data'] = abs(df1['data'])
 
    min_index = df2['data'].idxmin()
    min_time = df2['time'].iloc[min_index]
    min_value = df2['data'].iloc[min_index]
    # print(f"\nMinimum FSR value found at time: {min_time}, Value: {min_value}")
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
    
    # aligning fsr peaks with the ft peak
    # TODO: might find a more efficient way to align than just comparing the maxes
    
    t1_peak, y1_peak = find_max_peak(df1[time_col], df1[signal_col])
    t2_peak, y2_peak = find_max_peak(df2[time_col], df2[signal_col])
    
    if t1_peak is None or t2_peak is None:
        print("Could not detect peaks in one or both signals.")
        return df1, df2
    
    shift = t1_peak - t2_peak
    
    df2[time_col] += shift
    # scaling the fsr down to the force torque sensor, might need to find a more better way to do this right now
    # factor = y1_peak / y2_peak
    # df2[signal_col] *= factor
    factor = y2_peak / y1_peak

    df1[signal_col] *= factor

    
    min_ft_time = df1[time_col].min()
    max_ft_time = df1[time_col].max()

    # Keep only the rows where FSR time is within FT time range
    df2 = df2[(df2[time_col] >= min_ft_time) & (df2[time_col] <= max_ft_time)]

    
    # Plot aligned signals
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
    loads in the csv files and renames the columns that they are working with. returns such that the values are now aligned 
    from the offset of y-value 0 and also aligned with the peaks
    
    '''
    
    ft_data = load_csv('FT_test_data.csv')
    fsr_data = load_csv('FSR_test_data.csv')
    
    fsr_data.rename(columns={'Timestamp': 'time'}, inplace=True)
    fsr_data.rename(columns={'FSR 1 (mV)': 'data'}, inplace=True)
    ft_data.rename(columns={'ft_time': 'time'}, inplace=True)
    ft_data.rename(columns={'ft': 'data'}, inplace=True)
    # ft is always left and fsr is always right
    ft_data, fsr_data = preprocess_align(ft_data, fsr_data, 'time', 'data')

    return ft_data, fsr_data



def calibration_curve(ft_data, fsr_data):
    """
    Fits multiple calibration curve to map FSR values to FT values.
    """
    ft = np.interp(fsr_data['time'], ft_data['time'], ft_data['data'])
    # print(len(ft_interpolated))
    
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
    
    m, c = linear_regression(ft, fsr_data['data'])
    a, b, k = log_linear_regression(ft, fsr_data['data'])
    poly3_coeff = poly3_regression(ft, fsr_data['data'])
    poly5_coeff = poly5_regression(ft, fsr_data['data'])
    
    
    # creating the curves
    linear_fit = m * ft + c
    log_fit = log_func(ft, a, b, k)
    poly3_fit = np.polyval(poly3_coeff, ft)
    poly5_fit = np.polyval(poly5_coeff, ft)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # linear fit
    axs[0, 0].scatter(ft, fsr_data['data'], label='Raw Data', alpha=0.7, marker='.')
    axs[0, 0].plot(ft, linear_fit, label='Linear Fit', color='red')
    axs[0, 0].set_title('Linear Fit')
    axs[0, 0].set_xlabel('FT Signal')
    axs[0, 0].set_ylabel('FSR Signal')
    axs[0, 0].legend()

    # log fit
    axs[0, 1].scatter(ft, fsr_data['data'], label='Raw Data', alpha=0.7, marker='.')
    axs[0, 1].plot(ft, log_fit, label='Log Fit', color='green')
    axs[0, 1].set_title('Log Fit')
    axs[0, 1].set_xlabel('FT Signal')
    axs[0, 1].set_ylabel('FSR Signal')
    axs[0, 1].legend()

    # degree 3 poly fit
    axs[1, 0].scatter(ft, fsr_data['data'], label='Raw Data', alpha=0.7, marker='.')
    axs[1, 0].plot(ft, poly3_fit, label='Poly3 Fit', color='blue')
    axs[1, 0].set_title('Poly3 Fit')
    axs[1, 0].set_xlabel('FT Signal')
    axs[1, 0].set_ylabel('FSR Signal')
    axs[1, 0].legend()

    # degree 5 polyn fit
    axs[1, 1].scatter(ft, fsr_data['data'], label='Raw Data', alpha=0.7, marker='.')
    axs[1, 1].plot(ft, poly5_fit, label='Poly5 Fit', color='purple')
    axs[1, 1].set_title('Poly5 Fit')
    axs[1, 1].set_xlabel('FT Signal')
    axs[1, 1].set_ylabel('FSR Signal')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    return



def main():
    """
    runs in loading the data which also aligns, then using the aligned data to 
    create calibration curves.
    """
    ft_data, fsr_data = load_baseline()
    calibration_curve(ft_data, fsr_data)


