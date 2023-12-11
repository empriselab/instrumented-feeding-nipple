#Purpose: Calculates the calibration function from the sensor with the ground truth force (ATI Net F/T sensor) in the form of an exponential growth function.
#There are two parts to this script:
#Part 1: Combines the csv files containing the voltage readings and the force outputs from the ATI Net F/T sensor into one dataframe
#Part 2: fits the combined data into an exponential curve function and plots the estimated force from the sensor with the ground truth force

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


#---------------------------------------------------------   PART 1: COMBINE CSVS   ---------------------------------------------------------

# Replace these with your actual file paths
voltage_csv_path = 'Desktop/Moo/TaxelData/Callibration/Upper_t1.csv'
force_csv_path = 'Desktop/Moo/RosBagExtraction/testbag_2023-11-12-15-11-09/ft_sensor_netft_data.csv'

# Define the exponential function for the line of best fit
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Read data from the first CSV file
df1 = pd.read_csv(voltage_csv_path)['Voltage (mV)']

# Read data from the second CSV file and select the 'Fz' column
df2 = pd.read_csv(force_csv_path)['fz']

# Create a new DataFrame with the desired columns
result_df = pd.DataFrame({'Voltage (mV)': df1, 'Force (N)': df2})
result_df = result_df.dropna(subset=[result_df.columns[1]])

# Round the 'Force (N)' column to the nearest integer and take positive force
result_df['Force (N)'] = result_df['Force (N)'].round().astype(int)
result_df['Force (N)'] = result_df['Force (N)'] * -1

# Group the data by the 'Force (N)' column and calculate the mean of 'Voltage (mV)' in each group
grouped_data = result_df.groupby('Force (N)')['Voltage (mV)'].mean().reset_index()




#---------------------------------------------------------   PART 2: FIT TO CALIBRATION FUNCTION    ---------------------------------------------------------

#split combined data by columns into their own dataframes
x_data = grouped_data["Voltage (mV)"].values
y_data = grouped_data["Force (N)"].values


#guess values for the a and b parameters for the callibration function
a = 2.81E01
b = 6.69E-04
initial_guess = [a, b]

# Fit the data to the exponential function
params, covariance = curve_fit(exponential_func, x_data, y_data, p0=initial_guess)

# Extract the parameters 'a' and 'b' from the fit
a, b = params

# Calculate the fitted Y values
y_fit = exponential_func(x_data, a, b)

# Calculate R^2 value
r_squared = r2_score(y_data, y_fit)

# Print the R^2 value and a and b as a string for equation 
# print(f"R^2 Value: {r_squared}")
a_string = "{:.2e}".format(a)
b_string = "{:.2e}".format(b)
equation = "y=" + str(a_string) + "exp(" + str(b_string) + "x)" + " & r2 =" + str(round(r_squared, 2))

#takes the rolling average in the data after exponential curve fit is applied to smooth out data
grouped_data['Voltage (mV)'] = a * np.exp(b * grouped_data['Voltage (mV)'])
window_size = 2500
grouped_data['Voltage (mV)'] = grouped_data['Voltage (mV)'].rolling(window=window_size).mean()
grouped_data['Force (N)'] = grouped_data['Force (N)'].rolling(window=window_size).mean()

#plot it all
plt.figure(figsize=(10, 4))
ax = plt.gca()

# Customize the axis spines
ax.spines['top'].set_linewidth(3)    # Thickness of the top spine
ax.spines['right'].set_linewidth(3)  # Thickness of the right spine
ax.spines['bottom'].set_linewidth(3) # Thickness of the bottom spine
ax.spines['left'].set_linewidth(3)   # Thickness of the left spine

# Remove upper and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tick_params(axis='both', which='both', width=3, length=8)  # Set the width of tick marks
plt.xticks(fontsize=30)  # Set the font size of tick labels on the x-axis
plt.yticks(fontsize=30)  # Set the font size of tick labels on the y-axis
plt.rcParams['legend.fontsize'] = 20
result_array = np.arange(1, len(grouped_data['Voltage (mV)'])+1)
plt.plot(np.roll(grouped_data['Voltage (mV)']), linewidth=6,label='Estimated')
plt.plot(np.roll(grouped_data['Force (N)']), linewidth=6,label='True')

plt.ylim(0,50)
plt.xlabel('Time (ms)', fontsize = 30)
plt.ylabel('Force (N)', fontsize = 30)

plt.legend()
plt.show()

