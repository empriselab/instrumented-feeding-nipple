#Purpose: to be run directly after the ReadfromSensorsOnly.py script. Takes a csv file and translates the voltages to an estimate force before plotting it.
#Be sure to update lines 12-13 with the parameters from the calibration function of this specific sensor.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.patches as patches

#change to the parameters from calibration test trials. Equation is in the form y=ae^(bx), where x is the voltage output 
a = 2.69E01
b = 7.05E-04

#change to the csv file path generated from ReadFromSensorsOnly.py
data = pd.read_csv("Desktop/Moo/Milk/FieldTest2.csv")

a_string = "{:.2e}".format(a)
b_string = "{:.2e}".format(b)
upper_equation = "y=" + str(a_string) + "exp(" + str(b_string) + "x)"

#Extracts voltage output from csv
x_data = data["Voltage (mV)"].values

#applies calibration function to the voltages from the csv file
data['Voltage (mV)'] = a * np.exp(b * data['Voltage (mV)'])

#takes the rolling average in the data after exponential curve fit is applied to smooth out data
window_size = 10
data['Voltage (mV)'] = data['Voltage (mV)'].rolling(window=10).mean()

#plot it all
plt.figure(figsize=(5, 10))
ax = plt.gca()

# Customize the axis spines
ax.spines['top'].set_linewidth(2)    # Thickness of the top spine
ax.spines['right'].set_linewidth(2)  # Thickness of the right spine
ax.spines['bottom'].set_linewidth(2) # Thickness of the bottom spine
ax.spines['left'].set_linewidth(2)   # Thickness of the left spine

# Remove upper and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tick_params(axis='both', which='both', width=2, length=8)  # Set the width of tick marks
plt.xticks(fontsize=20)  # Set the font size of tick labels on the x-axis
plt.yticks(fontsize=20)  # Set the font size of tick labels on the y-axis

# Set global font size for legend
plt.rcParams['legend.fontsize'] = 20

plt.plot(np.roll(data['Upper Taxel Voltage (mV)']), linewidth=4, label='Healthy Bite')
plt.xlabel('Time (ms)', fontsize = 25)
plt.ylabel('Force (N)', fontsize = 25)
plt.xlim(0,500)
plt.ylim(0,30)
plt.legend()
plt.show()