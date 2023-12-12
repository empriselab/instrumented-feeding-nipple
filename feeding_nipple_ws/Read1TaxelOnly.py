#Purpose: collects data from the Arduino and writes to a csv file. This is the python file run at the same time as the data collection from the FT sensor when callibration.sh is run.

import serial
import csv
import time
import os


# Replace 'COMx' with the appropriate serial port on your system
arduino_port = '/dev/ttyACM0'
baud_rate = 9600

# Open serial connection to Arduino
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Specify the CSV file location
csv_file_path = 'Recent.csv'

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header to CSV file
    csv_writer.writerow(['Time', 'Voltage (mV)'])

    # Record data for 60 seconds
    start_time = time.time()
    while time.time() - start_time < 60:
        # Read data from Arduino
        arduino_data = ser.readline().decode().strip()

        # Get current time
        current_time = time.time() - start_time

        # Write data to CSV file
        csv_writer.writerow([current_time, arduino_data])

# Close serial connection
ser.close()

print(f"Data recording complete. CSV file saved at: {csv_file_path}")
