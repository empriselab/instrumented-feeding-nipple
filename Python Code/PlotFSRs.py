import serial
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Initialize Serial Communication
def initialize_serial(port, baud_rate):
    """Initialize serial communication with Arduino."""
    ser = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(3) # Wait for the connection to establish
    return ser

# Real-time data buffer
BUFFER_SIZE = 100  # Number of points to show in real-time plot
num_channels = 8   # Number of FSR sensors

# Data buffer for each channel
timestamps = deque(maxlen=BUFFER_SIZE)
fsr_data = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]

# Initialize serial communication
serial_port = 'COM4'  # Change as needed
baud_rate = 9600  # Match with Arduino
csv_filename = 'FSR_Data_1.csv'  # Change for each test run
arduino_serial = initialize_serial(serial_port, baud_rate)
arduino_serial.flush()  # Clears buffer to start fresh


# Initialize plot
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f'FSR {i+1}')[0] for i in range(num_channels)]
ax.set_xlim(0, BUFFER_SIZE)
ax.set_ylim(0, 1023)  # Adjust based on your FSR range
ax.legend()
ax.set_title('Real-time FSR Readings')
ax.set_xlabel('Time')
ax.set_ylabel('FSR Value')

# Update function for animation
def update(frame):
    """Read data from serial, update plot."""
    global timestamps, fsr_data

    # Read data from Arduino
    data = arduino_serial.readline().decode().strip()

    if data:
        print(f"Live Data: {data}")
        try:
            readings = list(map(int, data.split()))
            if len(readings) == num_channels:
                # Append new data
                timestamps.append(time.time())

                for i in range(num_channels):
                    fsr_data[i].append(readings[i])

                # Update plot
                for i in range(num_channels):
                    lines[i].set_data(range(len(fsr_data[i])), list(fsr_data[i]))
                
                ax.relim()
                ax.autoscale_view()
        except ValueError:
            print("Invalid data received:", data)

    return lines

# Animate the plot
ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

# Show the plot
plt.show()

# Close serial connection when done
arduino_serial.close()
