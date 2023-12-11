#Purpose: takes one serial input from arduino in addition to the force sensor and sends to csv for ease of data processing later on
#Run this script after the sketch has been uploaded and is currently running on the Arduino
#When done collecting data, use the command Control + C

import serial
import csv
import time

# Function to initialize serial communication with Arduino
def initialize_serial(port, baud_rate):
    ser = serial.Serial(port, baud_rate, timeout=1)
    return ser

# Function to read data from Arduino and write to CSV file
def write_data(ser, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header to CSV file
        csv_writer.writerow(['Timestamp', 'Voltage (mV)'])
        
        try:
            while True:
                # Read data from Arduino
                data = ser.readline().decode().strip()
                
                if data:
                    # Get current timestamp
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Split the data into individual values
                    voltage = int(data)
                    
                    # Print data to console
                    print(f'Timestamp: {timestamp}, Voltage (mV): {voltage}')
                    
                    # Write data to CSV file
                    csv_writer.writerow([timestamp, voltage])
                    
        except KeyboardInterrupt:
            print('Data collection stopped.')
            ser.close()


#main loop
if __name__ == "__main__":
    # Replace 'COM3' with your Arduino's serial port
    serial_port = '/dev/cu.usbmodem141301'
    baud_rate = 9600 #change to match the baud rate of the Arduino
    csv_filename = 'Test1.csv' # change name for every test run to prevent overwriting

    # Initialize serial communication
    arduino_serial = initialize_serial(serial_port, baud_rate)

    try:
        # Read and save data to CSV file
        write_data(arduino_serial, csv_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the serial port
        arduino_serial.close()