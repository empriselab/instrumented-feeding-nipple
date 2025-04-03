import serial
import csv
import time

def initialize_serial(port, baud_rate):
    """Initialize serial communication with Arduino."""
    ser = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(2)  # Allow time for the connection to initialize
    return ser

def write_data(ser, csv_filename, num_channels=13):
    """Read data from Arduino and write to CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Header: Timestamp + 12 FSRs + Temperature
        header = ['Timestamp'] + [f'FSR {i+1}' for i in range(num_channels - 1)] + ['Temperature (C)']
        csv_writer.writerow(header)

        try:
            while True:
                data = ser.readline().decode(errors='ignore').strip()

                if data:
                    timestamp = int(time.time() * 1_000_000)
                    readings = data.split()

                    if len(readings) == num_channels:
                        try:
                            fsr_values = [int(val) for val in readings[:-1]]
                            temp_value = float(readings[-1])
                            print(f'Timestamp: {timestamp}, FSRs: {fsr_values}, Temp: {temp_value:.2f} °C')
                            csv_writer.writerow([timestamp] + fsr_values + [temp_value])
                        except ValueError:
                            print(f"Warning: Invalid numeric value in data: {readings}")
                    else:
                        print(f"Warning: Incomplete data ({len(readings)} values): {data}")

        except KeyboardInterrupt:
            print('Data collection stopped.')
        finally:
            ser.close()

if __name__ == "__main__":
    serial_port = 'COM4'  # Change this to your port
    baud_rate = 9600
    csv_filename = 'FSR_with_temp.csv'  # Adjust the filename as needed

    arduino_serial = initialize_serial(serial_port, baud_rate)

    try:
        write_data(arduino_serial, csv_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arduino_serial.close()
