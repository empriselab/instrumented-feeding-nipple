import serial
import csv
import time

def initialize_serial(port, baud_rate):
    """Initialize serial communication with Arduino."""
    ser = serial.Serial(port, baud_rate, timeout=1)
    # Wait for the serial connection to initialize
    time.sleep(2)
    return ser

def write_data(ser, csv_filename, num_channels=12):
    """Read data from Arduino and write to CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Create header with sensor indices
        header = ['Timestamp'] + [f'FSR {i+1}' for i in range(num_channels)]
        csv_writer.writerow(header)

        try:
            while True:
                # Read data from Arduino
                data = ser.readline().decode(errors='ignore').strip()

                if data:
                    # Get current timestamp in microseconds
                    timestamp = int(time.time() * 1_000_000)  # Python's version of micros()

                    # Split the data into individual FSR readings
                    readings = data.split()

                    if len(readings) == num_channels:
                        try:
                            readings = [int(value) for value in readings]
                            # Print data to console
                            print(f'Timestamp: {timestamp}, Readings: {readings}')
                            # Write data to CSV file
                            csv_writer.writerow([timestamp] + readings)
                        except ValueError:
                            print("Warning: Non-integer value encountered.")
                    else:
                        print(f"Warning: Incomplete data ({len(readings)} values): {data}")

        except KeyboardInterrupt:
            print('Data collection stopped.')
        finally:
            ser.close()

if __name__ == "__main__":
    serial_port = 'COM4'  # Change as needed
    baud_rate = 9600      # Match with Arduino
    csv_filename = '/Data/FSR_flat_2mm_s1_exp.csv'

    arduino_serial = initialize_serial(serial_port, baud_rate)

    try:
        write_data(arduino_serial, csv_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arduino_serial.close()
