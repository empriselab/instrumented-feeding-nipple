# Instrumented Feeding Nipple
Below details how to calibrate and get voltage estimations from the constructed sensors.


## Calibration 

### Step 1: Upload the Arduino & start the ATI NET F/T sensor
Upload the `FeedingNippleArduino.ino` to your Arduino. Calibrate the upper and lower bounds of the analogRead and its corresponding voltages to ensure accurate readings. Be sure the readings are printing properly on the serial monitor. Ensure the serial monitor is closed throughout the duration of data collection.

### Step 2: Start ATI NET F/T sensor
Turn on the ATI NET F/T sensor. Open your terminal. Don't forget to run `source devel/setup.bash` everytime you open a new terminal in the root of your workspace to source it.

### Step 3: Begin collecting data
Run the bash script `REPLACE REPLACE REPLACE` to begin collecting data from the ATI NET F/T sensor.

### Step 4: Convert Rosbag into CSV file
Place the `extract_forque.py` script in the same folder as your rosbag files. Run the `extract_forque.py` script. This converts the rosbag into a CSV in its own individual folder.

### Step 4: Combine the Force and Voltage Readings using CurveFit.py
Run this script with two CSVs files as the input - the voltage readings from the Arduino and the rosbag force readings. This script will combine the two csv files and calculate a calibration function in the form y = ae^(bx).


## Post-Calibration 
After finding the calibration function, input the a and b parameters into `TranslateV2F.py`. Whenever performing future tests or in field tests, have `FeedingNippleArduino.ino` running on the Arduino before running `TranslateV2F.py`.




