# Instrumented Feeding Nipple

## CAD Files
All CAD files were printed using 5% rectilinear infill and no supports.

## Calibration 
The following must be run within the `feeding_nipple_ws` file.

### Step 1: Upload the Arduino code
Upload the `FeedingNippleArduino.ino` to your Arduino. Calibrate the upper and lower bounds of the analogRead and its corresponding voltages to ensure accurate readings. Be sure the readings are printing properly on the serial monitor. Ensure the serial monitor is closed throughout the duration of data collection.

### Step 2: Start ATI NET F/T sensor
Turn on the ATI NET F/T sensor. Open your terminal. Remember to run `source devel/setup.bash` everytime you open a new window in the root of your workspace to source it.

To initialize the core, type `roscore`.

In a new window, launch the sensor topic by typing `roslaunch netft_rdt_driver ft_sensor.launch ip_address:=192.168.1.6 bias:=True rate:=1000`.

To ensure the topic has been properly launched, type the command `rostopic echo /ft_sensor/netft_data`. This will allow you to monitor the readings from the F/T sensor in real time.

Calibrate the F/T sensor by holding it in the air and typing the command `rosservice call /ft_sensor/bias_cmd "cmd: 'bias'"`.


### Step 3: Begin collecting data
Run the bash script `callibration.sh` to begin collecting data from the ATI NET F/T sensor.

### Step 4: Convert Rosbag into CSV file
Place the `extract_forque.py` script in the same folder as your rosbag files. Run the `extract_forque.py` script. This converts the rosbag into a CSV in its own individual folder.

### Step 4: Combine the Force and Voltage Readings using CurveFit.py
Run this script with two CSVs files as the input - the voltage readings from the Arduino and the rosbag force readings. This script will combine the two csv files and calculate a calibration function in the form y = ae^(bx).


## Post-Calibration 
After finding the calibration function, input the a and b parameters into `TranslateV2F.py`. Whenever performing future tests or in field tests, have `FeedingNippleArduino.ino` running on the Arduino before running `TranslateV2F.py`.
