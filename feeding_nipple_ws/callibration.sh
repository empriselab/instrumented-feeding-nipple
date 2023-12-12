#!/bin/bash


rosbag record -o Trial1_30+ --duration 60 /ft_sensor/netft_data &

python Read1TaxelOnly.py &

wait

echo "all commands are done."
