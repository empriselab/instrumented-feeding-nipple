Instrumented-Feeding-Nipple
by Zoe Chen

Status as of Dec 10, 2023 

Summary:
Weaker biting forces correlate to diseased calves, especially those under six months suffering from diarrhea. The planned final product is a sensor that fits within current calf feeding systems to record the bite force of calves during the suckling process –  in hopes of providing another illness indicator that is safer and more convenient for caregivers. The product must be rugged enough to withstand variations in temperature, exposure to liquids, and over fourteen thousand bites over its lifetime – all while maintaining a soft exterior that is comfortable for the calves to chew on. 

The current iteration of the sensor is modeled after the feeding nipples attached to portable calf milk feeding bottles. Two force resistive sensors (FSRs) are encapsulated within two silicon layers and wired to a Wheatstone bridge. The output voltage is converted to a force via a calibration function in the form of an exponential growth function. All voltage-to-force translations are done on an Arduino Nano 33 IOT in real-time. The total cost of the sensor is $20.36, excluding the cost of the microcontroller.  

We have validated that the design’s calibration function adequately fits the industry standard. Comparisons between the developed sensor and the ground truth force have shown that the sensor accurately and precisely measures the true force up to 50N, with an error of ±3.09N. 

Preliminary field testing found that healthy bites (N=160) had an average force of 14.97N with a standard deviation of 4.30N, while weak bites (N=221) had an average force of 4.55N with a standard deviation of 2.35N. 

