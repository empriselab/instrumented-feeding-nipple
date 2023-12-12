//force sensor object declaration
#include "FX29K.h"
FX29K scale(FX29K0, 0010, &Wire);

//pin declarations for the upper sensors
int pos_Vout_upper = 14;
int neg_Vout_upper = 15;

//pin declarations for lower sensors
int pos_Vout_lower = 16;
int neg_Vout_lower = 17;
int c = 0;

const int numSamples = 3;  // Number of samples to collect before finding the peak
int peakValue = 0;  // Variable to store the peak value
int peakIndex = 0;  // Variable to store the index of the peak

void setup() {
  Serial.begin(9600);  // Initialize the serial communication
  pinMode(pos_Vout_upper, INPUT);
  pinMode(neg_Vout_upper, INPUT);
  pinMode(pos_Vout_lower, INPUT);
  pinMode(neg_Vout_lower, INPUT);
//  Wire.begin();
//  scale.tare(1000);
//  delay(1000);
}

void loop() {
  long UppersumSensor = 0;  // Variable to store the sum of the samples
  long LowersumSensor = 0;
  long sumforceReading = 0;

  // Read and sum the analog samples
  for (int i = 0; i < numSamples; i++) {
    float upper_Vout_analog = map(analogRead(pos_Vout_upper), 94, 950, 300, 3000) - map(analogRead(neg_Vout_upper), 170, 1023, 10, 3300);
    UppersumSensor += upper_Vout_analog;
//    float lower_Vout_analog = map(analogRead(pos_Vout_lower), 20, 960, 3, 3000) - map(analogRead(neg_Vout_lower), 0, 1023, 27, 3300);
//    LowersumSensor += lower_Vout_analog;
//    int forceReading = (scale.getPounds()) * 2 * 4.44822;
//    sumforceReading += forceReading;
    //    delay(1);  // Delay between each sample (adjust as needed)
  }

  // Calculate the average
  float avg_uppersensorValue = (float)UppersumSensor / numSamples;
//  float avg_lowersensorValue = (float)LowersumSensor / numSamples;
//  float force = (float)sumforceReading / numSamples;

  //   Print the average to the serial monitor
  //  Serial.print(c);
  //  Serial.print(", ");
  Serial.println(avg_uppersensorValue);
//  Serial.print(",");
//  Serial.print(avg_lowersensorValue);
//  Serial.print(",");
//  Serial.println(map(analogRead(neg_Vout_upper), 170, 1023, 10, 3300));
//  delay(10);  // Delay before the next set of samples (adjust as needed)
}
