//Sketch uploaded to Arduino Nano 33 IOT to read voltages from the positive and negative terminals of the output voltage. Before starting, be sure to 
//calibrate the bounds for the AnalogRead and their respective voltages (lines 21-35) using a multimeter and lines 45-47. The lower bounds are defined as 
//when no force is applied to the sensors and the upper bounds are defined as when the force applied results in a voltage of ~3.3V.



//pin declarations for the sensors
int pos_Vout = 14;
int neg_Vout = 15;
int c = 0;

// Number of samples to collect before finding the peak
const int numSamples = 3; 

// Variable to store the peak value
int peakValue = 0;

// Variable to store the index of the peak
int peakIndex = 0;

//analogRead bounds for pin 14
int lowerBoundAnalogRead_14 = 94;
int upperBoundAnalogRead_14 = 950;

//voltage bounds for pin 14
int lowerBoundVoltage_14 = 300;
int upperBoundVoltage_14 = 3000;

//analogRead bounds for pin 15
int lowerBoundAnalogRead_15 = 170;
int upperBoundAnalogRead_15 = 1023;

//voltage bounds for pin 15
int lowerBoundVoltage_15 = 10;
int upperBoundVoltage_15 = 3300;

void setup() {
  Serial.begin(9600);  // Initialize the serial communication
  pinMode(pos_Vout, INPUT);
  pinMode(neg_Vout, INPUT);
}

void loop() {
  //Uncomment the next three lines for bound calibration
  //Serial.println(pos_Vout);
  //Serial.println(neg_Vout);
  //delay(5);

  
  long SumSensor = 0;  // Variable to store the sum of the samples
  long sumforceReading = 0;

  // Read and sum the analog samples
  for (int i = 0; i < numSamples; i++) {
    float SumSensor_upper = map(analogRead(pos_Vout), lowerBoundAnalogRead_14, upperBoundAnalogRead_14, lowerBoundVoltage_14, upperBoundAnalogRead_14); 
    float SumSensor_lower = map(analogRead(neg_Vout), lowerBoundAnalogRead_15, upperBoundAnalogRead_15, lowerBoundVoltage_15, upperBoundVoltage_15);
    float difference = SumSensor_upper - SumSensor_lower;
    sumforceReading += difference;
  }

  // Calculate the average
  float average = (float)sumforceReading / numSamples;

  //print to the serial monitor for python to csv converter
  Serial.print(average);

}
