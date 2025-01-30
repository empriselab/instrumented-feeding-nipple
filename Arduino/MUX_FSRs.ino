// Pin declarations for the sensors and MUX control
int pos_Vout = A0; // Assume the multiplexer output is connected to A0
int neg_Vout = A1;
const int muxS0 = 2; // Multiplexer control pins
const int muxS1 = 3;
const int muxS2 = 4;
// const int muxS3 = 5;

// Number of FSRs and MUX channels
const int numFSRs = 16;

// Array to store offsets for zeroing out readings
float offset[numFSRs] = {0};

// Number of samples to collect for calibration and averaging
const int numSamples = 10;

void setup() {
  Serial.begin(9600);  // Initialize the serial communication

  // Set MUX control pins as output
  pinMode(muxS0, OUTPUT);
  pinMode(muxS1, OUTPUT);
  pinMode(muxS2, OUTPUT);
  pinMode(muxS3, OUTPUT);

  // Calibrate offsets
  Serial.println("Calibrating offsets...");
  calibrateOffsets();
  Serial.println("Calibration complete!");
}

void loop() {
  for (int fsrIndex = 0; fsrIndex < numFSRs; fsrIndex++) {
    // Set MUX to the correct channel for the current FSR
    setMuxChannel(fsrIndex);
    delay(5); // Allow time for MUX to settle

    // Initialize variables for averaging
    long sumForceReading = 0;

    // Read and sum samples
    for (int i = 0; i < numSamples; i++) {
      float sensorUpper = analogRead(pos_Vout);
      float sensorLower = analogRead(neg_Vout);
      sumForceReading += (sensorUpper - sensorLower);
    }

    // Calculate the average and apply offset
    float average = (float)sumForceReading / numSamples - offset[fsrIndex];

    // Print the reading for the current FSR
    Serial.print("FSR ");
    Serial.print(fsrIndex);
    Serial.print(": ");
    Serial.println(average);
  }

  delay(100); // Adjust delay as needed for reading frequency
}

// Function to calibrate offsets
void calibrateOffsets() {
  for (int fsrIndex = 0; fsrIndex < numFSRs; fsrIndex++) {
    // Set MUX to the correct channel for the current FSR
    setMuxChannel(fsrIndex);
    delay(5); // Allow time for MUX to settle

    // Initialize variables for averaging
    long sumForceReading = 0;

    // Read and sum samples
    for (int i = 0; i < numSamples; i++) {
      float sensorUpper = analogRead(pos_Vout);
      float sensorLower = analogRead(neg_Vout);
      sumForceReading += (sensorUpper - sensorLower);
    }

    // Calculate the average and store as offset
    offset[fsrIndex] = (float)sumForceReading / numSamples;

    Serial.print("Offset for FSR ");
    Serial.print(fsrIndex);
    Serial.print(": ");
    Serial.println(offset[fsrIndex]);
  }
}

// Function to set the MUX channel
void setMuxChannel(int channel) {
  digitalWrite(muxS0, channel & 0x01);
  digitalWrite(muxS1, (channel >> 1) & 0x01);
  digitalWrite(muxS2, (channel >> 2) & 0x01);
  digitalWrite(muxS3, (channel >> 3) & 0x01);
}

