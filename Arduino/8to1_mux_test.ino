// Pin Definitions
const int muxSigPin = A0;       // Signal pin connected to the MUX output
const int s0Pin = 4;            // MUX control pin A
const int s1Pin = 3;            // MUX control pin B
const int s2Pin = 2;            // MUX control pin C

// Number of MUX channels
const int numChannels = 8;

// FSR readings array
int fsrReadings[numChannels];

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set control pins as outputs
  pinMode(s0Pin, OUTPUT);
  pinMode(s1Pin, OUTPUT);
  pinMode(s2Pin, OUTPUT);

  // Initialize control pins to LOW
  digitalWrite(s0Pin, LOW);
  digitalWrite(s1Pin, LOW);
  digitalWrite(s2Pin, LOW);
}

void loop() {
  unsigned long timestamp = micros();

  // Read all 8 channels
  for (int channel = 0; channel < numChannels; channel++) {
    selectMuxChannel(channel);                // Select MUX channel
    fsrReadings[channel] = analogRead(muxSigPin); // Read the FSR value
  }

  // Print FSR readings
  for (int i = 0; i < numChannels; i++) {
    Serial.print(fsrReadings[i]);
    Serial.print(' ');
  }
  Serial.println();

  delay(200);
}

// Function to select MUX channel
void selectMuxChannel(int channel) {
  digitalWrite(s0Pin, (channel & 0x01)); // Least significant bit
  digitalWrite(s1Pin, (channel & 0x02) >> 1); // Second bit
  digitalWrite(s2Pin, (channel & 0x04) >> 2); // Most significant bit
}
