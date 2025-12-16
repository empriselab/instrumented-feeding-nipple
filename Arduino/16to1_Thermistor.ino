#include <math.h>

// --- MUX FSR SETUP ---
const int muxSigPin = A0;       // MUX signal pin (connected to FSRs)
const int s0Pin = 2;            // MUX control pins
const int s1Pin = 3;
const int s2Pin = 4;
const int s3Pin = 5;

const int numFSRChannels = 12;  // Number of FSRs
int fsrReadings[numFSRChannels];

// --- Thermistor SETUP ---
#define THERMISTORPIN A1  
#define THERMISTORNOMINAL 10000      
#define TEMPERATURENOMINAL 25   
#define NUMSAMPLES 5
#define BCOEFFICIENT 3950
#define SERIESRESISTOR 10000    

int samples[NUMSAMPLES];

void setup() {
  Serial.begin(9600);

  // MUX control pins
  pinMode(s0Pin, OUTPUT);
  pinMode(s1Pin, OUTPUT);
  pinMode(s2Pin, OUTPUT);
  pinMode(s3Pin, OUTPUT);

  // Initialize MUX control lines
  digitalWrite(s0Pin, LOW);
  digitalWrite(s1Pin, LOW);
  digitalWrite(s2Pin, LOW);
  digitalWrite(s3Pin, LOW);

  analogReference(AR_EXTERNAL);
}

void loop() {
  // --- Read FSRs ---
  for (int channel = 0; channel < numFSRChannels; channel++) {
    selectMuxChannel(channel);
    fsrReadings[channel] = analogRead(muxSigPin);
  }

  // --- Read Thermistor ---
  float average = 0;
  for (int i = 0; i < NUMSAMPLES; i++) {
    samples[i] = analogRead(THERMISTORPIN);
    delay(2);
  }
  for (int i = 0; i < NUMSAMPLES; i++) {
    average += samples[i];
  }
  average /= NUMSAMPLES;

  // Convert ADC reading to resistance
  average = 1023.0 / average - 1.0;
  average = SERIESRESISTOR / average;

  float steinhart;
  steinhart = average / THERMISTORNOMINAL;
  steinhart = log(steinhart);
  steinhart /= BCOEFFICIENT;
  steinhart += 1.0 / (TEMPERATURENOMINAL + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;

  // --- Print Data in One Line: FSR1 FSR2 ... TempC ---
  for (int i = 0; i < numFSRChannels; i++) {
    Serial.print(fsrReadings[i]);
    Serial.print(" ");
  }
  Serial.println(steinhart);  // temperature in °C
}

// --- Select MUX Channel ---
void selectMuxChannel(int channel) {
  digitalWrite(s0Pin, (channel & 0x01));
  digitalWrite(s1Pin, (channel & 0x02) >> 1);
  digitalWrite(s2Pin, (channel & 0x04) >> 2);
  digitalWrite(s3Pin, (channel & 0x08) >> 3);
}
