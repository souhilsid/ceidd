/*
 * TCS3200 Color Sensor -> Calibrated RGB output
 *
 * Serial output (parsed by RYB_SDL.py):
 *   RGB: R=<0..255> G=<0..255> B=<0..255> | Color: <label>
 *
 * Optional serial commands:
 *   w = capture WHITE reference
 *   b = capture BLACK reference
 *   p = print calibration
 */

#include <Arduino.h>
#include <math.h>

// -------------------- Pins --------------------
#define s0 8
#define s1 9
#define s2 11
#define s3 10
#define out 12
#define TCS_LED 7

// -------------------- Sensor read config --------------------
const int READ_SAMPLES = 10;
const unsigned long PULSE_TIMEOUT = 50000UL;
const int FILTER_SETTLE_MS = 3;
const int LOOP_DELAY_MS = 500;

// 20% scaling: stable default for most TCS3200 boards.
const bool USE_20_PERCENT_SCALING = true;

// -------------------- Initial calibration --------------------
// Update with serial commands on your setup.
int redWhite = 35, redBlack = 280;
int greenWhite = 30, greenBlack = 320;
int blueWhite = 40, blueBlack = 300;

// -------------------- Classification thresholds (label only) --------------------
const int DARK_MAX_RGB = 25;
const int CLEAR_MIN_BRIGHT = 120;
const float CLEAR_MAX_SAT = 0.10f;

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

int clampi(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// Convert raw pulse width to 0..255 intensity.
int pulseToRGB(int raw, int whitePulse, int blackPulse) {
  if (whitePulse == blackPulse) return 0;

  float x;
  if (whitePulse < blackPulse) {
    x = (float)(blackPulse - raw) / (float)(blackPulse - whitePulse);
  } else {
    x = (float)(raw - blackPulse) / (float)(whitePulse - blackPulse);
  }

  x = clampf(x, 0.0f, 1.0f);
  return clampi((int)roundf(255.0f * x), 0, 255);
}

int readPulseAveraged(int s2State, int s3State, int n = READ_SAMPLES) {
  digitalWrite(s2, s2State);
  digitalWrite(s3, s3State);
  delay(FILTER_SETTLE_MS);

  unsigned long sum = 0;
  for (int i = 0; i < n; i++) {
    unsigned long p = pulseIn(out, LOW, PULSE_TIMEOUT);
    if (p == 0) p = PULSE_TIMEOUT;
    sum += p;
  }
  return (int)(sum / (unsigned long)n);
}

String labelFromRGB(int r, int g, int b) {
  float maxVal = (float)max(r, max(g, b));
  float minVal = (float)min(r, min(g, b));
  float delta = maxVal - minVal;
  float sat = (maxVal <= 1e-6f) ? 0.0f : (delta / maxVal);

  if ((int)maxVal < DARK_MAX_RGB) return "No Liquid / Dark";
  if (maxVal > CLEAR_MIN_BRIGHT && sat < CLEAR_MAX_SAT) return "Clear (Water)";

  if (r >= g && r >= b) return "Red-dominant";
  if (g >= r && g >= b) return "Green-dominant";
  return "Blue-dominant";
}

void printCalibration() {
  Serial.println(F("[CAL] Current calibration (white, black):"));
  Serial.print(F("[CAL] R: ")); Serial.print(redWhite); Serial.print(F(", ")); Serial.println(redBlack);
  Serial.print(F("[CAL] G: ")); Serial.print(greenWhite); Serial.print(F(", ")); Serial.println(greenBlack);
  Serial.print(F("[CAL] B: ")); Serial.print(blueWhite); Serial.print(F(", ")); Serial.println(blueBlack);
}

void captureWhite() {
  int rawR = readPulseAveraged(LOW, LOW, 20);
  int rawB = readPulseAveraged(LOW, HIGH, 20);
  int rawG = readPulseAveraged(HIGH, HIGH, 20);

  redWhite = rawR;
  greenWhite = rawG;
  blueWhite = rawB;

  Serial.println(F("[CAL] Captured WHITE reference."));
  printCalibration();
}

void captureBlack() {
  int rawR = readPulseAveraged(LOW, LOW, 20);
  int rawB = readPulseAveraged(LOW, HIGH, 20);
  int rawG = readPulseAveraged(HIGH, HIGH, 20);

  redBlack = rawR;
  greenBlack = rawG;
  blueBlack = rawB;

  Serial.println(F("[CAL] Captured BLACK reference."));
  printCalibration();
}

void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == 'w' || c == 'W') {
      captureWhite();
    } else if (c == 'b' || c == 'B') {
      captureBlack();
    } else if (c == 'p' || c == 'P') {
      printCalibration();
    }
  }
}

void setup() {
  pinMode(s0, OUTPUT);
  pinMode(s1, OUTPUT);
  pinMode(s2, OUTPUT);
  pinMode(s3, OUTPUT);
  pinMode(out, INPUT);

  pinMode(TCS_LED, OUTPUT);
  digitalWrite(TCS_LED, HIGH);

  Serial.begin(9600);

  if (USE_20_PERCENT_SCALING) {
    digitalWrite(s0, HIGH);
    digitalWrite(s1, LOW);
  } else {
    digitalWrite(s0, LOW);
    digitalWrite(s1, HIGH);
  }

  delay(200);
  Serial.println(F("[INFO] TCS3200 RGB sketch started."));
  Serial.println(F("[INFO] Serial commands: w=white, b=black, p=print calib"));
  printCalibration();
}

void loop() {
  handleSerialCommands();

  int rawR = readPulseAveraged(LOW, LOW);    // Red filter
  int rawB = readPulseAveraged(LOW, HIGH);   // Blue filter
  int rawG = readPulseAveraged(HIGH, HIGH);  // Green filter

  int r = pulseToRGB(rawR, redWhite, redBlack);
  int g = pulseToRGB(rawG, greenWhite, greenBlack);
  int b = pulseToRGB(rawB, blueWhite, blueBlack);

  String label = labelFromRGB(r, g, b);

  // Format consumed by RYB_SDL.py
  Serial.print("RGB: R=");
  Serial.print(r);
  Serial.print(" G=");
  Serial.print(g);
  Serial.print(" B=");
  Serial.print(b);
  Serial.print(" | Color: ");
  Serial.println(label);

  delay(LOOP_DELAY_MS);
}
